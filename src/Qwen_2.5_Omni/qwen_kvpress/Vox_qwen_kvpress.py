import os
import sys
import json
import argparse
import warnings
import random
import traceback
import contextlib
import re
from typing import Dict, Any, List

import numpy as np
import torch
import transformers
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report



os.environ['NUMEXPR_MAX_THREADS'] = '64'
random.seed(42)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")


ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "VoxGender_QwenKVPress_Results")
print(f"KV Press config: Compression Ratio={ENV_COMPRESSION_RATIO}, Compression Type={ENV_PRESS_TYPE}")


sys.path.append("/data/to/your/code/Qwen_2.5")
try:
    from modeling_qwen2_5_omni_origin import Qwen2_5OmniForConditionalGeneration
    from processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    QWEN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] Qwen2.5-Omni import failed: {e}")
    QWEN_AVAILABLE = False


KV_PRESS_AVAILABLE = False
try:
    print(f"[Info] Transformers version: {transformers.__version__}")
    try:
        from kvpress import (
            ExpectedAttentionPress, KnormPress, ObservedAttentionPress,
            RandomPress, SnapKVPress, StreamingLLMPress, TOVAPress,
        )
        KV_PRESS_AVAILABLE = True
        print("[Info] KV Press library loaded successfully")
    except ImportError as e:
        print(f"[Warning] KV Press import failed: {e}")
        kvpress_path = os.path.join(os.path.dirname(__file__), '..', 'kvpress')
        if os.path.exists(kvpress_path):
            sys.path.insert(0, kvpress_path)
            from kvpress import (
                ExpectedAttentionPress, KnormPress, ObservedAttentionPress,
                RandomPress, SnapKVPress, StreamingLLMPress, TOVAPress,
            )
            KV_PRESS_AVAILABLE = True
            print("[Info] Loaded KV Press library from local directory successfully")
except Exception as e:
    print(f"[Warning] KV Press library load error: {e}")
    KV_PRESS_AVAILABLE = False



_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648


def parse_args():
    p = argparse.ArgumentParser(description="Vox Gender with Qwen2.5-Omni + KV Press")
    p.add_argument("--model-path", type=str, default="/data/to/your/model/Qwen2.5-Omni-3B")
    p.add_argument("--data-root", type=str, default="/data/to/your/dataset/VoxCeleb/concatenated_audio")
    p.add_argument("--meta-file", type=str, default="gender_id_task_meta.json")
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument("--min-seq-len", type=int, default=128)
    p.add_argument("--no-compress", action="store_true")
    return p.parse_args()


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    return obj


def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        return allocated, reserved
    return 0.0, 0.0


def patch_qwen_for_kvpress(model):
    """Patch Qwen2.5-Omni for rotary_emb, reuse logic from VESUS."""
    try_paths = []
    if hasattr(model, 'thinker'):
        try_paths.append(model.thinker)
        if hasattr(model.thinker, 'model'):
            try_paths.append(model.thinker.model)
    try_paths.append(model)
    base = None
    for cand in try_paths:
        if hasattr(cand, 'model') and hasattr(cand.model, 'layers'):
            base = cand.model
            break
        if hasattr(cand, 'layers'):
            base = cand
            break
    if base is None:
        return False
    if hasattr(base, 'rotary_emb') and base.rotary_emb is not None:
        return True
    if hasattr(base, 'layers') and len(base.layers) > 0:
        first_layer = base.layers[0]
        for attr in ['attn', 'self_attn']:
            if hasattr(first_layer, attr):
                sub = getattr(first_layer, attr)
                if hasattr(sub, 'rotary_emb') and sub.rotary_emb is not None:
                    base.rotary_emb = sub.rotary_emb
                    return True
    try:
        hidden_size = model.config.hidden_size if hasattr(model, 'config') else 0
        num_heads = model.config.num_attention_heads if hasattr(model, 'config') else 1
        head_dim = hidden_size // num_heads if num_heads else 0
        class SimpleRotaryEmbedding:
            def __init__(self, dim, max_position_embeddings=32768, base=10000):
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
                self.base = base
        base.rotary_emb = SimpleRotaryEmbedding(dim=head_dim)
        return True
    except Exception:
        return False


def create_kvpress_adapter(model, press_obj):
    class KVPressAdapter:
        def __init__(self, original_model, press_object):
            self.original_model = original_model
            self.press_object = press_object
        def __enter__(self):
            hooks = []
            try:
                if hasattr(self.original_model, 'thinker') and hasattr(self.original_model.thinker, 'model'):
                    base_model = self.original_model.thinker.model
                    if hasattr(base_model, 'layers'):
                        ok = 0
                        for layer in base_model.layers:
                            attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                            if attn is None:
                                continue
                            if (not hasattr(attn, 'rotary_emb') or getattr(attn, 'rotary_emb', None) is None) and hasattr(base_model, 'rotary_emb'):
                                attn.rotary_emb = base_model.rotary_emb
                            hook = attn.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
                            hooks.append(hook)
                            ok += 1
                        self.hooks = hooks
                        print(f"[KVPressAdapter] Registered hooks: {ok}/{len(base_model.layers)}")
                        return self
                if hasattr(self.original_model, 'model') and hasattr(self.original_model.model, 'layers'):
                    base_model = self.original_model.model
                    ok = 0
                    for layer in base_model.layers:
                        attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                        if attn is None:
                            continue
                        if hasattr(base_model, 'rotary_emb') and (not hasattr(attn, 'rotary_emb') or getattr(attn, 'rotary_emb', None) is None):
                            attn.rotary_emb = base_model.rotary_emb
                        hook = attn.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
                        hooks.append(hook)
                        ok += 1
                    self.hooks = hooks
                    print(f"[KVPressAdapter] Registered hooks: {ok}/{len(base_model.layers)}")
                    return self
                self.hooks = []
                return self
            except Exception as e:
                print(f"[KVPressAdapter] Error: {e}")
                for h in hooks:
                    try:
                        h.remove()
                    except:
                        pass
                self.hooks = []
                return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            removed = 0
            for h in getattr(self, 'hooks', []):
                try:
                    h.remove()
                    removed += 1
                except:
                    pass
            if removed:
                print(f"[KVPressAdapter] Cleaned hooks: {removed}")
    return KVPressAdapter(model, press_obj)


def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """Initialize KV Press compression, using HAD simplified version"""
    if not KV_PRESS_AVAILABLE:
        print("[Warning] KV Press not available, skipping compression initialization")
        return None

    print(f"Initializing KV Press: type={press_type}, compression ratio={compression_ratio}")

    try:
        if press_type == 'expected':
            press_obj = ExpectedAttentionPress(compression_ratio=compression_ratio)
        elif press_type == 'observed':
            press_obj = ObservedAttentionPress(compression_ratio=compression_ratio)
        elif press_type == 'random':
            press_obj = RandomPress(compression_ratio=compression_ratio)
            print("[KVPress] Using RandomPress (compatible with all models)")
        elif press_type == 'streaming':
            press_obj = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
        elif press_type == 'tova':
            press_obj = TOVAPress(compression_ratio=compression_ratio)
            print("[KVPress] Using TOVAPress")
        elif press_type == 'snap':
            press_obj = SnapKVPress(compression_ratio=compression_ratio)
            print("[KVPress] Using SnapKVPress")
        else:
            press_obj = KnormPress(compression_ratio=compression_ratio)
            print("[KVPress] Using default KnormPress (compatible with all models)")


        if hasattr(press_obj, 'min_seq_len'):
            try:
                setattr(press_obj, 'min_seq_len', min_seq_len)
            except Exception:
                pass

        print(f"[KVPress] Created {type(press_obj).__name__}")
        return press_obj
    except Exception as e:
        print(f"[KVPress] Failed to create {press_type} object: {e}")
        return None


def load_vox_gender_dataset(root_dir: str, meta_file: str) -> List[Dict[str, Any]]:
    meta_path = meta_file if os.path.isabs(meta_file) else os.path.join(root_dir, meta_file)
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file does not exist: {meta_path}")
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    samples = []
    for i, item in enumerate(metadata):
        rel_path = item.get("path") or ""
        wav_path = os.path.join(root_dir, "wav", rel_path)
        if not os.path.exists(wav_path):
            alt = os.path.join(root_dir, rel_path)
            wav_path = alt if os.path.exists(alt) else wav_path
        if not os.path.exists(wav_path):
            continue
        gender = (item.get("answer_gt", "").strip().lower())
        question = item.get("question", "What is the speaker's gender?")
        choice_a = item.get("choice_a", "male")
        choice_b = item.get("choice_b", "female")
        samples.append({
            "id": item.get("id", f"vox_gender_{i}"),
            "speaker_id": item.get("speaker_id_original", item.get("speaker_id", "unknown")),
            "wav_path": wav_path,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "answer_gt": gender,
        })

    if SAMPLE_LIMIT > 0 and len(samples) > SAMPLE_LIMIT:
        print(f"Sample limit: truncating to first {SAMPLE_LIMIT} samples")
        samples = samples[:SAMPLE_LIMIT]

    male = sum(1 for s in samples if s['answer_gt'] == 'male')
    female = sum(1 for s in samples if s['answer_gt'] == 'female')
    print(f"Loaded Vox gender data: {len(samples)} samples | male: {male}, female: {female}")
    random.shuffle(samples)
    return samples


def extract_gender_answer(text: str, choice_a="male", choice_b="female") -> str:
    if not text:
        return ""
    t = text.strip()
    tl = t.lower()

    if tl == 'a' or tl.startswith('a.') or tl.startswith('a)') or tl.startswith('a:'):
        return choice_a.lower().strip()
    if tl == 'b' or tl.startswith('b.') or tl.startswith('b)') or tl.startswith('b:'):
        return choice_b.lower().strip()
    if 'option a' in tl or 'choice a' in tl or 'a)' in tl:
        return choice_a.lower().strip()
    if 'option b' in tl or 'choice b' in tl or 'b)' in tl:
        return choice_b.lower().strip()

    a_l = choice_a.lower().strip()
    b_l = choice_b.lower().strip()
    if a_l in tl and b_l not in tl:
        return a_l
    if b_l in tl and a_l not in tl:
        return b_l

    male_match = re.search(r"\bmale\b", tl) is not None
    female_match = re.search(r"\bfemale\b", tl) is not None
    if male_match and not female_match:
        return 'male'
    if female_match and not male_match:
        return 'female'
    return ""


def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    valid = [(p, g) for p, g in zip(predictions, ground_truths) if p in ['male','female'] and g in ['male','female']]
    if not valid:
        return {
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "classification_report": "",
            "valid_samples": 0,
            "total_samples": len(predictions)
        }
    vp, vg = zip(*valid)
    acc = accuracy_score(vg, vp)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(vg, vp, average='weighted', zero_division=0)
    f1_macro, _, _, _ = precision_recall_fscore_support(vg, vp, average='macro', zero_division=0)
    cls_report = classification_report(vg, vp, labels=['male','female'], target_names=['male','female'], digits=4, zero_division=0)
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_macro": float(f1_macro),
        "classification_report": cls_report,
        "valid_samples": len(valid),
        "total_samples": len(predictions)
    }


def main():
    args = parse_args()
    result_dir = os.path.abspath(RESULTS_DIR_ENV)
    os.makedirs(result_dir, exist_ok=True)

    method_name = "no_compress" if args.no_compress else ENV_PRESS_TYPE
    ratio_str = f"{ENV_COMPRESSION_RATIO:.3f}"
    output_file = f"{result_dir}/vox_gender_results_kvpress_{method_name}_{ratio_str}.json"
    timing_output_file = f"{result_dir}/vox_gender_timing_kvpress_{method_name}_{ratio_str}.json"
    print(f"Results will be saved to: {output_file}")


    samples = load_vox_gender_dataset(args.data_root, args.meta_file)
    if not samples:
        print("No available samples, exiting.")
        return


    print("Loading Qwen2.5-Omni model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    

    will_use_compression = not args.no_compress and KV_PRESS_AVAILABLE
    

    attention_impl = "eager" if will_use_compression else "flash_attention_2"
    print(f"[Attention] Using {attention_impl} implementation (compression: {'enabled' if will_use_compression else 'disabled'})")
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation=attention_impl, trust_remote_code=True,
    )
    if hasattr(model, 'disable_talker'):
        try:
            model.disable_talker()
        except Exception:
            pass
    model.eval()


    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


    initial_allocated, initial_reserved = get_gpu_memory_usage()
    print(f"After model load GPU memory - Allocated: {initial_allocated:.2f}GB, Reserved: {initial_reserved:.2f}GB")


    if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
        cfg = model.thinker.config
        if not hasattr(cfg, 'audio_token_id'): cfg.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(cfg, 'image_token_id'): cfg.image_token_id = 151655
        if not hasattr(cfg, 'video_token_id'): cfg.video_token_id = 151656
        if not hasattr(cfg, 'audio_bos_token_id'): cfg.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(cfg, 'audio_eos_token_id'): cfg.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
        if not hasattr(cfg, 'image_layer_idx'): cfg.image_layer_idx = False
        if not hasattr(cfg, 'audio_layer_idx'): cfg.audio_layer_idx = None
        if not hasattr(cfg, 'audio_token_num'): cfg.audio_token_num = None
        if not hasattr(cfg, 'audio_token_start'): cfg.audio_token_start = None
        if not hasattr(cfg, 'audio_prune_ratio'): cfg.audio_prune_ratio = 0
        if not hasattr(cfg, 'random'): cfg.random = False
        if not hasattr(cfg, 'frame'): cfg.frame = False
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        cfg2 = model.thinker.model.config
        if not hasattr(cfg2, 'audio_token_id'): cfg2.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(cfg2, 'image_token_id'): cfg2.image_token_id = 151655
        if not hasattr(cfg2, 'video_token_id'): cfg2.video_token_id = 151656
        if not hasattr(cfg2, 'audio_bos_token_id'): cfg2.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(cfg2, 'audio_eos_token_id'): cfg2.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
        if not hasattr(cfg2, 'image_layer_idx'): cfg2.image_layer_idx = False
        if not hasattr(cfg2, 'audio_layer_idx'): cfg2.audio_layer_idx = None
        if not hasattr(cfg2, 'audio_token_num'): cfg2.audio_token_num = None
        if not hasattr(cfg2, 'audio_token_start'): cfg2.audio_token_start = None
        if not hasattr(cfg2, 'audio_prune_ratio'): cfg2.audio_prune_ratio = 0
        if not hasattr(cfg2, 'random'): cfg2.random = False
        if not hasattr(cfg2, 'frame'): cfg2.frame = False


    if args.no_compress:
        press_obj = None
        print("[KVPress] Compression disabled")
    else:
        press_obj = initialize_kv_press(model, ENV_PRESS_TYPE, ENV_COMPRESSION_RATIO, args.min_seq_len)
        if press_obj is None:
            print("[KVPress] Not enabled (creation failed or unavailable)")
        else:
            print(f"[KVPress] Enabled {ENV_PRESS_TYPE}, Compression Ratio: {ENV_COMPRESSION_RATIO}")

    results: List[Dict[str, Any]] = []
    correct = 0
    preds_labels: List[str] = []
    gts_labels: List[str] = []
    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0

    tqdm_kwargs = {'ascii': True, 'dynamic_ncols': True, 'file': sys.stdout}
    with tqdm(total=len(samples), desc="VoxGender Evaluation (Qwen2.5 KVPress)", **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                audio_path = sample['wav_path']
                question = sample['question']
                choice_a = sample['choice_a']; choice_b = sample['choice_b']
                answer_gt = (sample['answer_gt'] or '').lower().strip()

                instruction = (
                    f"{question}\n\n"
                    f"A) {choice_a}\nB) {choice_b}\n\n"
                    f"Please select the correct answer (A or B)."
                )
                sys_prompt = (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. "
                    "You are a helpful assistant that analyzes audio to identify speaker characteristics. Please Listen to this audio and identify the speaker's gender. "
                    "Is this a male or female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'."
                )
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": instruction},
                    ]},
                ]

                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list):
                    text = text[0]
                inputs = processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True, use_audio_in_video=True
                )
                inputs = inputs.to(model.device)
                inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}


                input_tokens = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                use_compression = press_obj is not None and input_tokens >= args.min_seq_len

                # Prefill
                prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if prefill_start: prefill_start.record()
                with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
                    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                if prefill_end: prefill_end.record()

                # Generate
                gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if gen_start: gen_start.record()
                with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
                    out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                if gen_end: gen_end.record()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()


                if torch.cuda.is_available():
                    current_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                    current_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    peak_allocated_gb = max(peak_allocated_gb, current_allocated)
                    peak_reserved_gb = max(peak_reserved_gb, current_reserved)

                prefill_time = prefill_start.elapsed_time(prefill_end)/1000 if prefill_start else 0.0
                total_time = gen_start.elapsed_time(gen_end)/1000 if gen_start else 0.0
                decode_time = max(total_time - prefill_time, 0.0)

                output_text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if "assistant\n" in output_text:
                    output_text = output_text.split("assistant\n")[-1].strip()

                pred_label = extract_gender_answer(output_text, choice_a, choice_b)
                is_correct = (pred_label == answer_gt) if (pred_label and answer_gt) else False

                preds_labels.append(pred_label if pred_label else "")
                gts_labels.append(answer_gt if answer_gt else "")
                correct += int(is_correct)
                output_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0

                results.append({
                    'id': sample['id'],
                    'speaker_id': sample.get('speaker_id', 'unknown'),
                    'question': question,
                    'choices': [choice_a, choice_b],
                    'correct_answer': answer_gt,
                    'predicted_label': pred_label,
                    'is_correct': is_correct,
                    'raw_response': output_text,
                    'timing': {
                        'prefill_time': prefill_time,
                        'decode_time': decode_time,
                        'total_time': prefill_time + decode_time,
                        'output_tokens': output_tokens,
                    },
                    'compression_info': {
                        'enabled': press_obj is not None,
                        'used': use_compression,
                        'press_type': ENV_PRESS_TYPE if press_obj else None,
                        'compression_ratio': ENV_COMPRESSION_RATIO if press_obj else None,
                        'min_seq_len': args.min_seq_len,
                        'input_tokens': input_tokens,
                    },
                })

                curr_acc = correct / (idx + 1)
                pbar.set_postfix({'acc': f'{curr_acc:.3f}'})
                pbar.update(1)
                del inputs, out_ids
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                traceback.print_exc()
                pbar.update(1)
                continue


    metrics = calculate_metrics(preds_labels, gts_labels)

    if results:
        avg_prefill = float(np.mean([r['timing']['prefill_time'] for r in results]))
        avg_decode = float(np.mean([r['timing']['decode_time'] for r in results]))
        avg_total = float(np.mean([r['timing']['total_time'] for r in results]))
    else:
        avg_prefill = avg_decode = avg_total = 0.0

    peak_alloc_gb = peak_reserved_gb = 0.0
    if torch.cuda.is_available():
        try:
            peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        except Exception:
            pass

    final = {
        'summary': {
            'total_samples': len(results),
            'overall_accuracy': metrics.get('accuracy', 0.0),
            'f1_weighted': metrics.get('f1_weighted', 0.0),
            'f1_macro': metrics.get('f1_macro', 0.0),
            'precision_weighted': metrics.get('precision_weighted', 0.0),
            'recall_weighted': metrics.get('recall_weighted', 0.0),
            'classification_report': metrics.get('classification_report', ''),
            'timing_summary': {
                'avg_prefill_time': avg_prefill,
                'avg_decode_time': avg_decode,
                'avg_total_time': avg_total,
            },
            'gpu_memory': {
                'initial_allocated_gb': initial_allocated,
                'initial_reserved_gb': initial_reserved,
                'peak_allocated_gb': peak_alloc_gb,
                'peak_reserved_gb': peak_reserved_gb,
                'device_id': gpu_id,
            },
            'kv_press': {
                'enabled': not args.no_compress and (press_obj is not None),
                'press_type': ENV_PRESS_TYPE if (not args.no_compress and press_obj) else None,
                'compression_ratio': ENV_COMPRESSION_RATIO if (not args.no_compress and press_obj) else None,
                'min_seq_len': args.min_seq_len,
            },
        },
        'detailed_results': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(final), f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_file}")


    try:
        with open(timing_output_file, 'w', encoding='utf-8') as tf:
            json.dump(convert_to_serializable({
                'total_samples': len(results),
                'timing_summary': final['summary']['timing_summary'],
                'gpu_memory': final['summary']['gpu_memory'],
                'kv_press': final['summary']['kv_press'],
            }), tf, ensure_ascii=False, indent=2)
        print(f"Timing statistics saved to: {timing_output_file}")
    except Exception as e:
        print(f"Failed to save timing statistics: {e}")


    print("=== Summary ===")
    print(
        f"Samples: {len(results)} | Acc: {metrics.get('accuracy',0.0):.4f} | "
        f"F1_w: {metrics.get('f1_weighted',0.0):.4f} | F1_macro: {metrics.get('f1_macro',0.0):.4f}"
    )
    if torch.cuda.is_available():
        print(
            f"Peak GPU memory (GB): allocated={peak_alloc_gb:.2f}, reserved={peak_reserved_gb:.2f}"
        )
    print("Detailed classification report has been written to the summary.classification_report field in the result JSON.")


if __name__ == '__main__':
    main()