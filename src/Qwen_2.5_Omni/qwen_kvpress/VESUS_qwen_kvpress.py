import os
import sys
import json
import argparse
import warnings
import random
import traceback
import contextlib
import numpy as np
import torch
import transformers
import re
from collections import defaultdict
from typing import Dict, Any, List
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
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "VESUS_QwenKVPress_Results")
print(f"KV Press config: compression ratio={ENV_COMPRESSION_RATIO}, type={ENV_PRESS_TYPE}")
SKIP_PERSONS_ENV = os.environ.get("SKIP_PERSONS", "").strip()
SKIP_EMOTION_ENV = os.environ.get("SKIP_EMOTION", "").strip()

sys.path.append("/data/to/your/qwen/code/path")
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
    print(f"[Warning] KV Press library error: {e}")
    KV_PRESS_AVAILABLE = False

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648


def parse_args():
    p = argparse.ArgumentParser(description="VESUS with Qwen2.5-Omni + KV Press")
    p.add_argument("--model-path", type=str, default="/data/to/your/model/path")
    p.add_argument("--data-root", type=str, default="/data/to/your/dataset/path")
    p.add_argument("--meta-file", type=str, default="audio_emotion_dataset.json")
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
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0.0, 0.0


def patch_qwen_for_kvpress(model):
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
                        for layer in base_model.layers:
                            if hasattr(layer, 'self_attn'):
                                if not hasattr(layer.self_attn, 'rotary_emb') or layer.self_attn.rotary_emb is None:
                                    if hasattr(base_model, 'rotary_emb'):
                                        layer.self_attn.rotary_emb = base_model.rotary_emb
                                hook = layer.self_attn.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
                                hooks.append(hook)
                        self.hooks = hooks
                        return self
                if hasattr(self.original_model, 'model') and hasattr(self.original_model.model, 'layers'):
                    base_model = self.original_model.model
                    for layer in base_model.layers:
                        if hasattr(layer, 'self_attn'):
                            if hasattr(base_model, 'rotary_emb'):
                                layer.self_attn.rotary_emb = base_model.rotary_emb
                            hook = layer.self_attn.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
                            hooks.append(hook)
                    self.hooks = hooks
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
            for h in getattr(self, 'hooks', []):
                try: h.remove()
                except: pass
    return KVPressAdapter(model, press_obj)


def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """Initialize KV Press compression, using HAD simplified version"""
    if not KV_PRESS_AVAILABLE:
        print("[Warning] KV Press not available, skipping compression initialization")
        return None

    print(f"Initializing KV Press: type={press_type}, compression_ratio={compression_ratio}")

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


def load_vesus_dataset(json_file_path: str, data_root: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_file_path):
        print(f"Error: Dataset file does not exist: {json_file_path}")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    def resolve_audio_path(base_root: str, raw_path: str) -> str:
        raw_path = (raw_path or "").replace("\\", "/").lstrip("/")
        candidates = []
        # 1) base_root/path
        if raw_path:
            candidates.append(os.path.join(base_root, raw_path))
            # 2) base_root/wav/path
            candidates.append(os.path.join(base_root, 'wav', raw_path))

        lowered = raw_path.lower()
        if "vesus/" in lowered:
            suffix = raw_path[lowered.index("vesus/") + len("vesus/"):]
            candidates.append(os.path.join(base_root, suffix))
            candidates.append(os.path.join(base_root, 'wav', suffix))

        if os.path.isabs(item.get('path', '')):
            candidates.insert(0, item.get('path'))
        for p in candidates:
            if p and os.path.exists(p):
                return os.path.normpath(p)
        return ""

    skip_persons = set()
    if SKIP_PERSONS_ENV:
        for s in SKIP_PERSONS_ENV.split(','):
            s = s.strip()
            if s:
                skip_persons.add(s)
    skip_emotion = SKIP_EMOTION_ENV.lower() if SKIP_EMOTION_ENV else ""

    dataset = []
    filtered_missing = 0
    filtered_skipped = 0
    for i, item in enumerate(data):
        path_rel = item.get('path') or item.get('audio_path') or ''
        full = resolve_audio_path(data_root, path_rel)
        if not full:
            raw = item.get('path') or item.get('audio_path') or ''
            if raw and os.path.exists(raw):
                full = raw
        if not full:
            filtered_missing += 1
            continue

        emotion = (item.get('emotion_label') or item.get('emotion') or 'unknown').strip()
        person = item.get('person_id', item.get('speaker_id', 'unknown'))

        person_str = str(person)

        if skip_persons and person_str in skip_persons and (not skip_emotion or emotion.lower() == skip_emotion):
            filtered_skipped += 1
            continue

        item_out = {
            'path': full,
            'question': item.get('question', 'What emotion is expressed in this audio segment?'),
            'choice_a': item.get('choice_a', 'angry'),
            'choice_b': item.get('choice_b', 'happy'),
            'choice_c': item.get('choice_c', 'sad'),
            'choice_d': item.get('choice_d', 'fearful'),
            'answer_gt': (item.get('answer_gt', '') or '').upper(),
            'emotion_label': emotion,
            'person_id': person_str,
            'id': item.get('id', f'vesus_{i}')
        }
        dataset.append(item_out)

    print(f"Loaded {len(dataset)} valid VESUS samples (filtered missing: {filtered_missing}, skipped speakers: {filtered_skipped})")
    return dataset


def extract_choice_from_text(text: str) -> str:
    if not text:
        return ""
    s = text.strip().upper()
    for ch in s:
        if ch in ["A", "B", "C", "D"]:
            return ch
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    return ""


def calculate_metrics(predictions, ground_truths):
    valid = [(p, g) for p, g in zip(predictions, ground_truths) if p and g]
    if not valid:
        return {
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "classification_report": ""
        }
    vp, vg = zip(*valid)
    acc = accuracy_score(vg, vp)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(vg, vp, average='weighted', zero_division=0)
    f1_macro, _, _, _ = precision_recall_fscore_support(vg, vp, average='macro', zero_division=0)
    cls_report = classification_report(vg, vp, digits=4, zero_division=0)
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_macro": float(f1_macro),
        "classification_report": cls_report,
    }


def main():
    args = parse_args()
    result_dir = os.path.abspath(RESULTS_DIR_ENV)
    os.makedirs(result_dir, exist_ok=True)

    method_name = "no_compress" if args.no_compress else ENV_PRESS_TYPE
    ratio_str = f"{ENV_COMPRESSION_RATIO:.3f}"
    output_file = f"{result_dir}/vesus_results_kvpress_{method_name}_{ratio_str}.json"
    timing_output_file = f"{result_dir}/vesus_timing_kvpress_{method_name}_{ratio_str}.json"
    print(f"Results will be saved to: {output_file}")

    samples = load_vesus_dataset(os.path.join(args.data_root, args.meta_file) if not os.path.isabs(args.meta_file) else args.meta_file, args.data_root)
    if SAMPLE_LIMIT > 0 and len(samples) > SAMPLE_LIMIT:
        samples = samples[:SAMPLE_LIMIT]
        print(f"Sample limit: {len(samples)}")

    print("Loading Qwen2.5-Omni model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    model.disable_talker()

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    initial_allocated, initial_reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - allocated: {initial_allocated:.2f}GB, reserved: {initial_reserved:.2f}GB")

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
            print("[KVPress] Not enabled (creation failed or not available)")
        else:
            print(f"[KVPress] Enabled {ENV_PRESS_TYPE}, compression ratio: {ENV_COMPRESSION_RATIO}")

    results = []
    correct = 0
    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0
    tqdm_kwargs = {'ascii': True, 'dynamic_ncols': True, 'file': sys.stdout}
    with tqdm(total=len(samples), desc="VESUS Evaluation (Qwen2.5 KVPress)", **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                audio_path = sample['path']
                question = sample['question']
                choice_a = sample['choice_a']; choice_b = sample['choice_b']
                choice_c = sample['choice_c']; choice_d = sample['choice_d']
                answer_gt = sample['answer_gt']

                instruction = f"{question}\n\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}\n\nPlease select the correct answer (A, B, C, or D)."
                sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are a helpful assistant that analyzes audio to answer questions about emotions. Please listen to the audio carefully and select the correct answer."
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": instruction},
                    ]},
                ]

                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): text = text[0]
                inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                inputs = inputs.to(model.device)
                inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}

                # Prefill
                prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if prefill_start: prefill_start.record()
                with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if press_obj else contextlib.nullcontext()):
                    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                if prefill_end: prefill_end.record()

                # Generate
                gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if gen_start: gen_start.record()
                with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if press_obj else contextlib.nullcontext()):
                    out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                if gen_end: gen_end.record()
                if torch.cuda.is_available(): torch.cuda.synchronize()

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
                pred_choice = extract_choice_from_text(output_text)
                is_correct = pred_choice == answer_gt
                correct += int(is_correct)
                output_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0

                results.append({
                    'id': sample['id'],
                    'question': question,
                    'choices': [choice_a, choice_b, choice_c, choice_d],
                    'correct_answer': answer_gt,
                    'predicted_choice': pred_choice,
                    'is_correct': is_correct,
                    'raw_response': output_text,
                    'emotion_label': sample.get('emotion_label', 'unknown'),
                    'person_id': sample.get('person_id', 'unknown'),
                    'timing': {
                        'prefill_time': prefill_time,
                        'decode_time': decode_time,
                        'total_time': prefill_time + decode_time,
                        'output_tokens': output_tokens,
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

    preds = [r['predicted_choice'] for r in results]
    gts = [r['correct_answer'] for r in results]
    metrics = calculate_metrics(preds, gts)

    if results:
        avg_prefill = float(np.mean([r['timing']['prefill_time'] for r in results]))
        avg_decode = float(np.mean([r['timing']['decode_time'] for r in results]))
        avg_total = float(np.mean([r['timing']['total_time'] for r in results]))
    else:
        avg_prefill = avg_decode = avg_total = 0.0

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
                'peak_allocated_gb': peak_allocated_gb,
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

    timing_summary = final['summary']['timing_summary']
    try:
        with open(timing_output_file, 'w', encoding='utf-8') as tf:
            json.dump(convert_to_serializable({
                'total_samples': len(results),
                'timing_summary': timing_summary,
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
            f"Peak GPU memory (GB): allocated={peak_allocated_gb:.2f}, reserved={peak_reserved_gb:.2f}"
        )
    print("Detailed classification report written to summary.classification_report in the result JSON.")


if __name__ == '__main__':
    main()