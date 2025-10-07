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
from typing import Dict, Any, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

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
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "GTZAN_QwenKVPress_Results")
print(f"KV Press config: compression_ratio={ENV_COMPRESSION_RATIO}, compression_type={ENV_PRESS_TYPE}")

sys.path.append("/data/to/your/code/path/")
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

    kvpress_origin_path = "/data/to/your/scripts/path/"
    if kvpress_origin_path not in sys.path:
        sys.path.insert(0, kvpress_origin_path)

    from kvpress import (
        ExpectedAttentionPress, KnormPress, ObservedAttentionPress,
        RandomPress, SnapKVPress, StreamingLLMPress, TOVAPress,
    )
    KV_PRESS_AVAILABLE = True
    print("[Info] KV Press library loaded successfully (using patched version)")
except Exception as e:
    print(f"[Warning] KV Press import failed: {e}")
    KV_PRESS_AVAILABLE = False

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648


def parse_args():
    p = argparse.ArgumentParser(description="GTZAN with Qwen2.5-Omni + KV Press")
    p.add_argument("--model-path", type=str, default="/data/to/your/model/path/")
    p.add_argument("--data-root", type=str, default="/data/to/your/dataset/path/")
    p.add_argument("--meta-file", type=str, default="music_genre_classification_meta.json")
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


def create_kvpress_adapter(model, press_obj):
    """Backward-compatible adapter wrapper for KVPress."""
    if press_obj is None:
        return contextlib.nullcontext()
    

    return press_obj(model)

def verify_multimodal_compatibility(model):
    """Verify model compatibility with multimodal compression methods"""
    try:
        print("[KVPress] Verifying Qwen2.5-Omni compatibility...")
        

        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            print(f"[KVPress] Detected Qwen2.5-Omni thinker.model structure")
            

            if hasattr(base_model, 'layers') and len(base_model.layers) > 0:
                print(f"[KVPress] Model has {len(base_model.layers)} layers")
                

                first_layer = base_model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    print("[KVPress] Multimodal compression method compatibility check passed")
                    return True
                    
        print("[KVPress] Compatibility check failed: model structure does not meet requirements")
        return False
        
    except Exception as e:
        print(f"[KVPress] Compatibility check error: {e}")
        return False

def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """Initialize KV Press compression, using HAD simplified version"""
    if not KV_PRESS_AVAILABLE:
        print("[Warning] KV Press not available, skip compression initialization")
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
            if verify_multimodal_compatibility(model):
                press_obj = TOVAPress(compression_ratio=compression_ratio)
                print("[KVPress] Using patched TOVAPress (supports Qwen2.5-Omni)")
            else:
                print("[KVPress] TOVA compatibility check failed, fallback to KnormPress")
                press_obj = KnormPress(compression_ratio=compression_ratio)
        elif press_type == 'snap':
            if verify_multimodal_compatibility(model):
                press_obj = SnapKVPress(compression_ratio=compression_ratio)
                print("[KVPress] Using patched SnapKVPress (supports Qwen2.5-Omni)")
            else:
                print("[KVPress] SnapKV compatibility check failed, fallback to KnormPress")
                press_obj = KnormPress(compression_ratio=compression_ratio)
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
        traceback.print_exc()
        return None
            
        def __enter__(self):
            """Register forward hooks to all self_attn layers"""
            try:

                if hasattr(self.original_model, 'thinker') and hasattr(self.original_model.thinker, 'model'):
                    self.base_model = self.original_model.thinker.model
                elif hasattr(self.original_model, 'model'):
                    self.base_model = self.original_model.model
                else:
                    print("[KVPressAdapter] Warning: Unable to locate base model")
                    self.hooks = []
                    return self
                
                if not hasattr(self.base_model, 'layers'):
                    print("[KVPressAdapter] Warning: Base model does not have 'layers' attribute")
                    self.hooks = []
                    return self
                
                layers = self.base_model.layers
                print(f"[KVPressAdapter] Registering hooks for {len(layers)} layers (method: {type(self.press_object).__name__})")
                

                if self.press_method in ['tovapress', 'snapkvpress']:
                    if not self._validate_multimodal_compatibility():
                        print(f"[KVPressAdapter] {self.press_method} multimodal compatibility check failed")
                        self.hooks = []
                        return self
                
                hooks = []
                successful_hooks = 0
                

                base_rotary_emb = getattr(self.base_model, 'rotary_emb', None)
                
                for layer_idx, layer in enumerate(layers):
                    try:
                        if not hasattr(layer, 'self_attn'):
                            continue
                            
                        attn_module = layer.self_attn
                        

                        if base_rotary_emb and (not hasattr(attn_module, 'rotary_emb') or attn_module.rotary_emb is None):
                            attn_module.rotary_emb = base_rotary_emb
                        

                        if self.press_method in ['tovapress', 'snapkvpress']:
                            hook_func = self.press_object.forward_hook
                        else:
                            hook_func = self.press_object.forward_hook
                        

                        if hasattr(self.press_object, 'forward_hook'):
                            hook = attn_module.register_forward_hook(
                                hook_func,
                                with_kwargs=True
                            )
                            hooks.append(hook)
                            successful_hooks += 1
                            
                    except Exception as e:
                        print(f"[KVPressAdapter] Layer {layer_idx} hook registration failed: {e}")
                        continue
                
                self.hooks = hooks
                print(f"[KVPressAdapter] Successfully registered {successful_hooks}/{len(layers)} hooks")
                

                if type(self.press_object).__name__ in ['SnapKVPress', 'TOVAPress']:
                    if successful_hooks == 0:
                        print(f"[KVPressAdapter] Warning: {type(self.press_object).__name__} method did not register any hooks")
                    else:
                        print(f"[KVPressAdapter] {type(self.press_object).__name__} method is ready")
                
                return self
                
            except Exception as e:
                print(f"[KVPressAdapter] Registration error: {e}")
                import traceback
                traceback.print_exc()
                

                for h in getattr(self, 'hooks', []):
                    try:
                        h.remove()
                    except:
                        pass
                        
                self.hooks = []
                return self
        
        def _validate_multimodal_compatibility(self):
            """Validate multimodal compatibility, especially for TOVA/SnapKV"""
            try:

                if hasattr(self.base_model, 'config'):
                    config = self.base_model.config

                    multimodal_attrs = ['audio_token_id', 'image_token_id', 'video_token_id']
                    has_multimodal = any(hasattr(config, attr) for attr in multimodal_attrs)
                    
                    if has_multimodal:
                        print(f"[KVPressAdapter] Detected multimodal config, enabling compatibility mode for {self.press_method}")
                        return True
                        
                return True
            except Exception as e:
                print(f"[KVPressAdapter] Multimodal compatibility check failed: {e}")
                return False
        
        def _create_safe_hook_wrapper(self, original_hook):
            """Create direct hook wrapper for TOVA/SnapKV (no error handling)"""
            def direct_hook_wrapper(module, args, kwargs, output):

                return original_hook(module, args, kwargs, output)
                    
            return direct_hook_wrapper
                
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Clean up all registered hooks"""
            removed_count = 0
            for hook in self.hooks:
                try:
                    hook.remove()
                    removed_count += 1
                except Exception as e:
                    pass
                    
            if removed_count > 0:
                print(f"[KVPressAdapter] Cleaned {removed_count} hooks")
                
            self.hooks = []
            

            if exc_type is not None and self.press_method not in ['tovapress', 'snapkvpress']:
                print(f"[KVPressAdapter] Exception detected on exit: {exc_type.__name__}: {exc_val}")
    
    return Qwen2_5OmniKVPressAdapter(model, press_obj)


def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """Initialize KV Press object, optimized for Qwen2.5-Omni structure"""
    if not KV_PRESS_AVAILABLE:
        print("[KVPress] Library not available")
        return None
        
    print(f"[KVPress] Initializing {press_type} method (compression ratio: {compression_ratio})")
    
    try:
        if press_type == 'expected':
            press_obj = ExpectedAttentionPress(compression_ratio=compression_ratio)
            print("[KVPress] ExpectedAttentionPress initialized")
            
        elif press_type == 'observed':
            press_obj = ObservedAttentionPress(compression_ratio=compression_ratio)
            print("[KVPress] ObservedAttentionPress initialized")
            
        elif press_type == 'random':
            press_obj = RandomPress(compression_ratio=compression_ratio)
            print("[KVPress] RandomPress initialized")
            
        elif press_type == 'tova':
            if verify_multimodal_compatibility(model):
                press_obj = TOVAPress(compression_ratio=compression_ratio)
                print("[KVPress] TOVAPress initialized (supports Qwen2.5-Omni)")
            else:
                print("[KVPress] TOVA compatibility check failed, fallback to KnormPress")
                press_obj = KnormPress(compression_ratio=compression_ratio)
                
        elif press_type == 'snap':
            if verify_multimodal_compatibility(model):
                press_obj = SnapKVPress(compression_ratio=compression_ratio)
                print("[KVPress] SnapKVPress initialized (supports Qwen2.5-Omni)")
            else:
                print("[KVPress] SnapKV compatibility check failed, fallback to KnormPress")
                press_obj = KnormPress(compression_ratio=compression_ratio)
            
        elif press_type == 'streaming':
            press_obj = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
            print("[KVPress] StreamingLLMPress initialized")
            
        else:

            press_obj = KnormPress(compression_ratio=compression_ratio)
            print("[KVPress] KnormPress (default) initialized")
            
        if press_obj is not None:
            print(f"[KVPress] Successfully created {type(press_obj).__name__} object")
            print(f"[KVPress] Compression ratio: {compression_ratio}")
            print(f"[KVPress] Min sequence length: {min_seq_len}")
            
        return press_obj
        
    except Exception as e:
        print(f"[KVPress] Failed to create {press_type} object: {e}")
        traceback.print_exc()
        return None


def load_gtzan_metadata(metadata_path: str, data_root: str) -> List[Dict[str, Any]]:
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file does not exist: {metadata_path}")
        return []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = []
    for i, item in enumerate(data):
        if not all(k in item for k in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            continue
        path_rel = item.get('path')
        full = path_rel if os.path.isabs(path_rel) else os.path.join(data_root, path_rel)
        if not os.path.exists(full):
            alt = os.path.join(data_root, 'wav', path_rel)
            full = alt if os.path.exists(alt) else full
        if not os.path.exists(full):
            continue
        dataset.append({
            'path': full,
            'question': item.get('question'),
            'choice_a': item.get('choice_a'),
            'choice_b': item.get('choice_b'),
            'choice_c': item.get('choice_c'),
            'choice_d': item.get('choice_d'),
            'answer_gt': str(item.get('answer_gt', '')).upper(),
            'id': item.get('id', f'gtzan_{i}'),
            'genre_label': item.get('genre', None),
        })
    print(f"Loaded {len(dataset)} valid GTZAN samples")
    return dataset


def build_gtzan_prompt(question: str, a: str, b: str, c: str, d: str) -> str:
    instruction = "Listen to this audio segment and identify the music genre based on what you hear."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    formatted = f"A) {a}\nB) {b}\nC) {c}\nD) {d}"
    prompt = f"{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted}\n\n{format_text}"
    return prompt


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
    output_file = f"{result_dir}/gtzan_results_kvpress_{method_name}_{ratio_str}.json"
    timing_output_file = f"{result_dir}/gtzan_timing_kvpress_{method_name}_{ratio_str}.json"
    print(f"Results will be saved to: {output_file}")

    meta_file = args.meta_file if os.path.isabs(args.meta_file) else os.path.join(args.data_root, args.meta_file)
    samples = load_gtzan_metadata(meta_file, args.data_root)
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
    print(f"After model load GPU memory - allocated: {initial_allocated:.2f}GB, reserved: {initial_reserved:.2f}GB")


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
            if ENV_PRESS_TYPE in ['snap', 'tova']:
                print(f"[KVPress] {ENV_PRESS_TYPE} method may require specific model config or version")
        else:
            print(f"[KVPress] Enabled {ENV_PRESS_TYPE}, compression ratio: {ENV_COMPRESSION_RATIO}")
            print(f"[KVPress] Object type: {type(press_obj).__name__}")

    results = []
    correct = 0
    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0
    
    tqdm_kwargs = {'ascii': True, 'dynamic_ncols': True, 'file': sys.stdout}
    with tqdm(total=len(samples), desc="GTZAN Evaluation (Qwen2.5 KVPress)", **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                audio_path = sample['path']
                q = sample['question']
                a = sample['choice_a']; b = sample['choice_b']
                c = sample['choice_c']; d = sample['choice_d']
                gt = sample['answer_gt']
                prompt = build_gtzan_prompt(q, a, b, c, d)
                sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are a helpful assistant that analyzes audio and classifies music genres."
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": prompt},
                    ]},
                ]

                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): text = text[0]
                inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                inputs = inputs.to(model.device)
                inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}


                prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if prefill_start: prefill_start.record()
                

                use_compression = press_obj is not None
                
                with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
                    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                        
                if prefill_end: prefill_end.record()


                gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if gen_start: gen_start.record()
                

                with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
                    out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                        
                if gen_end: gen_end.record()
                if torch.cuda.is_available(): torch.cuda.synchronize()
                

                if torch.cuda.is_available():
                    current_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                    current_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    peak_allocated_gb = max(peak_allocated_gb, current_allocated)
                    peak_reserved_gb = max(peak_reserved_gb, current_reserved)
                

                compression_used = use_compression
                

                prefill_time = prefill_start.elapsed_time(prefill_end)/1000 if prefill_start else 0.0
                total_time = gen_start.elapsed_time(gen_end)/1000 if gen_start else 0.0
                decode_time = max(total_time - prefill_time, 0.0)


                output_text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if "assistant\n" in output_text:
                    output_text = output_text.split("assistant\n")[-1].strip()
                    
                pred_choice = extract_choice_from_text(output_text)
                is_correct = pred_choice == gt
                correct += int(is_correct)
                output_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0


                results.append({
                    'id': sample['id'],
                    'question': q,
                    'choices': [a, b, c, d],
                    'correct_answer': gt,
                    'predicted_choice': pred_choice,
                    'is_correct': is_correct,
                    'raw_response': output_text,
                    'genre_label': sample.get('genre_label', None),
                    'timing': {
                        'prefill_time': prefill_time,
                        'decode_time': decode_time,
                        'total_time': prefill_time + decode_time,
                        'output_tokens': output_tokens,
                    },
                    'compression_info': {
                        'method': ENV_PRESS_TYPE if press_obj else 'no_compress',
                        'compression_used': compression_used,
                        'fallback_occurred': False,
                    },
                })


                curr_acc = correct / (idx + 1)
                compression_status = "✓" if compression_used else ("×" if press_obj else "-")
                pbar.set_postfix({
                    'acc': f'{curr_acc:.3f}',
                    'compress': compression_status
                })
                pbar.update(1)
                

                del inputs, out_ids
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                traceback.print_exc()
                pbar.update(1)
                continue


    total_samples = len(results)
    compression_stats = {
        'total_samples': total_samples,
        'compression_successful': 0,
        'no_compression': 0,
    }
    
    for result in results:
        comp_info = result.get('compression_info', {})
        if comp_info.get('compression_used', False):
            compression_stats['compression_successful'] += 1
        else:
            compression_stats['no_compression'] += 1
    

    preds = [r['predicted_choice'] for r in results]
    gts = [r['correct_answer'] for r in results]
    metrics = calculate_metrics(preds, gts)
    
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
            'total_samples': total_samples,
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
                'compression_stats': compression_stats,
                'success_rate': compression_stats['compression_successful'] / max(total_samples, 1),
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


    print("=" * 60)
    print("=== GTZAN Music Genre Classification Result Summary ===")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Accuracy: {metrics.get('accuracy',0.0):.4f}")
    print(f"F1 weighted: {metrics.get('f1_weighted',0.0):.4f}")
    print(f"F1 macro: {metrics.get('f1_macro',0.0):.4f}")
    
    if press_obj is not None:
        print(f"KV compression method: {ENV_PRESS_TYPE}")
        print(f"Compression ratio: {ENV_COMPRESSION_RATIO}")
        success_rate = compression_stats['compression_successful'] / max(len(results), 1)
        print(f"Compression success rate: {success_rate:.1%} ({compression_stats['compression_successful']}/{len(results)})")
        print(f"Compression status: direct execution (no error handling)")
    else:
        print("KV compression: not enabled")
        
    print(f"Average inference time: {avg_total:.3f}s (prefill: {avg_prefill:.3f}s, decode: {avg_decode:.3f}s)")
    
    if torch.cuda.is_available():
        print(f"Peak memory: allocated={peak_alloc_gb:.2f}GB, reserved={peak_reserved_gb:.2f}GB")
    print("=" * 60)


if __name__ == '__main__':
    main()