import os
import sys
import json
import time
import gc
import argparse
import warnings
import random
import traceback
import contextlib
import re
import numpy as np
import torch
import librosa
import soundfile as sf
import transformers
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score


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
if SAMPLE_LIMIT > 0:
    print(f"Sample limit set to: {SAMPLE_LIMIT}")

print(f"KV Press config: compression_ratio={ENV_COMPRESSION_RATIO}, press_type={ENV_PRESS_TYPE}")


sys.path.append("/data/to/your/code/path/Qwen_2.5")
try:
    from modeling_qwen2_5_omni_origin import Qwen2_5OmniForConditionalGeneration
    from processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    QWEN_AVAILABLE = True
    print("[INFO] Qwen2.5-Omni module loaded successfully")
except ImportError as e:
    print(f"[WARNING] Qwen2.5-Omni module import failed: {e}")
    QWEN_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] Qwen2.5-Omni module loading error: {e}")
    QWEN_AVAILABLE = False


KV_PRESS_AVAILABLE = False
try:
    import transformers
    transformers_version = transformers.__version__
    print(f"[INFO] Transformers version: {transformers_version}")
    

    from kvpress import (
        ExpectedAttentionPress,
        KnormPress,
        ObservedAttentionPress,
        RandomPress,
        SnapKVPress,
        StreamingLLMPress,
        TOVAPress,
    )
    KV_PRESS_AVAILABLE = True
    print("[INFO] KV Press library loaded successfully")
    
except ImportError as e:
    print(f"[WARNING] KV Press library import failed: {e}")
    KV_PRESS_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] KV Press library loading error: {e}")
    KV_PRESS_AVAILABLE = False

if not KV_PRESS_AVAILABLE:
    print("[ERROR] KV Press library not available")
    print("Please check if kvpress library is installed: pip install kvpress")


_AUDIO_TOKEN_ID = 151646          # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'


RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "DESED_QwenKVPress_Results")


KV_PRESS_CONFIG = {
    "compression_ratio": ENV_COMPRESSION_RATIO,
    "head_dims": None,
    "num_attention_heads": None,
    "press_type": ENV_PRESS_TYPE,
    "return_indices": True,
    "min_seq_len": 128,
    "model_kwargs": {
        "attn_implementation": "flash_attention_2",
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="DESED with Qwen2.5-Omni + KV Press")
    parser.add_argument("--model-path", type=str, default="/data/to/your/model/path/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-file", type=str, default="/data/to/your/dataset/path/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json")
    parser.add_argument("--audio-base-dir", type=str, default="/data/to/your/dataset/path/DESED_dataset/concatenated_audio")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum generated tokens")
    parser.add_argument("--min-seq-len", type=int, default=128, help="Compression threshold")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    return parser.parse_args()

def convert_to_serializable(obj):
    """Recursively convert objects to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return str(obj) if obj is not None else None

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files as required by Qwen2.5-Omni - Refer to SLUE's processing"""

    try:

        if not os.path.exists(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None
            

        try:
            audio_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Loaded successfully with librosa: shape={audio_np.shape}, sample rate={sr}Hz")
        except Exception as e:
            print(f"librosa loading failed: {e}")
            

            try:
                audio_np, sample_rate = sf.read(audio_path)
                

                if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
                    audio_np = np.mean(audio_np, axis=1)
                

                if sample_rate != target_sr:
                    from scipy import signal
                    audio_np = signal.resample(audio_np, int(len(audio_np) * target_sr / sample_rate))
                    
                audio_np = audio_np.astype(np.float32)
                sr = target_sr
                print(f"soundfile processed successfully: shape={audio_np.shape}, sample rate={sr}Hz")
                
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                return None
        

        if audio_np.size == 0:
            print(f"Audio is empty: {audio_path}")
            return None
            

        if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
            print(f"Audio contains invalid values: {audio_path}")
            return None
            

        return audio_path
        
    except Exception as e:
        print(f"Audio processing failed: {audio_path}, error: {e}")
        return None

def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    Load data from DESED task JSON file
    
    Args:
        json_file: DESED task JSON file path
        audio_base_dir: Base directory for audio files
    
    Returns:
        dataset: List containing task data
    """
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"JSON file does not exist: {json_file}")
        return []
    
    print(f"Loading DESED task JSON: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    

    if isinstance(data, list):

        tasks = data
    elif isinstance(data, dict) and 'tasks' in data:

        tasks = data['tasks']
    else:
        print("JSON format error: not a list and no 'tasks' field")
        return []
    
    print(f"Loaded {len(tasks)} tasks from JSON")
    

    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        try:

            required_fields = ["path", "question", "choices", "answer_gt"]
            missing_fields = [field for field in required_fields if field not in task]
            
            if missing_fields:
                print(f"Task {i} missing fields: {missing_fields}")
                continue
                

            choices = task.get("choices", {})
            if not isinstance(choices, dict) or not all(key in choices for key in ["A", "B", "C", "D"]):
                print(f"Task {i} choices format error, must contain options A, B, C, D")
                continue
            

            audio_path = task["path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_base_dir, audio_path)
            

            if not os.path.exists(audio_path):
                missing_files += 1
                if missing_files <= 5:
                    print(f"Audio file does not exist: {audio_path}")
                continue
            

            task_type = task.get("task_type", "unknown")
            task_type_stats[task_type] += 1
            

            dataset.append({
                "audio_path": audio_path,
                "question": task["question"],
                "choice_a": choices["A"],
                "choice_b": choices["B"], 
                "choice_c": choices["C"],
                "choice_d": choices["D"],
                "correct_answer": task["answer_gt"],
                "task_type": task.get("task_type", "unknown"),
                "task_id": task.get("uniq_id", f"task_{i}")
            })
            
        except Exception as e:
            print(f"Error processing task {i}: {e}")
            continue
    
    if missing_files > 5:
        print(f"Total missing audio files: {missing_files}")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    return dataset

def extract_answer_choice(response):
    """Extract answer choice (A, B, C, D) from model response"""
    if not response:
        return "unknown"
    


    if '<|im_start|>assistant' in response:
        response = response.split('<|im_start|>assistant')[-1]
    elif 'assistant' in response.lower() and ':' in response:
        response = response.split(':', 1)[1]
    

    response = re.sub(r'<\|.*?\|>', '', response)
    response = re.sub(r'<.*?>', '', response)
    

    response = response.strip().upper()
    

    if len(response) > 50:
        response = response[:50]
    

    for choice in ['A', 'B', 'C', 'D']:

        if f' {choice} ' in f' {response} ' or response.startswith(f'{choice} ') or response.endswith(f' {choice}'):
            return choice

        if f'({choice})' in response or f'{choice})' in response:
            return choice

        if f'OPTION {choice}' in response or f'CHOICE {choice}' in response:
            return choice
    

    if response in ['A', 'B', 'C', 'D']:
        return response
    

    response_words = response.split()
    for word in response_words:
        if word in ['A', 'B', 'C', 'D']:
            return word
    
    return "unknown"

def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate QA accuracy"""
    return predicted_choice.upper() == ground_truth_choice.upper()

@dataclass
class DESEDTimingStats:
    """Timing statistics for DESED sound event detection tasks"""
    samples: int = 0
    total_prefill_time: float = 0.0
    total_decode_time: float = 0.0
    total_output_tokens: int = 0
    total_input_tokens: int = 0
    peak_gpu_memory: float = 0.0
    initial_gpu_memory: float = 0.0
    correct_predictions: int = 0
    detailed_records: List[Dict] = None
    
    def __post_init__(self):
        if self.detailed_records is None:
            self.detailed_records = []
    
    def record_initial_memory(self):
        """Record initial GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated, _ = get_gpu_memory_usage()
            self.initial_gpu_memory = allocated
            print(f"Initial GPU memory: {self.initial_gpu_memory:.2f} GB")
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens=0, is_correct=False, peak_memory_gb=None, task_type=None):
        """Add single inference record"""
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_output_tokens += output_tokens
        self.total_input_tokens += input_tokens
        
        if is_correct:
            self.correct_predictions += 1
        
        if peak_memory_gb is not None:
            self.peak_gpu_memory = max(self.peak_gpu_memory, peak_memory_gb)
        

        self.detailed_records.append({
            "sample_id": self.samples,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "is_correct": is_correct,
            "task_type": task_type,
            "peak_memory_gb": peak_memory_gb
        })
    
    def get_summary(self):
        """Get summary statistics"""
        if self.samples == 0:
            return {"samples": 0, "message": "No data"}
        
        avg_prefill = self.total_prefill_time / self.samples
        avg_decode = self.total_decode_time / self.samples
        avg_total = avg_prefill + avg_decode
        avg_output_tokens = self.total_output_tokens / self.samples
        avg_input_tokens = self.total_input_tokens / self.samples
        accuracy = self.correct_predictions / self.samples
        
        return {
            "samples": self.samples,
            "accuracy": accuracy,
            "correct_predictions": self.correct_predictions,
            "avg_prefill_time": avg_prefill,
            "avg_decode_time": avg_decode,
            "avg_total_time": avg_total,
            "avg_output_tokens": avg_output_tokens,
            "avg_input_tokens": avg_input_tokens,
            "peak_gpu_memory": self.peak_gpu_memory,
            "initial_gpu_memory": self.initial_gpu_memory,
            "memory_increase": self.peak_gpu_memory - self.initial_gpu_memory
        }
    
    def export_to_json(self, output_file):
        """Export to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.detailed_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(result), f, indent=2, ensure_ascii=False)
        
        return output_file

def patch_qwen_for_kvpress(model):
    """Add rotary_emb compatibility for Qwen2.5-Omni, optimized based on actual model structure"""
    try:

        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            print("[KVPress] Found thinker.model as base model")
        elif hasattr(model, 'model'):
            base_model = model.model
            print("[KVPress] Found model as base model")
        else:
            print("[KVPress] Warning: Expected model structure not found")
            return False
        
        if not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
            print("[KVPress] Warning: Base model has no layers")
            return False
            
        print(f"[KVPress] Found {len(base_model.layers)} layers")
        

        existing_rotary_emb = getattr(base_model, 'rotary_emb', None)
        if existing_rotary_emb is not None:
            print("[KVPress] Found existing rotary_emb")

            for i, layer in enumerate(base_model.layers):
                if hasattr(layer, 'self_attn'):
                    if not hasattr(layer.self_attn, 'rotary_emb'):
                        layer.self_attn.rotary_emb = existing_rotary_emb
                        print(f"[KVPress] Added rotary_emb reference to layer {i}")
            return True
            

        first_layer = base_model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
            source_rotary = first_layer.self_attn.rotary_emb
            if source_rotary is not None:

                base_model.rotary_emb = source_rotary
                print("[KVPress] Extracted and set global rotary_emb from first layer")
                

                for i, layer in enumerate(base_model.layers):
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn.rotary_emb = source_rotary
                return True
        

        print("[KVPress] No existing rotary_emb found, attempting to create")
        

        config = None
        for candidate in [model, getattr(model, 'thinker', None), base_model]:
            if candidate and hasattr(candidate, 'config'):
                config = candidate.config
                break
                
        if config is None:
            print("[KVPress] Warning: Unable to get model config")
            return False
            

        class KVPressCompatibleRotaryEmbedding:
            def __init__(self, config):
                self.config = config

                self.max_position_embeddings = getattr(config, 'max_position_embeddings', 32768)
                self.rope_theta = getattr(config, 'rope_theta', 10000.0)
                

                self.hidden_size = getattr(config, 'hidden_size', 3584)
                self.num_attention_heads = getattr(config, 'num_attention_heads', 28)
                self.head_dim = self.hidden_size // self.num_attention_heads
                

                inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
                self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
                self.register_buffer("inv_freq", inv_freq)
                
                print(f"[KVPress] Created compatible rotary_emb: head_dim={self.head_dim}, theta={self.rope_theta}")
                
            def forward(self, x, position_ids=None):
                """Simplified forward pass, mainly for KVPress"""
                seq_len = x.shape[-2] if position_ids is None else position_ids.shape[-1]
                

                device = x.device
                dtype = x.dtype
                
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
                

                freqs = torch.outer(position_ids.float(), self.inv_freq.to(device))
                emb = torch.cat((freqs, freqs), dim=-1)
                
                cos = emb.cos().to(dtype=dtype)
                sin = emb.sin().to(dtype=dtype)
                
                return cos, sin
        

        compatible_rotary = KVPressCompatibleRotaryEmbedding(config)
        base_model.rotary_emb = compatible_rotary
        

        for i, layer in enumerate(base_model.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.rotary_emb = compatible_rotary
                
        print("[KVPress] Successfully created and set compatible rotary_emb")
        return True
        
    except Exception as e:
        print(f"[KVPress] patch_qwen_for_kvpress failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """Initialize KV Press compression, simplified HAD version"""
    if not KV_PRESS_AVAILABLE:
        print("[WARNING] KV Press not available, skipping compression initialization")
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


def create_kvpress_adapter(model, press_obj):
    """Create KV Press adapter for Qwen2.5-Omni, based on GTZAN implementation"""
    if press_obj is None:
        return contextlib.nullcontext()
        
    class Qwen2_5OmniKVPressAdapter:
        def __init__(self, original_model, press_object):
            self.original_model = original_model
            self.press_object = press_object
            self.hooks = []
            self.base_model = None
            self.press_method = type(press_object).__name__.lower()
            
        def __enter__(self):
            """Register forward hooks to all self_attn layers"""
            try:

                if hasattr(self.original_model, 'thinker') and hasattr(self.original_model.thinker, 'model'):
                    self.base_model = self.original_model.thinker.model
                elif hasattr(self.original_model, 'model'):
                    self.base_model = self.original_model.model
                else:
                    print("[KVPress Adapter] Warning: Unable to locate base model")
                    self.hooks = []
                    return self
                
                if not hasattr(self.base_model, 'layers'):
                    print("[KVPress Adapter] Warning: Base model has no layers attribute")
                    self.hooks = []
                    return self
                
                layers = self.base_model.layers
                print(f"[KVPress Adapter] Registering hooks for {len(layers)} layers (method: {type(self.press_object).__name__})")
                
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
                        

                        if hasattr(self.press_object, 'forward_hook'):
                            hook = attn_module.register_forward_hook(
                                self.press_object.forward_hook,
                                with_kwargs=True
                            )
                            hooks.append(hook)
                            successful_hooks += 1
                            
                    except Exception as e:
                        print(f"[KVPress Adapter] Layer {layer_idx} hook registration failed: {e}")
                        continue
                
                self.hooks = hooks
                print(f"[KVPress Adapter] Successfully registered {successful_hooks}/{len(layers)} hooks")
                
                return self
                
            except Exception as e:
                print(f"[KVPress Adapter] Error during registration: {e}")
                import traceback
                traceback.print_exc()
                

                for h in getattr(self, 'hooks', []):
                    try:
                        h.remove()
                    except:
                        pass
                        
                self.hooks = []
                return self
                
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
                print(f"[KVPress Adapter] Cleaned up {removed_count} hooks")
                
            self.hooks = []
            

            if exc_type is not None:
                print(f"[KVPress Adapter] Exception detected on exit: {exc_type.__name__}: {exc_val}")
    
    return Qwen2_5OmniKVPressAdapter(model, press_obj)

def main():
    """Main function"""
    args = parse_args()
    

    if not QWEN_AVAILABLE:
        print("[ERROR] Qwen2.5-Omni module not available")
        return
    

    results_dir = RESULTS_DIR_ENV
    os.makedirs(results_dir, exist_ok=True)
    

    output_file = os.path.join(results_dir, f"desed_results_kvpress_{ENV_PRESS_TYPE}_{ENV_COMPRESSION_RATIO}_gpu{gpu_id}.json")
    timing_file = os.path.join(results_dir, f"desed_timing_kvpress_{ENV_PRESS_TYPE}_{ENV_COMPRESSION_RATIO}_gpu{gpu_id}.json")
    
    print(f"Results file: {output_file}")
    print(f"Timing statistics file: {timing_file}")
    

    timing_stats = DESEDTimingStats()
    

    will_use_compression = not args.no_compress and KV_PRESS_AVAILABLE
    

    attention_impl = "eager" if will_use_compression else "flash_attention_2"
    print(f"[Attention] Using {attention_impl} implementation (Compression: {'Enabled' if will_use_compression else 'Disabled'})")
    
    try:

        print("Loading Qwen2.5-Omni model...")
        device_map = {"": 0}
        
        processor = Qwen2_5OmniProcessor.from_pretrained(
            args.model_path, 
            trust_remote_code=True
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_impl,
            trust_remote_code=True
        )
        model.disable_talker()
        
        print("Model loaded successfully")
        

        print("Configuring model audio parameters...")
        

        if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
            if not hasattr(model.thinker.config, 'audio_token_id'):
                model.thinker.config.audio_token_id = _AUDIO_TOKEN_ID
            if not hasattr(model.thinker.config, 'image_token_id'):
                model.thinker.config.image_token_id = 151655
            if not hasattr(model.thinker.config, 'video_token_id'):
                model.thinker.config.video_token_id = 151656
            if not hasattr(model.thinker.config, 'audio_bos_token_id'):
                model.thinker.config.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
            if not hasattr(model.thinker.config, 'audio_eos_token_id'):
                model.thinker.config.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
            if not hasattr(model.thinker.config, 'image_layer_idx'):
                model.thinker.config.image_layer_idx = False
            if not hasattr(model.thinker.config, 'audio_layer_idx'):
                model.thinker.config.audio_layer_idx = None
            if not hasattr(model.thinker.config, 'audio_token_num'):
                model.thinker.config.audio_token_num = None
            if not hasattr(model.thinker.config, 'audio_token_start'):
                model.thinker.config.audio_token_start = None
            if not hasattr(model.thinker.config, 'audio_prune_ratio'):
                model.thinker.config.audio_prune_ratio = 0
            if not hasattr(model.thinker.config, 'random'):
                model.thinker.config.random = False
            if not hasattr(model.thinker.config, 'frame'):
                model.thinker.config.frame = False
            print("[CONFIG] thinker.config audio config params set")
        

        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            if not hasattr(model.thinker.model.config, 'audio_token_id'):
                model.thinker.model.config.audio_token_id = _AUDIO_TOKEN_ID
            if not hasattr(model.thinker.model.config, 'image_token_id'):
                model.thinker.model.config.image_token_id = 151655
            if not hasattr(model.thinker.model.config, 'video_token_id'):
                model.thinker.model.config.video_token_id = 151656
            if not hasattr(model.thinker.model.config, 'audio_bos_token_id'):
                model.thinker.model.config.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
            if not hasattr(model.thinker.model.config, 'audio_eos_token_id'):
                model.thinker.model.config.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
            if not hasattr(model.thinker.model.config, 'image_layer_idx'):
                model.thinker.model.config.image_layer_idx = False
            if not hasattr(model.thinker.model.config, 'audio_layer_idx'):
                model.thinker.model.config.audio_layer_idx = None
            if not hasattr(model.thinker.model.config, 'audio_token_num'):
                model.thinker.model.config.audio_token_num = None
            if not hasattr(model.thinker.model.config, 'audio_token_start'):
                model.thinker.model.config.audio_token_start = None
            if not hasattr(model.thinker.model.config, 'audio_prune_ratio'):
                model.thinker.model.config.audio_prune_ratio = 0
            if not hasattr(model.thinker.model.config, 'random'):
                model.thinker.model.config.random = False
            if not hasattr(model.thinker.model.config, 'frame'):
                model.thinker.model.config.frame = False
            print("[CONFIG] thinker.model.config audio config params set")
        

        timing_stats.record_initial_memory()
        

        if not args.no_compress:
            press_obj = initialize_kv_press(
                model, 
                ENV_PRESS_TYPE, 
                ENV_COMPRESSION_RATIO, 
                args.min_seq_len
            )
        else:
            print("Skipping KV Press compression")
            press_obj = None
        

        print("Loading DESED dataset...")
        dataset = load_desed_qa_dataset(args.dataset_file, args.audio_base_dir)
        
        if not dataset:
            print("Dataset is empty, exiting")
            return
        
        if SAMPLE_LIMIT > 0:
            dataset = dataset[:SAMPLE_LIMIT]
            print(f"Sample limit set to: {len(dataset)}")
        
        print(f"Begin processing {len(dataset)} samples...")
        

        results = []
        correct_predictions = 0
        

        for idx, sample in enumerate(tqdm(dataset, desc="Processing DESED samples")):
            try:

                audio_path = sample["audio_path"]
                

                if not os.path.exists(audio_path):
                    print(f"Skipping missing audio file: {audio_path}")
                    results.append({
                        "sample_id": idx,
                        "task_id": sample["task_id"],
                        "question": sample["question"],
                        "choices": {
                            "A": sample["choice_a"],
                            "B": sample["choice_b"],
                            "C": sample["choice_c"],
                            "D": sample["choice_d"]
                        },
                        "ground_truth": sample["correct_answer"],
                        "prediction": "skip",
                        "raw_response": "file not found",
                        "task_type": sample.get("task_type", "unknown"),
                        "status": "skipped",
                        "reason": "audio_file_not_found"
                    })
                    continue
                
                audio_data = prepare_audio_for_qwen_omni(audio_path)
                if audio_data is None:
                    print(f"Skipping audio processing failed file: {audio_path}")
                    results.append({
                        "sample_id": idx,
                        "task_id": sample["task_id"],
                        "question": sample["question"],
                        "choices": {
                            "A": sample["choice_a"],
                            "B": sample["choice_b"],
                            "C": sample["choice_c"],
                            "D": sample["choice_d"]
                        },
                        "ground_truth": sample["correct_answer"],
                        "prediction": "skip",
                        "raw_response": "audio processing failed",
                        "task_type": sample.get("