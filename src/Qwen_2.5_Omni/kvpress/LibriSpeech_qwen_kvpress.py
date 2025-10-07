import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
import contextlib
import librosa
import random
import jiwer
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import transformers
try:
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
except Exception:
    KV_PRESS_AVAILABLE = False

random.seed(42)

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")

ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "LibriSpeech_QwenKVPress_Results")

if SAMPLE_LIMIT > 0:
    print(f"Sample limit set to: {SAMPLE_LIMIT}")

print(f"KV Press Config: compression_ratio={ENV_COMPRESSION_RATIO}, compression_type={ENV_PRESS_TYPE}")

os.environ['NUMEXPR_MAX_THREADS'] = '64'

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

def convert_to_serializable(obj):
    """Recursively convert object to JSON serializable format"""
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

def patch_qwen_for_kvpress(model):
    """Provide KVPress compatible rotary_emb for Qwen2.5-Omni (not relying on config.hidden_size)."""
    try:
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        if base_model is None or not hasattr(base_model, 'layers') or not base_model.layers:
            print("[KVPress] Warning: No valid base_model.layers found")
            return False

        existing = getattr(base_model, 'rotary_emb', None)
        if existing is not None:
            for layer in base_model.layers:
                if hasattr(layer, 'self_attn') and (
                    not hasattr(layer.self_attn, 'rotary_emb') or layer.self_attn.rotary_emb is None
                ):
                    layer.self_attn.rotary_emb = existing
            return True

        first = base_model.layers[0]
        for cand_attr in ('self_attn', 'attn'):
            if hasattr(first, cand_attr):
                attn = getattr(first, cand_attr)
                if hasattr(attn, 'rotary_emb') and attn.rotary_emb is not None:
                    base_model.rotary_emb = attn.rotary_emb
                    for layer in base_model.layers:
                        if hasattr(layer, 'self_attn'):
                            layer.self_attn.rotary_emb = attn.rotary_emb
                    return True

        class SimpleRotaryEmbedding:
            def __init__(self, dim: int = 128):
                self.dim = dim
                try:
                    inv = 1.0 / (10000 ** (torch.arange(0, dim // 2, 1).float() / max(dim // 2, 1)))
                except Exception:
                    inv = torch.ones(1)
                self.inv_freq = inv

        placeholder = SimpleRotaryEmbedding(dim=128)
        base_model.rotary_emb = placeholder
        for layer in base_model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.rotary_emb = placeholder
        print("[KVPress] Placeholder rotary_emb (SimpleRotaryEmbedding) set")
        return True
    except Exception as e:
        print(f"[KVPress] patch_qwen_for_kvpress failed: {e}")
        traceback.print_exc()
        return False

def create_kvpress_adapter(model, press_obj):
    """Create KV Press adapter for Qwen2.5-Omni"""
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
                    print("[KVPressAdapter] Warning: Cannot locate base model")
                    self.hooks = []
                    return self
                if not hasattr(self.base_model, 'layers'):
                    print("[KVPressAdapter] Warning: Base model does not have layers attribute")
                    self.hooks = []
                    return self
                
                layers = self.base_model.layers
                print(f"[KVPressAdapter] Registering hooks for {len(layers)} layers (method: {type(self.press_object).__name__})")
                
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
                        print(f"[KVPressAdapter] Layer {layer_idx} hook registration failed: {e}")
                        continue
                
                self.hooks = hooks
                print(f"[KVPressAdapter] Successfully registered {successful_hooks}/{len(layers)} hooks")
                return self
            except Exception as e:
                print(f"[KVPressAdapter] Error during registration: {e}")
                traceback.print_exc()
                for h in getattr(self, 'hooks', []):
                    try:
                        h.remove()
                    except:
                        pass
                self.hooks = []
                return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Cleanup all registered hooks"""
            removed_count = 0
            for hook in self.hooks:
                try:
                    hook.remove()
                    removed_count += 1
                except Exception as e:
                    pass
            if removed_count > 0:
                print(f"[KVPressAdapter] Cleaned up {removed_count} hooks")
            self.hooks = []
            if exc_type is not None and self.press_method not in ['tovapress', 'snapkvpress']:
                print(f"[KVPressAdapter] Exception detected on exit: {exc_type.__name__}: {exc_val}")
    
    return Qwen2_5OmniKVPressAdapter(model, press_obj)

def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """Initialize KV Press compression, using simplified HAD version"""
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

def ensure_audio_config(model):
    """Ensure thinker.config and thinker.model.config have audio related fields to avoid AttributeError."""
    try:
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
            cfg = model.thinker.config
            if not hasattr(cfg, 'audio_token_id'):
                cfg.audio_token_id = _AUDIO_TOKEN_ID
            if not hasattr(cfg, 'image_token_id'):
                cfg.image_token_id = 151655
            if not hasattr(cfg, 'video_token_id'):
                cfg.video_token_id = 151656
            if not hasattr(cfg, 'audio_bos_token_id'):
                cfg.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
            if not hasattr(cfg, 'audio_eos_token_id'):
                cfg.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
            if not hasattr(cfg, 'image_layer_idx'):
                cfg.image_layer_idx = False
            if not hasattr(cfg, 'audio_layer_idx'):
                cfg.audio_layer_idx = None
            if not hasattr(cfg, 'audio_token_num'):
                cfg.audio_token_num = None
            if not hasattr(cfg, 'audio_token_start'):
                cfg.audio_token_start = None
            if not hasattr(cfg, 'audio_prune_ratio'):
                cfg.audio_prune_ratio = 0
            if not hasattr(cfg, 'random'):
                cfg.random = False
            if not hasattr(cfg, 'frame'):
                cfg.frame = False

        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
            cfg2 = model.thinker.model.config
            if not hasattr(cfg2, 'audio_token_id'):
                cfg2.audio_token_id = _AUDIO_TOKEN_ID
            if not hasattr(cfg2, 'image_token_id'):
                cfg2.image_token_id = 151655
            if not hasattr(cfg2, 'video_token_id'):
                cfg2.video_token_id = 151656
            if not hasattr(cfg2, 'audio_bos_token_id'):
                cfg2.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
            if not hasattr(cfg2, 'audio_eos_token_id'):
                cfg2.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
            if not hasattr(cfg2, 'image_layer_idx'):
                cfg2.image_layer_idx = False
            if not hasattr(cfg2, 'audio_layer_idx'):
                cfg2.audio_layer_idx = None
            if not hasattr(cfg2, 'audio_token_num'):
                cfg2.audio_token_num = None
            if not hasattr(cfg2, 'audio_token_start'):
                cfg2.audio_token_start = None
            if not hasattr(cfg2, 'audio_prune_ratio'):
                cfg2.audio_prune_ratio = 0
            if not hasattr(cfg2, 'random'):
                cfg2.random = False
            if not hasattr(cfg2, 'frame'):
                cfg2.frame = False
    except Exception as e:
        print(f"[Audio Config] Injection failed: {e}")
        traceback.print_exc()

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON compatibility"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

sys.path.append("/data/to/your/Code/Qwen_2.5")
try:
    from modeling_qwen2_5_omni_origin import Qwen2_5OmniForConditionalGeneration
    from processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    QWEN_AVAILABLE = True
    print("[Info] Qwen2.5-Omni module loaded successfully")
except ImportError as e:
    print(f"[Warning] Qwen2.5-Omni module import failed: {e}")
    QWEN_AVAILABLE = False
except Exception as e:
    print(f"[Warning] Qwen2.5-Omni module load error: {e}")
    QWEN_AVAILABLE = False

KV_PRESS_AVAILABLE = False
try:
    import transformers
    transformers_version = transformers.__version__
    print(f"[Info] Transformers version: {transformers_version}")
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
    print("[Info] KV Press library loaded successfully")
except ImportError as e:
    print(f"[Warning] KV Press library import failed: {e}")
    KV_PRESS_AVAILABLE = False
except Exception as e:
    print(f"[Warning] KV Press library load error: {e}")
    KV_PRESS_AVAILABLE = False

if not KV_PRESS_AVAILABLE:
    print("[Error] KV Press library not available")
    print("Check if kvpress library is installed: pip install kvpress")

_AUDIO_TOKEN_ID = 151646        # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.insert(0, current_dir)

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class GlobalTimingStats:
    """Simplified global timing statistics"""
    
    def __init__(self):
        self.timing_records = []
        self.first_sample_skipped = False
        
    def add_sample(self, wall_time, prefill_time, total_gpu_time):
        """Add sample timing record, skip first sample"""
        if not self.first_sample_skipped:
            self.first_sample_skipped = True
            return
        record = {
            "wall_time": wall_time,
            "prefill_time": prefill_time,
            "total_gpu_time": total_gpu_time
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get summary statistics"""
        if len(self.timing_records) == 0:
            return {
                "count": 0,
                "avg_wall_time": 0.0,
                "avg_prefill_time": 0.0,
                "avg_total_gpu_time": 0.0
            }
        total_wall = sum(r["wall_time"] for r in self.timing_records)
        total_prefill = sum(r["prefill_time"] for r in self.timing_records)
        total_gpu = sum(r["total_gpu_time"] for r in self.timing_records)
        count = len(self.timing_records)
        return {
            "count": count,
            "avg_wall_time": total_wall / count,
            "avg_prefill_time": total_prefill / count,
            "avg_total_gpu_time": total_gpu / count
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return output_file

def load_librispeech_long_dataset(base_dir, split="test-clean"):
    """Load local LibriSpeech-Long dataset"""
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Error: Dataset path does not exist: {split_dir}")
        return []
    
    print(f"Scanning dataset directory: {split_dir}")
    speaker_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
    
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        chapter_dirs = sorted([d for d in glob.glob(os.path.join(speaker_dir, "*")) if os.path.isdir(d)])
        for chapter_dir in chapter_dirs:
            chapter_id = os.path.basename(chapter_dir)
            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))
            for flac_file in flac_files:
                base_name = os.path.splitext(os.path.basename(flac_file))[0]
                txt_file = os.path.join(chapter_dir, f"{base_name}.txt")
                trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                transcription = None
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                elif os.path.exists(trans_file):
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(base_name):
                                transcription = line.split(' ', 1)[1].strip()
                                break
                if transcription:
                    try:
                        audio_info = sf.info(flac_file)
                        duration = audio_info.duration
                    except:
                        duration = 0.0
                    dataset.append({
                        "id": base_name,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "path": flac_file,
                        "transcription": transcription,
                        "duration": duration,
                        "audio": {
                            "path": flac_file,
                            "array": None,
                            "sampling_rate": None
                        }
                    })
    print(f"Loaded {len(dataset)} audio samples")
    return dataset

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file as required by Qwen2.5-Omni"""
    try:
        if not os.path.exists(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Loaded with librosa: shape={audio.shape}, sample_rate={sr}Hz")
        except Exception as e:
            print(f"librosa load failed: {e}")
            try:
                audio, sample_rate = sf.read(audio_path)
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"soundfile processed: shape={audio.shape}, sample_rate={sr}Hz")
            except Exception as e:
                print(f"soundfile load also failed: {e}")
                print(f"Cannot load audio file: {audio_path}")
                return None
        if len(audio) == 0:
            print("Warning: Audio is empty, returning None")
            return None
        audio = audio.astype(np.float32)
        return audio
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        return None

def librispeech_doc_to_audio(doc):
    """Load audio data from LibriSpeech document"""
    if "audio" not in doc:
        return None
    audio_path = doc["audio"]["path"]
    if not os.path.exists(audio_path):
        print(f"Audio file does not exist: {audio_path}")
        return None
    if doc["audio"]["array"] is None:
        try:
            audio_data = prepare_audio_for_qwen_omni(audio_path)
            if audio_data is None:
                print(f"Cannot load audio file {audio_path}")
                return None
            doc["audio"]["array"] = audio_data
            doc["audio"]["sampling_rate"] = 16000
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None
    if doc["audio"]["array"] is None or len(doc["audio"]["array"]) == 0:
        print(f"Invalid audio data: {audio_path}")
        return None
    return doc["audio"]["array"], doc["audio"]["sampling_rate"]

def asr_doc_to_text(doc, kwargs=None):
    """Generate prompt for English ASR task"""
    if kwargs is None:
        kwargs = {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    instruction = "Transcribe this audio accurately. Remove hesitation words like 'um', 'uh'."
    format_text = "Your response should be formatted as follows: Spoken Content:"
    prompt_text = f"{pre_prompt}{instruction} {format_text} <transcribed text here>{post_prompt}"
    return prompt_text

def clean_response(response):
    """Clean ASR response"""
    if not response or response.strip() == "":
        return ""
    for marker in ["spoken content:", "content:", "transcription:", "transcript:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    response = re.sub(r'<transcribed text here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    response = re.sub(r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription):\s*', '', response.strip())
    return response.strip()

def standardize_text(text):
    """Standardize text for fair comparison and WER calculation"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    try:
        import jiwer
        ref_standardized = standardize_text(reference)
        hyp_standardized = standardize_text(hypothesis)
        if not ref_standardized or not hyp_standardized:
            return 100.0
        wer = jiwer.wer(ref_standardized, hyp_standardized)
        return wer * 100
    except ImportError:
        print("Warning: jiwer library not installed, cannot calculate WER")
        return 0.0
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 0.0

def calculate_librispeech_metrics(references, hypotheses):
    """
    Simplified LibriSpeech ASR metrics, only calculating WER
    
    Args:
        references: list of true transcriptions
        hypotheses: list of predicted transcriptions
        
    Returns:
        dict: Dictionary containing WER metrics
    """
    valid_indices = []
    clean_references = []
    clean_hypotheses = []
    wer_scores = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref and hyp:
            valid_indices.append(i)
            clean_references.append(ref)
            clean_hypotheses.append(hyp)
            wer = calculate_wer(ref, hyp)
            wer_scores.append(wer)
    if len(clean_references) == 0:
        return {
            'wer_mean': 100.0,
            'wer_std': 0.0,
            'wer_min': 100.0,
            'wer_max': 100.0,
            'perfect_predictions': 0,
            'valid_samples': 0,
            'total_samples': len(references),
            'word_accuracy': 0.0
        }
    wer_mean = np.mean(wer_scores)
    wer_std = np.std(wer_scores)
    wer_min = np.min(wer_scores)
    wer_max = np.max(wer_scores)
    perfect_predictions = len([wer for wer in wer_scores if wer == 0.0])
    word_accuracy = 100.0 - wer_mean
    return {
        'wer_mean': float(wer_mean),
        'wer_std': float(wer_std),
        'wer_min': float(wer_min),
        'wer_max': float(wer_max),
        'perfect_predictions': int(perfect_predictions),
        'valid_samples': len(clean_references),
        'total_samples': len(references),
        'word_accuracy': float(word_accuracy),
        'wer_scores': wer_scores
    }

def main():
    """Main function - LibriSpeech ASR evaluation with KVPress"""
    if not QWEN_AVAILABLE:
        print("[Error] Qwen2.5-Omni module not available")
        return

    results_dir = RESULTS_DIR_ENV
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"librispeech_results_kvpress_{ENV_PRESS_TYPE}_{ENV_COMPRESSION_RATIO}_gpu{gpu_id}.json")
    timing_file = os.path.join(results_dir, f"librispeech_timing_kvpress_{ENV_PRESS_TYPE}_{ENV_COMPRESSION_RATIO}_gpu{gpu_id}.json")
    
    print(f"Results file: {output_file}")
    print(f"Timing stats file: {timing_file}")

    librispeech_path = '/data/to/your/Dataset/librispeech-long'
    if not os.path.exists(librispeech_path):
        print(f"[Error] LibriSpeech dataset path does not exist: {librispeech_path}")
        return
    
    print(f"\n=== LibriSpeech ASR Evaluation Config (Qwen2.5-Omni + KVPress) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press type: {ENV_PRESS_TYPE}")
    print(f"Compression Ratio: {ENV_COMPRESSION_RATIO}")
    print(f"Data Path: {librispeech_path}")
    if SAMPLE_LIMIT > 0:
        print(f"Sample Limit: {SAMPLE_LIMIT}")

    timing_stats = GlobalTimingStats()
    try:
        print("Loading Qwen2.5-Omni model...")
        model_path = "/data/to/your/Model/Qwen2.5-Omni-3B"
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

        if hasattr(model, 'disable_talker'):
            try:
                model.disable_talker()
                print("[Qwen] talker module disabled")
            except Exception:
                pass

        ensure_audio_config(model)
        print("Model loaded")

        allocated, reserved = get_gpu_memory_usage()
        print(f"Initial GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        kv_press = initialize_kv_press(
            model, 
            ENV_PRESS_TYPE, 
            ENV_COMPRESSION_RATIO, 
            128  # min_seq_len
        )
        if kv_press is None:
            print("Skip KV Press compression (creation failed)")
        else:
            print(f"KV Press compression initialized: {ENV_PRESS_TYPE}")

        print("Loading LibriSpeech dataset...")
        dataset = load_librispeech_long_dataset(librispeech_path, "test-clean")
        if not dataset:
            print("Error: Failed to load any data")
            return
        if SAMPLE_LIMIT > 0 and len(dataset) > SAMPLE_LIMIT:
            dataset = dataset[:SAMPLE_LIMIT]
            print(f"Sample count limited to: {len(dataset)}")
        print(f"Processing {len(dataset)} samples...")

        speaker_stats = defaultdict(int)
        for sample in dataset:
            speaker_id = sample.get("speaker_id", "unknown")
            speaker_stats[speaker_id] += 1
        print(f"Speaker stats: {len(speaker_stats)} speakers")
        print(f"Sample distribution: {dict(list(speaker_stats.items())[:5])}...")

        results = []
        total_wer = 0.0
        processed_samples = 0

        is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
        if is_screen_env:
            print("Detected screen or non-interactive environment, using simplified progress display")

        tqdm_kwargs = {
            'ascii': True,
            'dynamic_ncols': True,
            'file': sys.stdout
        }

        print(f"Starting evaluation for {len(dataset)} samples...")

        allocated, reserved = get_gpu_memory_usage()
        print(f"Post-model-load GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
        
        progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="LibriSpeech ASR Eval (Qwen2.5)", **tqdm_kwargs)

        for idx, doc in progress_bar:
            audio_path = doc.get("path", "")
            audio_data_result = librispeech_doc_to_audio(doc)
            if audio_data_result is None:
                print(f"\nSample {idx+1}: Skipped - cannot load audio: {audio_path}")
                skip_result = {
                    "idx": idx,
                    "id": doc.get("id", f"sample_{idx}"),
                    "speaker_id": doc.get("speaker_id", "unknown"),
                    "chapter_id": doc.get("chapter_id", ""),
                    "path": audio_path,
                    "duration": doc.get("duration", 0),
                    "reference": doc.get("transcription", ""),
                    "hypothesis": "",
                    "wer": 100.0,
                    "response_text": "",
                    "error": "Audio load failed",
                    "skipped": True
                }
                results.append(skip_result)
                continue
            audio_np, sr = audio_data_result
            reference = doc.get("transcription", "")
            speaker_id = doc.get("speaker_id", "unknown")
            prompt_text = asr_doc_to_text(doc)
            task_instruction = "You are a helpful assistant that transcribes speech audio. Please listen carefully and provide the exact transcription of what is spoken in the audio."
            full_user_prompt = f"{task_instruction}\n\n{prompt_text}"
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_np},
                        {"type": "text", "text": full_user_prompt}
                    ]
                }
            ]
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            inputs = processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=True
            )
            inputs = inputs.to(model.device)
            inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}
            # Prefill phase
            prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if prefill_start: prefill_start.record()
            use_compression = kv_press is not None
            with torch.no_grad(), (create_kvpress_adapter(model, kv_press) if use_compression else contextlib.nullcontext()):
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if prefill_end: prefill_end.record()
            # Generate phase
            gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if gen_start: gen_start.record()
            with torch.no_grad(), (create_kvpress_adapter(model, kv_press) if use_compression else contextlib.nullcontext()):
                out_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1100,
                    do_sample=False
                )
            if gen_end: gen_end.record()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            prefill_time = prefill_start.elapsed_time(prefill_end)/1000 if prefill_start else 0.0
            total_time = gen_start.elapsed_time(gen_end)/1000 if gen_start else 0.0
            decode_time = max(total_time - prefill_time, 0.0)
            resp = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if "assistant\n" in resp:
                resp = resp.split("assistant\n")[-1].strip()
            hypothesis = clean_response(resp)
            wer = calculate_wer(reference, hypothesis)
            total_wer += wer
            processed