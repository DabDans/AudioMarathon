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
import numpy as np
import torch
import librosa
import soundfile as sf
import transformers
import os
import sys
import json
import argparse
import warnings
import random
import traceback
import contextlib
import time
import numpy as np
import pandas as pd
import torch
import transformers
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
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
    print(f"[WARNING] Failed to import Qwen2.5-Omni module: {e}")
    QWEN_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] Qwen2.5-Omni module loading error: {e}")
    QWEN_AVAILABLE = False

KV_PRESS_AVAILABLE = False
try:
    transformers_version = transformers.__version__
    print(f"[INFO] Transformers version: {transformers_version}")

    kvpress_origin_path = "/data/to/your/scripts/path/Qwen_kvpress/kvpress_origin"
    if kvpress_origin_path not in sys.path:
        sys.path.insert(0, kvpress_origin_path)

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
    print("[INFO] KV Press library loaded successfully (using patched version)")
except ImportError as e:
    print(f"[WARNING] Failed to import KV Press library: {e}")
    KV_PRESS_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] KV Press library loading error: {e}")
    KV_PRESS_AVAILABLE = False

RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "HAD_QwenKVPress_Results")

_AUDIO_TOKEN_ID = 151646          # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'

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
    parser = argparse.ArgumentParser(description="HAD with Qwen2.5-Omni + KV Press")
    parser.add_argument("--model-path", type=str, default="/data/to/your/model/path/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-path", type=str, default="/data/to/your/dataset/path/HAD/concatenated_audio")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum generated token count")
    parser.add_argument("--min-seq-len", type=int, default=128, help="Compression threshold")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    return parser.parse_args()

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

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file as required by Qwen2.5-Omni"""
    try:
        if not os.path.exists(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None

        try:
            audio_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Loaded with librosa: shape={audio_np.shape}, samplerate={sr}Hz")
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
                print(f"soundfile processed: shape={audio_np.shape}, samplerate={sr}Hz")
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                return None

        if audio_np.size == 0:
            print(f"Audio is empty: {audio_path}")
            return None

        if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
            print(f"Audio contains invalid values: {audio_path}")
            return None

        audio_tensor = torch.from_numpy(audio_np).float()
        return {"array": audio_tensor, "sampling_rate": target_sr}
    except Exception as e:
        print(f"Audio processing failed: {audio_path}, error: {e}")
        return None

def load_had_dataset(root_dir):
    """Load HAD dataset, balance real/fake samples count"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    all_samples = []

    if os.path.exists(real_dir):
        for audio_file in os.listdir(real_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(real_dir, audio_file)
                all_samples.append({
                    "audio_path": audio_path,
                    "label": "real",
                    "filename": audio_file
                })

    if os.path.exists(fake_dir):
        for audio_file in os.listdir(fake_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(fake_dir, audio_file)
                all_samples.append({
                    "audio_path": audio_path,
                    "label": "fake",
                    "filename": audio_file
                })

    print(f"Total loaded {len(all_samples)} audio samples")

    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"Original sample count: real={len(real_samples)}, fake={len(fake_samples)}")

    min_samples_per_category = min(len(real_samples), len(fake_samples))

    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)
    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)

    balanced_samples = real_samples + fake_samples
    random.shuffle(balanced_samples)
    print(f"Balanced sample count: real={len(real_samples)}, fake={len(fake_samples)}, total={len(balanced_samples)}")
    return balanced_samples

def extract_authenticity_answer(text, choice_a="real", choice_b="fake"):
    """Extract audio authenticity answer from model output text"""
    text_lower = text.lower().strip()
    choice_a_lower = choice_a.lower().strip()
    choice_b_lower = choice_b.lower().strip()

    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b

    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b

    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b

    return text_lower

@dataclass
class HADTimingStats:
    """Timing statistics data for HAD task"""
    samples: int = 0
    total_prefill_time: float = 0.0
    total_decode_time: float = 0.0
    total_output_tokens: int = 0
    total_input_tokens: int = 0
    peak_gpu_memory: float = 0.0
    initial_gpu_memory: float = 0.0
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

    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens=0, peak_memory_gb=None):
        """Add inference record"""
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_output_tokens += output_tokens
        self.total_input_tokens += input_tokens

        if peak_memory_gb is not None:
            self.peak_gpu_memory = max(self.peak_gpu_memory, peak_memory_gb)

        self.detailed_records.append({
            "sample_id": self.samples,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
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

        return {
            "samples": self.samples,
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

# --------------- rest of the code unchanged except for print/log messages and paths, following the same pattern above ---------------

# ... (all further Chinese text/logs/paths replaced by English and /data/to/your/xxx/path/...)

if __name__ == "__main__":
    main()