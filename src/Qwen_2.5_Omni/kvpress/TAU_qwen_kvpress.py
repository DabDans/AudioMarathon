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
import pandas as pd
import soundfile as sf
import numpy as np
import torch
import librosa
import transformers
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score


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
    print(f"[WARNING] Qwen2.5-Omni module load error: {e}")
    QWEN_AVAILABLE = False


KV_PRESS_AVAILABLE = False
try:

    import transformers
    transformers_version = transformers.__version__
    print(f"[INFO] Transformers version: {transformers_version}")
    

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
        print("[INFO] KV Press library loaded successfully")
    except ImportError as e:
        print(f"[WARNING] Failed to import KV Press library: {e}")
        print("[INFO] Trying to import from local kvpress directory...")
        

        try:
            import sys
            kvpress_path = os.path.join(os.path.dirname(__file__), "..", "kvpress")
            if os.path.exists(kvpress_path):
                sys.path.insert(0, kvpress_path)
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
                print("[INFO] Successfully loaded KV Press from local directory")
            else:
                print(f"[ERROR] Local kvpress directory does not exist: {kvpress_path}")
        except Exception as local_e:
            print(f"[ERROR] Local kvpress import failed: {local_e}")

except ImportError as e:
    print(f"[WARNING] Failed to import Transformers library: {e}")
    KV_PRESS_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] KV Press library load error: {e}")
    KV_PRESS_AVAILABLE = False


if not KV_PRESS_AVAILABLE:
    print("[ERROR] KV Press library not available but compression is required")
    print("Please check the following:")
    print("1. Is kvpress library installed: pip install kvpress")
    print("2. Is transformers version compatible")
    print("3. Is local kvpress directory present")




_AUDIO_TOKEN_ID = 151646          # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'


TAU_SCENE_CLASSES = [
    "airport", "shopping_mall", "metro_station", "street_pedestrian",
    "public_square", "street_traffic", "tram", "bus", "metro", "park"
]


ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "TAU_QwenKVPress_Results")


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
    parser = argparse.ArgumentParser(description="TAU with Qwen2.5-Omni + KV Press")
    parser.add_argument("--model-path", type=str, default="/data/to/your/model/path/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-path", type=str, default="/data/to/your/dataset/path/TAU/concatenated_resampled")
    parser.add_argument("--meta-file", type=str, default="acoustic_scene_task_meta.json", help="Metadata filename")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum number of generated tokens")
    parser.add_argument("--min-seq-len", type=int, default=128, help="Compression threshold")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    return parser.parse_args()


def convert_to_serializable(obj):
    """Recursively convert object to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, '__dict__'):

        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    else:
        return obj

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file as required by Qwen2.5-Omni, refer to TAU_qwen2.5.py"""
    
    try:

        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Loaded with librosa: shape={audio.shape}, sample_rate={sr}Hz")
        except Exception as e:
            print(f"Librosa load failed: {e}")
            

            try:
                audio, sample_rate = sf.read(audio_path)
                

                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                

                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"Soundfile processed: shape={audio.shape}, sample_rate={sr}Hz")
                
            except Exception as e:
                print(f"Soundfile load also failed: {e}")

                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generated silent substitute audio")
        

        if len(audio) == 0:
            print("Warning: Audio is empty, creating 3 seconds of silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            

        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def load_tau_acoustic_scene_dataset(root_dir):
    """Load acoustic scene classification task from TAU dataset, refer to TAU_qwen2.5.py"""

    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    

    scene_counts = {}
    

    for item in metadata:

        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)
        

        if not os.path.exists(wav_path):
            print(f"Warning: File does not exist {wav_path}")
            continue
        

        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"] # A, B, C, D
        

        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        

        all_samples.append({
            "scene_label": scene_label,
            "audio_path": wav_path,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification",
            "id": item.get("uniq_id", f"tau_{os.path.basename(wav_path)}"),
            "filename": os.path.basename(wav_path),
            "duration": item.get("duration", 0),
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    

    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count} samples ({count/len(all_samples)*100:.1f}%)")
    

    if SAMPLE_LIMIT > 0 and SAMPLE_LIMIT < len(all_samples):
        print(f"Due to sample limit set, randomly selecting {SAMPLE_LIMIT} samples for evaluation")
        all_samples = random.sample(all_samples, SAMPLE_LIMIT)
        

    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """Extract acoustic scene answer option (A/B/C/D) from model output text, refer to TAU_qwen2.5.py"""
    text_lower = text.lower().strip()
    

    options = ['a', 'b', 'c', 'd']
    

    if text_lower in options:
        return text_lower.upper()
    

    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    

    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    

    if choices:
        best_match = None
        max_overlap = 0
        
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()

            if choice_lower in text_lower:
                return chr(65 + i)  # A, B, C, D
            

            keywords = choice_lower.split(' - ')[0].split()
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        
        if best_match and max_overlap > 1:
            return best_match
    

    return ""

def calculate_acoustic_metrics(predictions, ground_truths, scene_labels):
    """Calculate acoustic scene classification metrics: Accuracy, Precision, Recall, F1 score, refer to TAU_qwen2.5.py"""

    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in scene_labels and t in scene_labels]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'valid_samples': 0,
            'total_samples': len(predictions)
        }
    
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    

    label_map = {label: idx for idx, label in enumerate(sorted(scene_labels))}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions),
        'label_mapping': label_map
    }


def load_tau_dataset_backup(dataset_path, meta_file="meta.csv"):
    """Load TAU dataset, support multiple metadata formats, refer to implementation in TAU_qwen_dart.py"""
    meta_path = os.path.join(dataset_path, meta_file)
    

    json_meta_path = os.path.join(dataset_path, "acoustic_scene_task_meta.json")
    if os.path.exists(json_meta_path):
        print(f"[Load] Found JSON metadata file: {json_meta_path}")
        try:
            with open(json_meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            dataset = []
            missing_files = 0
            scene_counts = Counter()
            
            for item in metadata:
                filename = item.get("path", item.get("filename", ""))
                scene_label = item.get("scene_label", "")
                
                audio_path = os.path.join(dataset_path, filename)
                if not os.path.exists(audio_path):
                    missing_files += 1
                    continue
                
                try:
                    audio_info = sf.info(audio_path)
                    duration = audio_info.duration
                    sample_rate = audio_info.samplerate
                except Exception as e:
                    continue
                
                scene_counts[scene_label] += 1
                

                choices = [
                    item.get("choice_a", ""),
                    item.get("choice_b", ""), 
                    item.get("choice_c", ""),
                    item.get("choice_d", "")
                ]
                
                dataset_item = {
                    "filename": filename,
                    "audio_path": audio_path,
                    "scene_label": scene_label,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "id": item.get("uniq_id", f"tau_{filename}"),
                    "question": item.get("question", "Listen to this audio and identify the acoustic scene. Choose the most appropriate option."),
                    "choices": choices,
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2], 
                    "choice_d": choices[3],
                    "answer_gt": item.get("answer_gt", ""),
                    "correct_answer": item.get("answer_gt", ""),
                    "task": "Acoustic_Scene_Classification"
                }
                dataset.append(dataset_item)
            
            print(f"[Load] Loaded {len(dataset)} samples from JSON")
            print(f"[Stats] Scene distribution: {dict(scene_counts)}")
            return dataset
            
        except Exception as e:
            print(f"[ERROR] JSON metadata load failed: {e}")
    

    if not os.path.exists(meta_path):
        print(f"[ERROR] Metadata file does not exist: {meta_path}")
        return []
    
    try:
        df = pd.read_csv(meta_path, sep='\t')
    except Exception as e:
        print(f"[ERROR] Failed to read metadata: {e}")
        return []
    
    dataset = []
    missing_files = 0
    
    for _, row in df.iterrows():
        filename = row['filename']
        scene_label = row['scene_label']
        
        audio_path = os.path.join(dataset_path, filename)
        if not os.path.exists(audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"[WARNING] Audio does not exist: {audio_path}")
            continue
        
        try:
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"[ERROR] Cannot read audio: {audio_path}")
            continue
        
        item = {
            "filename": filename,
            "audio_path": audio_path,
            "scene_label": scene_label,
            "duration": duration,
            "sample_rate": sample_rate,
            "id": f"tau_{filename}",
            "choices": [],
            "task": "Acoustic_Scene_Classification"
        }
        dataset.append(item)
    
    if missing_files > 5:
        print(f"[INFO] Total {missing_files} audio files missing")
    
    print(f"[Load] Successfully loaded {len(dataset)} samples")
    

    scene_counts = Counter([item["scene_label"] for item in dataset])
    print(f"[Stats] Scene distribution: {dict(scene_counts)}")
    
    return dataset


def extract_scene_prediction(response, scene_classes=TAU_SCENE_CLASSES):
    """Extract scene prediction from model output, support multiple formats"""
    if not response:
        return ""
    
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    
    response_lower = response.lower().strip()
    

    choice_options = ['a', 'b', 'c', 'd']
    for opt in choice_options:
        if response_lower == opt or response_lower.startswith(f"{opt}.") or response_lower.startswith(f"{opt})"):
            return opt.upper()
    

    for scene in scene_classes:
        if scene.lower() in response_lower:
            return scene
    

    for scene in scene_classes:
        scene_spaced = scene.replace("_", " ")
        if scene_spaced.lower() in response_lower:
            return scene
    

    words = response_lower.split()
    if words:
        first_word = words[0].strip('.,!?;:')

        for scene in scene_classes:
            if scene.lower().startswith(first_word) or first_word in scene.lower():
                return scene
        return first_word
    
    return ""




@dataclass
class TAUSampleResult:
    id: str
    audio_path: str
    filename: str
    ground_truth_scene: str
    predicted_scene: str
    is_correct: bool
    raw_response: str
    timing: Dict[str, Any]

class TAUTimingStats:
    def __init__(self):
        self.records = []
        self.scene_stats = defaultdict(list)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
            self.initial_mem = torch.cuda.memory_allocated()

    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, scene_label=None):
        peak_mem = torch.cuda.max_memory_allocated() if self.cuda else 0
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "scene_label": scene_label,
            "gpu_memory_peak_gb": peak_mem / 1024**3 if peak_mem else 0
        }
        self.records.append(record)
        if scene_label:
            self.scene_stats[scene_label].append(record)

    def get_summary(self):
        if not self.records:
            return {"error": "No samples processed"}
        
        df = pd.DataFrame(self.records)
        
        overall_summary = {
            "total_samples": len(df),
            "avg_prefill_time": df.prefill_time.mean(),
            "avg_decode_time": df.decode_time.mean(),
            "avg_total_time": df.total_time.mean(),
            "avg_tokens_per_sec": df.tokens_per_sec.mean(),
            "total_tokens": int(df.output_tokens.sum()),
            "avg_audio_duration": df.audio_duration.mean() if 'audio_duration' in df.columns else 0,
            "avg_gpu_peak_mem_gb": df.gpu_memory_peak_gb.mean()
        }
        
        scene_summaries = {}
        for scene, records in self.scene_stats.items():
            if records:
                scene_df = pd.DataFrame(records)
                scene_summaries[scene] = {
                    "samples": len(records),
                    "avg_prefill_time": scene_df.prefill_time.mean(),
                    "avg_decode_time": scene_df.decode_time.mean(),
                    "avg_total_time": scene_df.total_time.mean(),
                    "avg_tokens_per_sec": scene_df.tokens_per_sec.mean()
                }
        
        return {
            "overall_summary": overall_summary,
            "scene_summaries": scene_summaries
        }


def calculate_tau_metrics(y_true, y_pred, scene_classes=TAU_SCENE_CLASSES):

    clean_true, clean_pred = [], []
    for t, p in zip(y_true, y_pred):
        if t and p:
            clean_true.append(t)
            clean_pred.append(p)
    
    if not clean_true:
        return {"accuracy": 0.0}
    

    accuracy = accuracy_score(clean_true, clean_pred)
    

    precision, recall, f1, support = precision_recall_fscore_support(
        clean_true, clean_pred, labels=scene_classes, average=None, zero_division=0
    )
    

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_true, clean_pred, average='macro', zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        clean_true, clean_pred, average='weighted', zero_division=0
    )
    

    cm = confusion_matrix(clean_true, clean_pred, labels=scene_classes)
    

    report = classification_report(
        clean_true, clean_pred, 
        labels=scene_classes,
        target_names=scene_classes,
        zero_division=0,
        digits=4
    )
    

    per_class_metrics = {}
    for i, scene in enumerate(scene_classes):
        per_class_metrics[scene] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_w),
        'recall_weighted': float(rec_w),
        'f1_weighted': float(f1_w),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'valid_samples': len(clean_true),
        'total_samples': len(y_true)
    }


def patch_qwen_for_kvpress(model):
    """Add rotary_emb compatibility for Qwen2.5-Omni, refer to TAU_kvpress.py implementation"""

    try_paths = []
    

    if hasattr(model, 'thinker'):
        try_paths.append(model.thinker)
        if hasattr(model.thinker, 'model'):
            try_paths.append(model.thinker.model)
    

    if hasattr(model, 'model'):
        try_paths.append(model.model)
    try_paths.append(model)
    
    base = None
    for cand in try_paths:
        if hasattr(cand, 'layers'):
            base = cand
            print(f"[PATCH] Found base model: {type(cand).__name__}")
            break
    
    if base is None:
        print("[WARNING] Cannot find base structure, KV Press may not work correctly")
        return False
    

    if hasattr(base, 'rotary_emb') and base.rotary_emb is not None:
        print("[PATCH] Model already has global rotary_emb attribute")
        base._kvpress_patched = True
        return True
    

    if hasattr(base, 'layers') and len(base.layers) > 0:
        for layer_idx, layer in enumerate(base.layers):

            attn_layer = None
            for attr in ['self_attn', 'attn']:
                if hasattr(layer, attr):
                    attn_layer = getattr(layer, attr)
                    break
            
            if attn_layer and hasattr(attn_layer, 'rotary_emb') and attn_layer.rotary_emb is not None:
                base.rotary_emb = attn_layer.rotary_emb
                print(f"[PATCH] Extracted and added global rotary_emb attribute from layer {layer_idx}")
                base._kvpress_patched = True
                return True
    

    try:

        config = None
        if hasattr(model, 'config'):
            if hasattr(model.config, 'thinker_config') and hasattr(model.config.thinker_config, 'text_config'):
                config = model.config.thinker_config.text_config
            elif hasattr(model.config, 'text_config'):
                config = model.config.text_config
            elif hasattr(model.config, 'hidden_size'):
                config = model.config
        

        if config and hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
        else:

            hidden_size = 3584
            num_heads = 28
            print(f"[PATCH] Using default config parameters")
        
        head_dim = hidden_size // num_heads
        
        class SimpleRotaryEmbedding:
            def __init__(self, dim, max_position_embeddings=32768, base=10000):
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
                self.base = base
        
        base.rotary_emb = SimpleRotaryEmbedding(dim=head_dim)
        base._kvpress_patched = True
        print(f"[PATCH] Created placeholder rotary_emb, head_dim={head_dim}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create placeholder rotary_emb: {e}")
        return False

def create_kvpress_adapter(model, press_obj):
    """Create KV Press adapter for Qwen2.5-Omni, consistent hook method with HAD_qwen_kvpress"""
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

                        attn_module = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                        if attn_module is None:
                            continue


                        if base_rotary_emb and (not hasattr(attn_module, 'rotary_emb') or attn_module.rotary_emb is None):
                            attn_module.rotary_emb = base_rotary_emb

                        if hasattr(self.press_object, 'forward_hook'):
                            hook = attn_module.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
                            hooks.append(hook)
                            successful_hooks += 1
                    except Exception as e:
                        print(f"[KVPress Adapter] Layer {layer_idx} hook registration failed: {e}")
                        continue

                self.hooks = hooks
                print(f"[KVPress Adapter] Successfully registered {successful_hooks}/{len(layers)} hooks")
                return self

            except Exception as e:
                print(f"[KVPress Adapter] Registration error: {e}")
                traceback.print_exc()
                for h in getattr(self, 'hooks', []):
                    try:
                        h.remove()
                    except:
                        pass
                self.hooks = []
                return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            removed_count = 0
            for hook in self.hooks:
                try:
                    hook.remove()
                    removed_count += 1
                except Exception:
                    pass
            if removed_count:
                print(f"[KVPress Adapter] Cleaned up {removed_count} hooks")
            self.hooks = []

    return Qwen2_5OmniKVPressAdapter(model, press_obj)

def verify_kvpress_compatibility(model, press_type):
    """Verify compatibility for specific KVPress method (ported from HAD_qwen_kvpress)"""
    try:
        print(f"[KVPress] Verifying {press_type} method compatibility...")
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            print("[KVPress] Compatibility verification failed: Unable to locate base model")
            return False

        if not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
            print("[KVPress] Compatibility verification failed: Base model has no valid layers")
            return False

        if press_type in ['snap', 'tova']:
            global_rotary = getattr(base_model, 'rotary_emb', None)
            if global_rotary is None:
                print(f"[KVPress] {press_type} compatibility verification failed: missing global rotary_emb")
                return False
            layer_check_count = min(3, len(base_model.layers))
            for i in range(layer_check_count):
                layer = base_model.layers[i]
                attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                if attn is None or getattr(attn, 'rotary_emb', None) is None:
                    print(f"[KVPress] {press_type} compatibility verification failed: layer {i} missing rotary_emb/self_attn")
                    return False

        first_layer = base_model.layers[0]
        attn = getattr(first_layer, 'self_attn', None) or getattr(first_layer, 'attn', None)
        if attn is None:
            print("[KVPress] Compatibility verification failed: first layer has no attention module")
            return False
        for attr in ['q_proj', 'k_proj', 'v_proj']:
            if not hasattr(attn, attr):
                print(f"[KVPress] Compatibility verification failed: attention missing {attr}")
                return False
        print(f"[KVPress] {press_type} compatibility verification passed")
        return True
    except Exception as e:
        print(f"[KVPress] {press_type} compatibility verification error: {e}")
        return False

def verify_tova_multimodal_compatibility(model):
    """Verify TOVA compatibility with multimodal input (ported from HAD_qwen_kvpress)"""
    try:
        print("[KVPress] Verifying TOVA multimodal compatibility...")
        base_model = None
        config = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            config = getattr(model.thinker, 'config', None)
        elif hasattr(model, 'model'):
            base_model = model.model
            config = getattr(model, 'config', None)
        if base_model is None:
            print("[KVPress] TOVA multimodal verification failed: Unable to locate base model")
            return False
        rotary_emb = getattr(base_model, 'rotary_emb', None)
        if rotary_emb is None:
            print("[KVPress] TOVA multimodal verification failed: missing rotary_emb")
            return False
        if hasattr(rotary_emb, 'head_dim'):
            head_dim = rotary_emb.head_dim
            if head_dim <= 0 or head_dim % 2 != 0:
                print(f"[KVPress] TOVA multimodal verification failed: head_dim {head_dim} not suitable for TOVA")
                return False
        print("[KVPress] TOVA multimodal compatibility verification passed")
        return True
    except Exception as e:
        print(f"[KVPress] TOVA multimodal compatibility verification error: {e}")
        return False

def verify_snapkv_multimodal_compatibility(model):
    """Verify SnapKV compatibility with multimodal input (ported from HAD_qwen_kvpress)"""
    try:
        print("[KVPress] Verifying SnapKV multimodal compatibility...")
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        if base_model is None:
            print("[KVPress] SnapKV multimodal verification failed: Unable to locate base model")
            return False
        rotary_emb = getattr(base_model, 'rotary_emb', None)
        if rotary_emb is None or not hasattr(rotary_emb, 'forward'):
            print("[KVPress] SnapKV multimodal verification failed: rotary_emb missing or no forward")
            return False
        print("[KVPress] SnapKV multimodal compatibility verification passed")
        return True
    except Exception as e:
        print(f"[KVPress] SnapKV multimodal compatibility verification error: {e}")
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
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    

    print("=== Environment Diagnostics ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")
    

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not installed")
    

    if not QWEN_AVAILABLE:
        print("\n[ERROR] Qwen2.5-Omni module not available")
        print("Please check the following:")
        print("1. Is model path correct: /data/to/your/code/path/Qwen_2.5")
        print("2. Are the following files present:")
        print("   - modeling_qwen2_5_omni_origin.py")
        print("   - processing_qwen2_5_omni.py") 
        print("   - qwen_omni_utils.py")
        print("3. Are you running in the correct environment")
        return
    

    if not KV_PRESS_AVAILABLE:
        print("\n[ERROR] KV Press library not available but compression required")
        print("Please check the following:")
        print("1. Is kvpress library installed: pip install kvpress")
        print("2. Is transformers version compatible")
        print("3. Is local kvpress directory present")
        print("4. Is Python environment correct")
        return
    

    if args.no_compress:
        print("\n[WARNING] Ignoring --no-compress, forcibly enabling KV Press compression")
        args.no_compress = False
    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda"
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\nNo GPU available, using CPU")
    

    result_dir = os.path.abspath(RESULTS_DIR_ENV)
    os.makedirs(result_dir, exist_ok=True)
    

    samples, scene_counts = load_tau_acoustic_scene_dataset(args.dataset_path)
    if not samples:
        print("[ERROR] Failed to load dataset")
        return
    
    if SAMPLE_LIMIT > 0 and len(samples) > SAMPLE_LIMIT:

        import random
        samples = random.sample(samples, SAMPLE_LIMIT)
        print(f"[LIMIT] Randomly selected {len(samples)} samples")
    

    print("[LOAD] Qwen2.5-Omni model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    

    will_use_compression = not args.no_compress and KV_PRESS_AVAILABLE
    

    attention_impl = "eager" if will_use_compression else "flash_attention_2"
    print(f"[Attention] Using {attention_impl} implementation (compression: {'enabled' if will_use_compression else 'disabled'})")
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_impl,
        trust_remote_code=True,
    )
    
    if hasattr(model, 'disable_talker'):
        model.disable_talker()
    model.eval()
    

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


    initial_allocated, initial_reserved = get_gpu_memory_usage()
    print(f"After model load GPU memory - allocated: {initial_allocated:.2f}GB, reserved: {initial_reserved:.2f}GB")
    

    print("[CONFIG] Adding necessary config attributes for Qwen2.5-Omni...")
    

    if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
        if not hasattr(model.thinker.config, 'audio_token_id'):
            model.thinker.config.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(model.thinker.config, 'image_token_id'):
            model.thinker.config.image_token_id = 151655
        if not hasattr(model.thinker.config, 'video_token_id'):
            model.thinker.config.video_token_id = 151656
        if not hasattr(model.thinker.config, 'audio_bos_token_id'):
            model.thinker.config.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(model.thinker.config, 'audio