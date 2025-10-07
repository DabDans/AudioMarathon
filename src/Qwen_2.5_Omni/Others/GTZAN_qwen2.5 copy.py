import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
from scipy.io import wavfile
from scipy import signal
import librosa
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import random
import sys

random.seed(42)

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

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")

prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

use_random = (prune_method == "random")
use_frame = (prune_method == "frame")
if use_random == False and use_frame == False:
    prune_method = "fast_v"

if prune_ratio == 0:
    method_is = "base"
else:
    method_is = prune_method

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

data_path_root = '/data/to/your/dataset/path//GTZAN/concatenated_audio'
result_dir = './GTZAN_Results'
os.makedirs(result_dir, exist_ok=True)

output_file = f'{result_dir}/GTZAN_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/GTZAN_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing stats will be saved to: {timing_output_file}")

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class GlobalTimingStats:
    """Global timing statistics class, collecting first 100 samples (excluding the first one)"""
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_time = 0.0
        self.total_input_tokens = 0
        self.total_audio_tokens = 0
        self.timing_records = []
        self.max_samples = 100
    
    def add_record(self, prefill_time, total_time, output_tokens, input_tokens, audio_tokens, sample_index):
        if sample_index == 0:
            return
        
        if self.samples >= self.max_samples:
            return
            
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_time += total_time
        self.total_input_tokens += input_tokens
        self.total_audio_tokens += audio_tokens
        
        self.timing_records.append({
            "sample_index": sample_index,
            "prefill_time": prefill_time,
            "total_time": total_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "audio_tokens": audio_tokens
        })
    
    def get_summary(self):
        """Get statistics summary"""
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_total_time": 0,
                "avg_input_tokens": 0,
                "avg_audio_tokens": 0
            }
        
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_total_time": self.total_time / self.samples,
            "avg_input_tokens": self.total_input_tokens / self.samples,
            "avg_audio_tokens": self.total_audio_tokens / self.samples
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def extract_gtzan_answer(text):
    """Extract GTZAN answer choice (A, B, C, D) from model output text"""
    if not text:
        return ""
    
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    
    text = text.strip().upper()
    
    if text in ['A', 'B', 'C', 'D']:
        return text
    
    if text == 'A' or text.startswith('A.') or text.startswith('A)') or text.endswith(' A'):
        return 'A'
    if text == 'B' or text.startswith('B.') or text.startswith('B)') or text.endswith(' B'):
        return 'B'
    if text == 'C' or text.startswith('C.') or text.startswith('C)') or text.endswith(' C'):
        return 'C'
    if text == 'D' or text.startswith('D.') or text.startswith('D)') or text.endswith(' D'):
        return 'D'
        
    if "option a" in text.lower() or "choice a" in text.lower() or "a)" in text.lower():
        return 'A'
    if "option b" in text.lower() or "choice b" in text.lower() or "b)" in text.lower():
        return 'B'
    if "option c" in text.lower() or "choice c" in text.lower() or "c)" in text.lower():
        return 'C'
    if "option d" in text.lower() or "choice d" in text.lower() or "d)" in text.lower():
        return 'D'
    
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    
    match = re.search(r'[(\[]?([ABCD])[)\].]?', text)
    if match:
        return match.group(1)
    
    return ""

def calculate_gtzan_metrics(y_true, y_pred):
    """
    Calculate detailed evaluation metrics for GTZAN music genre classification
    
    Args:
        y_true: True labels list (A/B/C/D)
        y_pred: Predicted labels list (A/B/C/D) 
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in ['A', 'B', 'C', 'D'] and pred_label in ['A', 'B', 'C', 'D']:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'per_class_metrics': {},
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true)
        }
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    labels = ['A', 'B', 'C', 'D']
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=labels, average=None, zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='weighted', zero_division=0
    )
    
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=labels,
        target_names=[f"Choice {label}" for label in labels],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true)
    }

def load_audio_for_gtzan(audio_path, audio_cache=None, target_sr=16000):
    """
    Load audio file, return format consistent with Qwen2.5-Omni
    """
    if audio_cache is not None and audio_path in audio_cache:
        audio_np, sr = audio_cache[audio_path]
    else:
        try:
            audio_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Loaded successfully with librosa: shape={audio_np.shape}, sample_rate={sr}Hz")
        except Exception as e:
            print(f"Librosa loading failed: {e}")
            
            try:
                audio_np, sample_rate = sf.read(audio_path)
                
                if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
                    audio_np = np.mean(audio_np, axis=1)
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio_np = signal.resample(audio_np, int(len(audio_np) * target_sr / sample_rate))
                    
                audio_np = audio_np.astype(np.float32)
                sr = target_sr
                print(f"Soundfile processing successful: shape={audio_np.shape}, sample_rate={sr}Hz")
                
            except Exception as e:
                print(f"Soundfile loading also failed: {e}")
                audio_np = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generated silence as replacement audio")
        
        audio_np = audio_np.astype(np.float32)
        
        if audio_cache is not None:
            audio_cache[audio_path] = (audio_np, sr)
    
    return audio_np, sr

def create_gtzan_prompt(question, options):
    """Create prompt for GTZAN task"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio segment and identify the music genre based on what you hear."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options += f"{letter}. {opt}\n"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def load_gtzan_metadata(metadata_path):
    """Load GTZAN metadata file"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    valid_samples = []
    for item in metadata:
        if all(key in item for key in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            valid_samples.append(item)
    
    print(f"Loaded {len(valid_samples)} valid samples from {len(metadata)} entries")
    return valid_samples

def calculate_gtzan_metrics(y_true, y_pred, genre_labels=None):
    """
    Calculate detailed evaluation metrics for GTZAN music genre classification
    
    Args:
        y_true: True labels list (A/B/C/D format)
        y_pred: Predicted labels list (A/B/C/D format)
        genre_labels: Genre labels list, if None then automatically get from data
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    valid_labels = ['A', 'B', 'C', 'D']
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in valid_labels and pred_label in valid_labels:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'per_class_metrics': {},
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true),
            'class_labels': valid_labels
        }
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=valid_labels, average=None, zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=valid_labels, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=valid_labels, average='weighted', zero_division=0
    )
    
    per_class_metrics = {}
    for i, label in enumerate(valid_labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=valid_labels,
        target_names=[f"Choice {label}" for label in valid_labels],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true),
        'class_labels': valid_labels
    }

def main():
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
    prune_method = os.environ.get("PRUNE_METHOD", "base")

    use_random = (prune_method == "random")
    use_frame = (prune_method == "frame")
    if use_random == False and use_frame == False:
        prune_method = "fast_v"
    
    if prune_ratio == 0:
        method_is = "base"
    else:
        method_is = prune_method

    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    data_path_root = '/data/to/your/dataset/path//GTZAN/concatenated_audio'
    metadata_file = os.path.join(data_path_root, 'music_genre_classification_meta.json')
    result_dir = os.environ.get("RESULTS_DIR", './GTZAN_Results')
    os.makedirs(result_dir, exist_ok=True)

    output_file = f'{result_dir}/gtzan_results_qwen25.json'
    timing_output_file = f'{result_dir}/timing_stats_qwen25_{method_is}_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

    timing_stats = GlobalTimingStats()

    print(f"\n=== GTZAN Evaluation Config (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {data_path_root}")
    print(f"Metadata file: {metadata_file}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    print("Loading Qwen2.5-Omni model...")
    model_path = "/data/to/your/Qwen_2.5_Model/path/"
    device_map = {"": 0}
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
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
        print(f"Initialized thinker.model.config pruning configuration parameters")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Loading GTZAN metadata: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file does not exist: {metadata_file}")
        return
    
    samples = load_gtzan_metadata(metadata_file)
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, processing {len(samples)} samples")

    audio_cache = {}
    print(f"Applied sample limit, processing {len(samples)} samples")

    print(f"Starting GTZAN music genre classification evaluation - Qwen2.5-Omni")
    print(f"Output file: {output_file}")
    print(f"GPU device: {gpu_id}")
    print("="*80)
    
    model_responses = []
    all_predictions = []
    all_ground_truths = []
    timing_stats = GlobalTimingStats()
    
    total_start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc="GTZAN Evaluation")):
        if sample is None:
            continue
            
        try:
            start_time = time.time()
            
            audio_rel = sample["path"]
            audio_full = os.path.join(data_path_root, 'wav', audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"Warning: Audio file does not exist: {audio_full}")
                continue
            
            options = [
                sample["choice_a"],
                sample["choice_b"], 
                sample["choice_c"],
                sample["choice_d"]
            ]
            question = sample["question"]
            correct_answer = sample["answer_gt"]
            
            instruction = "Listen to this audio segment and identify the music genre based on what you hear."
            format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
            
            formatted_options = ""
            for j, opt in enumerate(options):
                letter = chr(65 + j)
                formatted_options += f"{letter}. {opt}\n"
            
            prompt_text = f"{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}"
            
            task_instruction = "You are a helpful audio analysis assistant."
            full_user_prompt = f"{task_instruction}\n\n{prompt_text}"
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ]
                },
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_full},
                    {"type": "text", "text": full_user_prompt}
                ]}
            ]
            
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(text, list):
                text = text[0]

            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            )
            inputs = inputs.to(model.device).to(model.dtype)
            
            audio_token_length = 0
            audio_token_start = 0
            input_token_length = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
            
            audio_detected = False
            
            if hasattr(inputs, 'input_ids'):
                token_ids = inputs.input_ids[0].tolist()
                
                bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                
                if bos_positions and eos_positions:
                    audio_token_start = bos_positions[0]
                    audio_token_end = eos_positions[0]
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    audio_detected = True
                    
                    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                        model.thinker.model.config.image_layer_idx = False
                        model.thinker.model.config.audio_layer_idx = prune_layer_idx
                        model.thinker.model.config.audio_token_num = audio_token_length
                        model.thinker.model.config.audio_token_start = audio_token_start
                        model.thinker.model.config.audio_prune_ratio = prune_ratio
                        model.thinker.model.config.random = use_random
                        model.thinker.model.config.frame = use_frame

                elif not audio_detected and _AUDIO_TOKEN_ID in token_ids:
                    audio_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_TOKEN_ID]
                    if audio_positions:
                        audio_token_start = audio_positions[0]
                        audio_token_length = len(audio_positions)
                        
                        audio_detected = True
                        
                        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                            model.thinker.model.config.image_layer_idx = False
                            model.thinker.model.config.audio_layer_idx = prune_layer_idx
                            model.thinker.model.config.audio_token_num = audio_token_length
                            model.thinker.model.config.audio_token_start = audio_token_start
                            model.thinker.model.config.audio_prune_ratio = prune_ratio
                            model.thinker.model.config.random = use_random
                            model.thinker.model.config.frame = use_frame
            
            if not audio_detected:
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                    model.thinker.model.config.audio_layer_idx = None
                    model.thinker.model.config.audio_prune_ratio = 0
            
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            
            with torch.no_grad():
                prefill_output = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1,
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prefill_end_event.record()
            
            total_start_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)

            total_start_event.record()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=5,
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            total_end_event.record()
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            total_gpu_time = total_start_event.elapsed_time(total_end_event) / 1000.0
            
            response = processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            if "assistant\n" in response:
                assistant_start = response.rfind("assistant\n") + len("assistant\n")
                response = response[assistant_start:].strip()
            
            predicted_answer = extract_gtzan_answer(response)
            
            end_time = time.time()
            wall_time = end_time - start_time
            
            if hasattr(output, 'shape') and len(output.shape) > 1:
                output_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            else:
                output_tokens = 0
            
            timing_stats.add_record(prefill_time, total_gpu_time, output_tokens, input_token_length, audio_token_length, i)
            
            result = {
                "sample_id": i,
                "question": question,
                "ground_truth": correct_answer,
                "model_response": response,
                "predicted_answer": predicted_answer,
                "is_correct": predicted_answer == correct_answer,
                "wall_time": wall_time,
                "prefill_time": prefill_time,
                "total_gpu_time": total_gpu_time,
                "audio_path": audio_full,
                "options": options
            }
            model_responses.append(result)
            
            all_predictions.append(predicted_answer)
            all_ground_truths.append(correct_answer)
            
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t) / len(all_predictions)
            
            print(f"Sample {i+1:3d}: Predicted={predicted_answer:1s} | Ground Truth={correct_answer:1s} | "
                  f"Correct={predicted_answer == correct_answer} | "
                  f"Accuracy={current_accuracy:.3f} | "
                  f"Time={wall_time:.2f}s")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            result = {
                "sample_id": i,
                "question": question if 'question' in locals() else "N/A",
                "ground_truth": correct_answer if 'correct_answer' in locals() else "N/A",
                "model_response": f"ERROR: {str(e)}",
                "predicted_answer": "",
                "is_correct": False,
                "wall_time": 0.0,
                "prefill_time": 0.0,
                "total_gpu_time": 0.0
            }
            model_responses.append(result)
            all_predictions.append("")
            all_ground_truths.append(correct_answer if 'correct_answer' in locals() else "")
            continue
    
    total_end_time = time.time()
    
    metrics = calculate_gtzan_metrics(all_ground_truths, all_predictions)
    
    final_results = {
        "model_name": "Qwen2.5-Omni-3B",
        "dataset": "GTZAN",
        "total_samples": len(samples),
        "valid_samples": metrics['valid_samples'],
        "total_time": total_end_time - total_start_time,
        
        "accuracy": metrics['accuracy'],
        "f1_macro": metrics['f1_macro'],
        "f1_weighted": metrics['f1_weighted'],
        
        "precision_macro": metrics['precision_macro'],
        "recall_macro": metrics['recall_macro'],
        "precision_weighted": metrics['precision_weighted'],
        "recall_weighted": metrics['recall_weighted'],
        
        "per_class_metrics": metrics['per_class_metrics'],
        "classification_report": metrics['classification_report'],
        
        "timing_stats": timing_stats.get_summary(),
        
        "config": {
            "device": f"cuda:{gpu_id}",
            "audio_sample_rate": 16000,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "gpu_id": str(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')),
            "pruning_config": {
                "prune_layer_idx": prune_layer_idx,
                "prune_ratio": prune_ratio, 
                "prune_method": prune_method
            }
        },
        
        "detailed_results": model_responses
    }
    
    print(f"\nSaving results to: {output_file}")
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("GTZAN Music Genre Classification Evaluation Completed!")
    print("="*80)
    print(f"Model: Qwen2.5-Omni-3B")
    print(f"Total samples: {len(samples)}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score (macro average): {metrics['f1_macro']:.4f}")
    print(f"F1 score (weighted average): {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro average): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro average): {metrics['recall_macro']:.4f}")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average time per sample: {timing_stats.get_summary()['avg_total_time']:.2f} seconds")
    
    print("\nDetailed metrics for each choice:")
    print("-"*50)
    for choice, metrics_detail in metrics['per_class_metrics'].items():
        print(f"Choice {choice}: Precision={metrics_detail['precision']:.4f}, "
              f"Recall={metrics_detail['recall']:.4f}, "
              f"F1={metrics_detail['f1_score']:.4f}, "
              f"Support={metrics_detail['support']}")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()