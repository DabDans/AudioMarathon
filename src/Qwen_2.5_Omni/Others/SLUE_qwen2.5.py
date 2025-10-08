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


random.seed(42)


sys.path.append("/data/to/your/Modeling/path/")
from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)


from qwen_omni_utils import process_mm_info

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types to ensure JSON compatibility"""
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


_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      




os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"


logging.set_verbosity_error()
warnings.filterwarnings("ignore")


gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")


prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 1))
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

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

class SLUETimingStats:
    """Track inference timing statistics for SLUE tasks"""
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
        self.max_timing_samples = 100  
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
        """Add a timing record, limited to the first 100 samples (excluding the first one)"""
        
        if self.total_samples < self.max_timing_samples:
            self.total_samples += 1
            self.total_prefill_time += prefill_time
            self.total_decode_time += decode_time
            self.total_tokens += output_tokens
            if audio_duration:
                self.total_audio_duration += audio_duration
            
            record = {
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "audio_duration": audio_duration,
                "task_type": task_type,
                "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
            }
            self.timing_records.append(record)
            
            
            if task_type:
                self.task_type_stats[task_type].append(record)
    
    def get_summary(self):
        """Get overall statistics summary"""
        if self.total_samples == 0:
            return {
                "overall_summary": {},
                "task_summaries": {}
            }
        
        avg_prefill = self.total_prefill_time / self.total_samples
        avg_decode = self.total_decode_time / self.total_samples
        avg_total = avg_prefill + avg_decode
        avg_tokens_per_sec = self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        
        summary = {
            "total_samples": self.total_samples,
            "avg_prefill_time": avg_prefill,
            "avg_decode_time": avg_decode,
            "avg_total_time": avg_total,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_audio_duration": self.total_audio_duration,
            "avg_audio_duration": self.total_audio_duration / self.total_samples if self.total_samples > 0 else 0
        }
        
        
        task_summaries = {}
        for task_type, records in self.task_type_stats.items():
            if len(records) > 0:
                task_prefill = sum(r["prefill_time"] for r in records)
                task_decode = sum(r["decode_time"] for r in records)
                task_tokens = sum(r["output_tokens"] for r in records)
                task_duration = sum(r["audio_duration"] for r in records if r["audio_duration"])
                
                task_summaries[task_type] = {
                    "samples": len(records),
                    "avg_prefill_time": task_prefill / len(records),
                    "avg_decode_time": task_decode / len(records),
                    "avg_total_time": (task_prefill + task_decode) / len(records),
                    "avg_tokens": task_tokens / len(records),
                    "avg_tokens_per_sec": task_tokens / task_decode if task_decode > 0 else 0,
                    "avg_audio_duration": task_duration / len(records) if task_duration > 0 else 0
                }
        
        return {
            "overall_summary": summary,
            "task_summaries": task_summaries
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files according to Qwen2.5-Omni requirements"""
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio.shape}, sample rate={sr}Hz")
        except Exception as e:
            print(f"librosa loading failed: {e}")
            
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"soundfile processing successful: shape={audio.shape}, sample rate={sr}Hz")
                
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generating silent replacement audio")
        
        
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

def load_slue_dataset(json_file, audio_base_dir):
    """
    Load SLUE task data from JSON file
    
    Args:
        json_file: SLUE format JSON task file path
        audio_base_dir: Audio file base directory
    
    Returns:
        dataset: List containing task data
    """
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []
    
    print(f"Loading SLUE JSON file: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    
    if not isinstance(data, list):
        print(f"Error: JSON file format is incorrect, expected list format")
        return []
    
    print(f"Loaded {len(data)} tasks from JSON")
    
    
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(data):
        
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task {i} missing audio path")
            continue
        
        
        if not os.path.exists(full_audio_path):
            if missing_files < 5:  
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            missing_files += 1
            continue
        
        
        task_name = task.get("task_name", "unknown")
        dataset_name = task.get("dataset_name", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        
        
        choice_a = task.get("choice_a", "")
        choice_b = task.get("choice_b", "")
        choice_c = task.get("choice_c", "")
        choice_d = task.get("choice_d", "")
        
        
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"Warning: Unable to get audio information {full_audio_path}: {e}")
            duration = 0
            sample_rate = 16000
        
        
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_name": task_name,
            "dataset_name": dataset_name,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "entity_count": task.get("entity_count", 0),
            "entity_types": task.get("entity_types", []),
            "source_count": task.get("source_count", 0),
            "audio_duration_info": task.get("audio_duration_info", ""),
            "source_folder": task.get("source_folder", ""),
            "source_file": task.get("source_file", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")
    return dataset

def create_slue_prompt(doc):
    """Generate prompts for SLUE format tasks (consistent with original version)"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    
    
    prompt_text = f"""{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."""
    
    
    return prompt_text

def extract_answer_choice(response):
    """Extract answer choices (A, B, C, D) from model responses, handling various output formats"""
    if not response:
        return ""
    
    
    response = response.strip().upper()
    
    
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    
    if response.startswith('A') and len(response) <= 3:
        return 'A'
    if response.startswith('B') and len(response) <= 3:
        return 'B'
    if response.startswith('C') and len(response) <= 3:
        return 'C'
    if response.startswith('D') and len(response) <= 3:
        return 'D'
    
    
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    
    
    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)
    
    
    match = re.search(r'(?:option|choice)\s+([ABCD])', response)
    if match:
        return match.group(1)
    
    
    return ""

def evaluate_slue_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate SLUE task accuracy"""
    try:
        
        pred = predicted_choice.strip().upper() if predicted_choice else ""
        gt = ground_truth_choice.strip().upper() if ground_truth_choice else ""
        
        
        accuracy = 1.0 if pred == gt else 0.0
        
        return {
            "accuracy": accuracy,
            "predicted_choice": pred,
            "ground_truth_choice": gt,
            "is_correct": pred == gt
        }
    except Exception as e:
        print(f"Error evaluating SLUE accuracy: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def calculate_slue_metrics(predictions, ground_truths):
    """Calculate F1 score and other metrics for SLUE tasks"""
    try:
        
        valid_pairs = []
        for pred, gt in zip(predictions, ground_truths):
            if pred and gt and pred in ['A', 'B', 'C', 'D'] and gt in ['A', 'B', 'C', 'D']:
                valid_pairs.append((pred, gt))
        
        if len(valid_pairs) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'macro_f1': 0.0,
                'valid_samples': 0
            }
        
        
        valid_preds, valid_gts = zip(*valid_pairs)
        
        
        accuracy = accuracy_score(valid_gts, valid_preds)
        
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_gts, valid_preds, average='macro', zero_division=0
        )
        
        
        _, _, macro_f1, _ = precision_recall_fscore_support(
            valid_gts, valid_preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'macro_f1': float(macro_f1),
            'valid_samples': len(valid_pairs)
        }
    except Exception as e:
        print(f"Error calculating SLUE metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'macro_f1': 0.0,
            'valid_samples': 0
        }

def main():
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 1))
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

    
    slue_json_file = "/path/to/your/subsetSLUE/merged_audio_data.json"
    audio_base_dir = "/path/to/your/subsetSLUE"
    
    print(f"SLUE JSON file: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    
    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    
    result_dir = os.environ.get("RESULTS_DIR", './SLUE_Results')
    os.makedirs(result_dir, exist_ok=True)

    
    output_file = f'{result_dir}/slue_results_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    timing_output_file = f'{result_dir}/slue_timing_stats_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    
    timing_stats = SLUETimingStats()

    print(f"\n=== SLUE NER Task Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Original method parameter: {os.environ.get('PRUNE_METHOD', 'N/A')}")
    print(f"use_random: {use_random}, use_frame: {use_frame}")
    print(f"SLUE JSON file: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    
    print("Loading Qwen2.5-Omni model...")
    model_path = "/path/to/your/model"
    device_map = {"": 0}  
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    
    
    model.disable_talker()
    
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    

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
    print(f"Initializing thinker.model.config pruning configuration parameters")

    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Using dataset: {len(samples)} samples")
    
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, processing {len(samples)} samples")

    
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    for sample in samples:
        task_name = sample.get("task_name", "unknown")
        dataset_name = sample.get("dataset_name", "unknown")
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1
    
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0
    
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)
    dataset_correct = defaultdict(int)
    dataset_total = defaultdict(int)

    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    
    tqdm_kwargs = {
        'ascii': True,        
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(samples)} samples...")
    
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="SLUE evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            
            audio_path = sample["path"]
            audio_np = prepare_audio_for_qwen_omni(audio_path, target_sr=16000)
            duration = len(audio_np) / 16000  
            
            
            prompt_text = create_slue_prompt(sample)
            
            
            task_name = sample.get("task_name", "unknown")
            dataset_name = sample.get("dataset_name", "unknown")
            ground_truth = sample.get("answer_gt", "")
            
            
            task_instruction = "You are a helpful assistant that analyzes speech audio for named entity recognition. Please listen carefully and extract the requested named entities from the speech."
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
                        {"type": "audio", "audio": audio_path},  
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
                    
                    
                    model.thinker.model.config.image_layer_idx = False  
                    model.thinker.model.config.audio_layer_idx = prune_layer_idx
                    model.thinker.model.config.audio_token_num = audio_token_length
                    model.thinker.model.config.audio_token_start = audio_token_start
                    model.thinker.model.config.audio_prune_ratio = prune_ratio
                    model.thinker.model.config.random = use_random
                    model.thinker.model.config.frame = use_frame 
                    
            if not audio_detected:
                model.thinker.model.config.audio_layer_idx = None
                model.thinker.model.config.audio_prune_ratio = 0
            
            
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            
            
            audio_tokens = 0
            if hasattr(processor.tokenizer, 'audio_bos_token_id') and hasattr(processor.tokenizer, 'audio_eos_token_id'):
                input_ids = inputs['input_ids'][0]
                audio_bos_positions = (input_ids == processor.tokenizer.audio_bos_token_id).nonzero(as_tuple=True)[0]
                audio_eos_positions = (input_ids == processor.tokenizer.audio_eos_token_id).nonzero(as_tuple=True)[0]
                
                if len(audio_bos_positions) > 0 and len(audio_eos_positions) > 0:
                    for bos_pos in audio_bos_positions:
                        eos_candidates = audio_eos_positions[audio_eos_positions > bos_pos]
                        if len(eos_candidates) > 0:
                            eos_pos = eos_candidates[0]
                            audio_tokens += eos_pos - bos_pos - 1
                            
                
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
                    if hasattr(model.thinker.model.config, 'sparse_attention_config'):
                        model.thinker.model.config.sparse_attention_config['audio_tokens'] = audio_tokens.item() if hasattr(audio_tokens, 'item') else audio_tokens
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prefill_end_event.record()

            
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)

            decode_start_event.record()
            out_ids = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=5,
                thinker_do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            decode_end_event.record()
            
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0  
            
            
            tokens = out_ids[:, inputs['input_ids'].shape[1]:]
            output_tokens = len(tokens[0])
            response = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            
            
            if not response.strip():
                response = processor.batch_decode(
                    out_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                output_tokens = len(out_ids[0]) - len(inputs["input_ids"][0])
            
            input_tokens = len(inputs["input_ids"][0])
            
            
            predicted_choice = extract_answer_choice(response)
            
            
            if idx < 5:
                print(f"\n=== SLUE sample {idx} debug information ===")
                print(f"Model raw output: '{response}'")
                print(f"Extracted answer: '{predicted_choice}'")
                print(f"Correct answer: '{ground_truth}'")
                print(f"Output token count: {output_tokens}")
                print("=" * 30)
            
            
            eval_result = evaluate_slue_accuracy(predicted_choice, ground_truth)
            accuracy = eval_result["accuracy"]
            
            
            total_accuracy += accuracy
            processed_samples += 1
            
            
            task_type_total[task_name] += 1
            if eval_result["is_correct"]:
                task_type_correct[task_name] += 1
            
            
            dataset_total[dataset_name] += 1
            if eval_result["is_correct"]:
                dataset_correct[dataset_name] += 1
            
            
            result_item = {
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "task_name": task_name,
                "dataset_name": dataset_name,
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "ground_truth_choice": ground_truth,
                "predicted_choice": predicted_choice,
                "response": response,
                "accuracy": accuracy,
                "is_correct": eval_result["is_correct"],
                "audio_path": audio_path,
                "audio_duration": duration,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "output_tokens": output_tokens,
                "input_tokens": input_tokens
            }
            results.append(result_item)
            
            
            if idx > 0 and idx <= 100:
                timing_stats.add_record(
                    prefill_time, decode_time,
                    output_tokens,
                    input_tokens,
                    duration, task_name
                )
            
            
            current_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0
            
            
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'acc': f'{current_accuracy:.3f}',
                    'pred': predicted_choice,
                    'gt': ground_truth,
                    'task': task_name[:8]
                })
                
                if is_screen_env:
                    
                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.3f}")
            else:
                progress_bar.set_postfix({
                    'acc': f'{current_accuracy:.3f}',
                    'pred': predicted_choice,
                    'gt': ground_truth,
                    'task': task_name[:8]
                })
            
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            
            
            result_item = {
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "task_name": sample.get("task_name", "unknown"),
                "dataset_name": sample.get("dataset_name", "unknown"),
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "ground_truth_choice": sample.get("answer_gt", ""),
                "predicted_choice": "",
                "response": "ERROR",
                "accuracy": 0.0,
                "is_correct": False,
                "audio_path": sample.get("path", ""),
                "audio_duration": 0.0,
                "prefill_time": 0.0,
                "decode_time": 0.0,
                "total_time": 0.0,
                "output_tokens": 0,
                "input_tokens": 0
            }
            results.append(result_item)
            processed_samples += 1
            
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            continue

    
    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0

    
    all_predictions = [sample["predicted_choice"] for sample in results]
    all_ground_truths = [sample["ground_truth_choice"] for sample in results]
    
    overall_metrics = calculate_slue_metrics(all_predictions, all_ground_truths)

    
    task_type_accuracies = {}
    task_type_metrics = {}
    for task_name in task_type_stats.keys():
        if task_type_total[task_name] > 0:
            task_accuracy = task_type_correct[task_name] / task_type_total[task_name]
            task_type_accuracies[task_name] = task_accuracy
            
            
            task_predictions = [r["predicted_choice"] for r in results if r["task_name"] == task_name]
            task_ground_truths = [r["ground_truth_choice"] for r in results if r["task_name"] == task_name]
            task_metrics = calculate_slue_metrics(task_predictions, task_ground_truths)
            task_type_metrics[task_name] = task_metrics

    
    dataset_accuracies = {}
    dataset_metrics = {}
    for dataset_name in dataset_stats.keys():
        if dataset_total[dataset_name] > 0:
            dataset_accuracy = dataset_correct[dataset_name] / dataset_total[dataset_name]
            dataset_accuracies[dataset_name] = dataset_accuracy
            
            
            dataset_predictions = [r["predicted_choice"] for r in results if r["dataset_name"] == dataset_name]
            dataset_ground_truths = [r["ground_truth_choice"] for r in results if r["dataset_name"] == dataset_name]
            dataset_metrics_result = calculate_slue_metrics(dataset_predictions, dataset_ground_truths)
            dataset_metrics[dataset_name] = dataset_metrics_result

    
    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "f1_score": overall_metrics["f1_score"],
        "precision": overall_metrics["precision"], 
        "recall": overall_metrics["recall"],
        "macro_f1": overall_metrics["macro_f1"],
        "valid_samples": overall_metrics["valid_samples"],
        "task_type_stats": dict(task_type_stats),
        "dataset_stats": dict(dataset_stats),
        "task_type_accuracies": task_type_accuracies,
        "task_type_metrics": task_type_metrics,
        "dataset_accuracies": dataset_accuracies,
        "dataset_metrics": dataset_metrics,
        "task_type_correct": dict(task_type_correct),
        "task_type_total": dict(task_type_total),
        "dataset_correct": dict(dataset_correct),
        "dataset_total": dict(dataset_total),
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "slue_json_file": slue_json_file,
            "audio_base_dir": audio_base_dir,
            "timing_sample_count": min(100, max(0, len(results) - 1))
        },
        "timing": timing_stats.get_summary()
    }

    
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== SLUE NER Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Valid samples: {overall_metrics['valid_samples']}")
    print(f"Overall accuracy: {final_accuracy:.3f}")
    print(f"F1 score: {overall_metrics['f1_score']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"Macro F1: {overall_metrics['macro_f1']:.4f}")
    print(f"Task type count: {len(task_type_stats)}")
    print(f"Dataset count: {len(dataset_stats)}")
    
    print(f"\nPer-task detailed metrics:")
    for task_name, acc in task_type_accuracies.items():
        correct_num = task_type_correct[task_name]
        total_num = task_type_total[task_name]
        task_f1 = task_type_metrics[task_name]["f1_score"]
        print(f"  {task_name}:")
    print(f"    Accuracy: {acc:.3f} ({correct_num}/{total_num})")
    print(f"    F1 score: {task_f1:.4f}")
    
    print(f"\nPer-dataset detailed metrics:")
    for dataset_name, acc in dataset_accuracies.items():
        correct_num = dataset_correct[dataset_name]
        total_num = dataset_total[dataset_name]
        dataset_f1 = dataset_metrics[dataset_name]["f1_score"]
        print(f"  {dataset_name}:")
    print(f"    Accuracy: {acc:.3f} ({correct_num}/{total_num})")
    print(f"    F1 score: {dataset_f1:.4f}")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\nTime statistics (based on first {timing_sample_count} samples, excluding the first):")
    print(f"Statistical sample count: {overall_summary.get('total_samples', 0)}")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()
