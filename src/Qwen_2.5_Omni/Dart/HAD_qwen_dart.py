#!/usr/bin/env python3

import os
import sys
import torch
import json
import argparse
import logging
import time
import warnings
import random
import tempfile
import traceback
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import gc
import re
from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

random.seed(42)

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni_dart import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      
_AUDIO_SPECIAL_TOKEN_ID = 200011

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attn_implementation')
    parser.add_argument('--sparse', type=str_to_bool, default=True, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--image_token_start_index', type=int, default=None, help='image_token_start_index')
    parser.add_argument('--image_token_length', type=int, default=None, help='image_token_length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    return parser.parse_args()

def configure_DART(model, args):
    """Configure DART sparse attention mechanism - adapted for Qwen2.5-Omni"""
    if not hasattr(model.config, 'DART_config'):
        model.config.DART_config = {}
    
    if args.sparse:
        DART_config = {
            "K": args.pruned_layer,
            "sparse": True,
            "enable_dart": True,
            
            "image_token_start_index": args.image_token_start_index, 
            "image_token_length": args.image_token_length,
            
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            
            "reduction_ratio": args.reduction_ratio,
            
            "pivot_image_token": getattr(args, 'pivot_image_token', args.pivot_audio_token),
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            
            "text_length": 1,
            
            "qwen_dart_enabled": True,
            "multimodal_pruning": True,
        }
        
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            model.thinker.model.config.DART_config = DART_config
            print("DART configuration set to thinker.model.config")
        elif hasattr(model, 'config'):
            model.config.DART_config = DART_config
            print("DART configuration set to model.config")
        else:
            print("Warning: Cannot set DART configuration")
    else:
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            model.thinker.model.config.DART_config = None
        elif hasattr(model, 'config'):
            model.config.DART_config = None
    
    print(f"Qwen2.5-Omni DART configuration: sparse={args.sparse}, "
          f"reduction_ratio={args.reduction_ratio}, "
          f"pruned_layer={args.pruned_layer}")

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")
print(f"CUDA_VISIBLE_DEVICES: {gpu_temp}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

data_path_root = '/data/to/your/dataset/path/HAD/concatenated_audio'
result_dir = os.environ.get("RESULTS_DIR", './HAD_Results')
os.makedirs(result_dir, exist_ok=True)

class HADTimingStats:
    """Track HAD task inference time statistics using CUDA Event for precise measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, label=None):
        """Add a timing record"""
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "label": label
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get overall statistics summary"""
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        df = pd.DataFrame(self.timing_records)
        summary = {
            "total_samples": len(df),
            "avg_total_time": df["total_time"].mean(),
            "avg_prefill_time": df["prefill_time"].mean(),
            "avg_decode_time": df["decode_time"].mean(),
            "avg_decode_tokens_per_sec": df["decode_tokens_per_sec"].mean(),
            "prefill_percentage": (df["prefill_time"].sum() / df["total_time"].sum()) * 100,
            "decode_percentage": (df["decode_time"].sum() / df["total_time"].sum()) * 100
        }
        
        return summary
    
    def export_to_json(self, output_file):
        """Export timing statistics to JSON file"""
        summary = self.get_summary()
        data = {
            "summary": summary,
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def extract_had_answer(response, choice_a="real", choice_b="fake"):
    """Extract HAD answer from model response"""
    if not response:
        return ""
    
    response = response.strip().upper()
    
    if response in ['A', 'B']:
        return choice_a if response == 'A' else choice_b
    
    if response.startswith('A') and len(response) <= 3:
        return choice_a
    if response.startswith('B') and len(response) <= 3:
        return choice_b
    
    match = re.search(r'\b([AB])\b', response)
    if match:
        return choice_a if match.group(1) == 'A' else choice_b
    
    match = re.search(r'[(\[]?([AB])[)\].]?', response)
    if match:
        return choice_a if match.group(1) == 'A' else choice_b
    
    choice_a_lower = choice_a.lower()
    choice_b_lower = choice_b.lower()
    response_lower = response.lower()
    
    if choice_a_lower in response_lower and choice_b_lower not in response_lower:
        return choice_a
    if choice_b_lower in response_lower and choice_a_lower not in response_lower:
        return choice_b
    
    return ""

def load_had_dataset(root_dir):
    """Load HAD dataset, balance real and fake samples"""
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
                    "id": f"real_{audio_file}",
                    "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'.",
                    "choice_a": "real",
                    "choice_b": "fake",
                    "answer_gt": "real",
                    "task": "Audio_Authenticity_Detection"
                })
    
    if os.path.exists(fake_dir):
        for audio_file in os.listdir(fake_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(fake_dir, audio_file)
                all_samples.append({
                    "audio_path": audio_path,
                    "label": "fake",
                    "id": f"fake_{audio_file}",
                    "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'.",
                    "choice_a": "real",
                    "choice_b": "fake",
                    "answer_gt": "fake",
                    "task": "Audio_Authenticity_Detection"
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
    
    if choice_a_lower == "real" and choice_b_lower == "fake":
        real_keywords = ["real", "authentic", "genuine", "original", "natural"]
        fake_keywords = ["fake", "synthetic", "artificial", "generated", "deepfake"]
        
        real_count = sum(1 for keyword in real_keywords if keyword in text_lower)
        fake_count = sum(1 for keyword in fake_keywords if keyword in text_lower)
        
        if real_count > fake_count:
            return "real"
        elif fake_count > real_count:
            return "fake"
    
    return ""

def calculate_had_metrics(predictions, ground_truths):
    """Calculate F1 score and other metrics for HAD task"""
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['real', 'fake'] and t in ['real', 'fake']]
    
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
    
    label_map = {'real': 0, 'fake': 1}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        'total_samples': len(predictions)
    }

def create_had_prompt(sample):
    """Create prompt for HAD task"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?"
    format_text = " If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'."
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def load_had_metadata(data_path):
    """Load HAD metadata"""
    
    metadata = {}
    
    metadata_files = [
        data_path / "labels.csv",
        data_path / "metadata.csv",
        data_path / "annotations.txt"
    ]
    
    for meta_file in metadata_files:
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    if lines and ('filename' in lines[0].lower() or 'activity' in lines[0].lower()):
                        lines = lines[1:]
                    
                    for line in lines:
                        if ',' in line:
                            parts = line.strip().split(',')
                        else:
                            parts = line.strip().split('\t')
                        
                        if len(parts) >= 2:
                            filename = parts[0].strip()
                            activity = parts[1].strip()
                            
                            metadata[filename] = {
                                'activity': activity
                            }
                break
            except Exception as e:
                print(f"Error reading {meta_file}: {e}")
                continue
    
    print(f"Loaded metadata for {len(metadata)} HAD samples")
    return metadata

def parse_activity_from_filename(filename: str) -> str:
    """Parse activity from HAD filename"""
    
    filename_lower = filename.lower()
    
    activities = {
        'walking': ['walk', 'walking', 'step'],
        'running': ['run', 'running', 'jog'],
        'sitting': ['sit', 'sitting', 'seated'],
        'standing': ['stand', 'standing'],
        'lying': ['lie', 'lying', 'laying'],
        'eating': ['eat', 'eating', 'meal'],
        'drinking': ['drink', 'drinking'],
        'talking': ['talk', 'talking', 'speak'],
        'typing': ['type', 'typing', 'keyboard'],
        'writing': ['write', 'writing'],
        'reading': ['read', 'reading'],
        'sleeping': ['sleep', 'sleeping']
    }
    
    for activity, keywords in activities.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return activity
    
    return 'unknown'

def create_dummy_had_samples(count: int) -> List[Dict]:
    """Create dummy HAD samples for testing"""
    
    samples = []
    activities = [
        'walking', 'running', 'sitting', 'standing', 'lying',
        'eating', 'drinking', 'talking', 'typing', 'writing'
    ]
    
    for i in range(count):
        activity = activities[i % len(activities)]
        filename = f"{activity}_{i:03d}.wav"
        
        samples.append({
            'audio_path': f"/dummy/had/{filename}",
            'filename': filename,
            'file_id': f"{activity}_{i:03d}",
            'activity': activity,
            'task': 'human_activity_detection'
        })
    
    print(f"Created {len(samples)} dummy HAD samples")
    return samples

def process_had_sample(sample: Dict, processor, model, timing_stats: HADTimingStats, device) -> Dict:
    """Process single HAD sample for activity detection"""
    
    try:
        audio_path = sample['audio_path']
        if not os.path.exists(audio_path):
            audio = np.random.randn(16000 * 6)
            sr = 16000
        else:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert in human activity recognition from audio signals. Listen to the audio carefully and identify what human activity is being performed. Common activities include: walking, running, sitting, standing, lying, eating, drinking, talking, typing, writing, reading, sleeping."
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "What human activity is being performed in this audio? Provide only the activity name."}
                ]
            }
        ]
        
        start_event, end_event = timing_stats.start_timing()
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio], return_tensors="pt", sampling_rate=16000).to(device)
        
        prefill_duration = timing_stats.end_timing(start_event, end_event)
        timing_stats.record_prefill(prefill_duration)
        
        start_event, end_event = timing_stats.start_timing()
        
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        decode_duration = timing_stats.end_timing(start_event, end_event)
        timing_stats.record_decode(decode_duration)
        
        total_duration = prefill_duration + decode_duration
        timing_stats.record_total(total_duration)
        
        response = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].size(1):], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        predicted_activity = parse_activity_response(response)
        
        result = {
            'file_id': sample['file_id'],
            'filename': sample['filename'],
            'audio_path': sample['audio_path'],
            'true_activity': sample['activity'],
            'predicted_activity': predicted_activity,
            'raw_response': response,
            'correct': predicted_activity.lower() == sample['activity'].lower(),
            'processing_time': total_duration
        }
        
        del inputs, generate_ids
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        print(f"Error processing sample {sample.get('file_id', 'unknown')}: {e}")
        return {
            'file_id': sample.get('file_id', 'unknown'),
            'filename': sample.get('filename', ''),
            'audio_path': sample.get('audio_path', ''),
            'true_activity': sample.get('activity', ''),
            'predicted_activity': 'error',
            'raw_response': f"Error: {e}",
            'correct': False,
            'processing_time': 0.0
        }

def parse_activity_response(response: str) -> str:
    """Parse activity detection response"""
    
    response = response.lower().strip()
    
    activities = [
        'walking', 'running', 'sitting', 'standing', 'lying',
        'eating', 'drinking', 'talking', 'typing', 'writing',
        'reading', 'sleeping'
    ]
    
    activity_aliases = {
        'walk': 'walking',
        'run': 'running',
        'jog': 'running',
        'sit': 'sitting',
        'stand': 'standing',
        'lie': 'lying',
        'lay': 'lying',
        'eat': 'eating',
        'drink': 'drinking',
        'talk': 'talking',
        'speak': 'talking',
        'type': 'typing',
        'write': 'writing',
        'read': 'reading',
        'sleep': 'sleeping'
    }
    
    for activity in activities:
        if activity in response:
            return activity
    
    for alias, activity in activity_aliases.items():
        if alias in response:
            return activity
    
    words = response.split()
    if words:
        first_word = words[0].lower()
        if first_word in activities:
            return first_word
        elif first_word in activity_aliases:
            return activity_aliases[first_word]
    
    return 'unknown'

def evaluate_had_results(results: List[Dict]) -> Dict:
    """Evaluate HAD human activity detection results"""
    
    valid_results = [r for r in results if r['predicted_activity'] != 'error']
    
    if not valid_results:
        return {
            'accuracy': 0.0,
            'total_samples': len(results),
            'valid_samples': 0,
            'error_rate': 1.0
        }
    
    correct_predictions = sum(1 for r in valid_results if r['correct'])
    accuracy = correct_predictions / len(valid_results)
    
    activity_stats = {}
    all_activities = set(r['true_activity'] for r in valid_results)
    
    for activity in all_activities:
        activity_results = [r for r in valid_results if r['true_activity'] == activity]
        activity_correct = sum(1 for r in activity_results if r['correct'])
        activity_stats[activity] = {
            'total': len(activity_results),
            'correct': activity_correct,
            'accuracy': activity_correct / len(activity_results) if activity_results else 0.0
        }
    
    processing_times = [r['processing_time'] for r in valid_results if r['processing_time'] > 0]
    
    evaluation = {
        'accuracy': accuracy,
        'total_samples': len(results),
        'valid_samples': len(valid_results),
        'correct_predictions': correct_predictions,
        'error_rate': (len(results) - len(valid_results)) / len(results) if results else 0,
        'activity_statistics': activity_stats,
        'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
        'unique_activities': len(all_activities)
    }
    
    return evaluation

def main():
    random.seed(42)
    
    args = parse_arguments()
    
    print(f"\n=== HAD DART Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    if args.sparse:
        method_name = "sparse"
        ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    else:
        method_name = "base"
        ratio_str = "ratio_1.000"
    
    output_file = f'{result_dir}/HAD_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/HAD_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    
    timing_stats = HADTimingStats()
    
    print("Loading Qwen2.5-Omni model...")
    model_path = "/data/to/your/Qwen_2.5Omni-3B/Model/folder"
    device_map = {"": 0}
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.disable_talker()
    
    configure_DART(model, args)
    print("Model loaded successfully")
    
    samples = load_had_dataset(data_path_root)
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")
    
    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)
    
    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"Classification statistics: real samples={real_count}, fake samples={fake_count}")
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
            "metrics": {},
            "timing": {}
        }
    }
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    all_predictions = []
    all_ground_truths = []
    
    with tqdm(total=len(samples), desc="Processing HAD audio authenticity detection samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            
            audio_path_for_inference = sample["audio_path"]
            if not os.path.exists(audio_path_for_inference):
                print(f"Warning: Audio file does not exist: {audio_path_for_inference}")
                pbar.update(1)
                continue

            question = sample.get("question", "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?")
            choice_a = sample.get("choice_a", "real")
            choice_b = sample.get("choice_b", "fake")
            
            instruction = f"{question}\n"
            instruction += f"A: {choice_a}\nB: {choice_b}\n"
            instruction += "Respond with only the letter of your answer (A or B)."

            qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            task_prompt = "You are a helpful assistant that analyzes audio to detect authenticity. Please listen to the audio carefully and determine if it is real or contains synthetic/fake content."
            sys_prompt = f"{qwen_intro} {task_prompt}"

            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path_for_inference},
                    {"type": "text", "text": instruction},
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
                if _AUDIO_BOS_TOKEN_ID in token_ids and _AUDIO_EOS_TOKEN_ID in token_ids:
                    audio_start = token_ids.index(_AUDIO_BOS_TOKEN_ID)
                    audio_end = token_ids.index(_AUDIO_EOS_TOKEN_ID)
                    audio_token_start = audio_start
                    audio_token_length = audio_end - audio_start + 1
                    audio_detected = True

            if not audio_detected:
                audio_token_start = args.audio_token_start_index
                audio_token_length = args.audio_token_length

            if args.sparse:
                args.audio_token_start_index = audio_token_start
                args.audio_token_length = audio_token_length
                configure_DART(model, args)

            full_start_event = torch.cuda.Event(enable_timing=True)
            full_end_event = torch.cuda.Event(enable_timing=True)
            
            first_token_start_event = torch.cuda.Event(enable_timing=True)
            first_token_end_event = torch.cuda.Event(enable_timing=True)

            full_start_event.record()
            
            first_token_start_event.record()
            with torch.no_grad():
                first_token_output = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
            first_token_end_event.record()
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
            full_end_event.record()

            torch.cuda.synchronize()
            first_token_time = first_token_start_event.elapsed_time(first_token_end_event) / 1000.0
            total_time = full_start_event.elapsed_time(full_end_event) / 1000.0
            
            prefill_time = first_token_time
            decode_time = max(0.0, total_time - prefill_time)

            output_text = processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            if "assistant\n" in output_text:
                output_text = output_text.split("assistant\n")[-1].strip()

            if hasattr(output, 'shape') and len(output.shape) > 1:
                output_tokens = output.shape[1] - inputs.input_ids.shape[1]
            else:
                output_tokens = len(output_text.split())

            output_text = output_text.strip()

            predicted_answer = extract_had_answer(output_text, choice_a, choice_b)
            ground_truth = sample["answer_gt"]

            is_correct = (predicted_answer == ground_truth)

            all_predictions.append(predicted_answer if predicted_answer else "ERROR")
            all_ground_truths.append(ground_truth)

            current_label = sample["label"]
            if current_label == "real":
                results["summary"]["real_total"] += 1
                if is_correct:
                    results["summary"]["real_correct"] += 1
            elif current_label == "fake":
                results["summary"]["fake_total"] += 1
                if is_correct:
                    results["summary"]["fake_correct"] += 1

            if is_correct:
                results["summary"]["correct_samples"] += 1

            results["summary"]["total_samples"] += 1

            if idx > 0 and idx <= 100:
                timing_stats.add_record(prefill_time, decode_time, output_tokens, input_token_length, 
                                      sample.get("duration", 0), current_label)

            sample_result = {
                "id": sample.get("id", f"had_{idx}"),
                "audio_path": audio_path_for_inference,
                "label": current_label,
                "question": question,
                "choices": {"A": choice_a, "B": choice_b},
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "raw_response": output_text,
                "is_correct": is_correct,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "total_time": total_time
            }

            results["samples"].append(sample_result)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            pbar.update(1)

    if len(all_predictions) > 0:
        metrics = calculate_had_metrics(all_predictions, all_ground_truths)
        results["summary"]["metrics"] = metrics
        
        print(f"\n=== HAD Evaluation Results ===")
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")
    
    timing_summary = timing_stats.get_summary()
    results["summary"]["timing"] = timing_summary
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()