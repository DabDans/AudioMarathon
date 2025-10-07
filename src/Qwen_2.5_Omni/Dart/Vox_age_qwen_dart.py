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
from sklearn.metrics import f1_score, precision_score, recall_score
import librosa

random.seed(42)

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni_dart import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      

from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class VoxAgeTimingStats:
    """VoxCeleb age classification task inference timing statistics"""
    def __init__(self):
        self.timing_records = []
        self.age_group_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   age_group=None):
        """Add a timing record"""
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "age_group": age_group
        }
        
        self.timing_records.append(record)
        
        if age_group:
            self.age_group_stats[age_group].append(record)
    
    def get_summary(self):
        """Get overall statistical summary"""
        if self.total_samples == 0:
            return {"error": "No samples processed"}
        
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
            "avg_tokens_per_sec": avg_tokens_per_sec
        }
        
        age_group_summaries = {}
        for age_group, records in self.age_group_stats.items():
            if len(records) > 0:
                age_group_summaries[age_group] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "age_group_summaries": age_group_summaries
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

def configure_DART(model, args):
    """Configure DART (Dynamic Attention Reduction for Audio Token) for Qwen model"""
    
    if not hasattr(model.config, 'sparse'):
        print("Adding DART configuration to model config...")
        model.config.sparse = args.sparse
        model.config.reduction_ratio = args.reduction_ratio
        model.config.pruned_layer = args.pruned_layer
        model.config.audio_token_start_index = args.audio_token_start_index
        model.config.audio_token_length = args.audio_token_length
        model.config.pivot_audio_token = args.pivot_audio_token
        model.config.pivot_text_token = args.pivot_text_token
    
    dart_config = {
        'sparse': args.sparse,
        'reduction_ratio': args.reduction_ratio,
        'pruned_layer': args.pruned_layer,
        'audio_token_start_index': args.audio_token_start_index,
        'audio_token_length': args.audio_token_length,
        'pivot_audio_token': args.pivot_audio_token,
        'pivot_text_token': args.pivot_text_token
    }
    
    print(f"DART Configuration: {dart_config}")
    
    return dart_config

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  
            'reserved': torch.cuda.memory_reserved() / 1024**3,    
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def str_to_bool(value):
    """String to boolean conversion"""
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VoxCeleb Age Classification with Qwen2.5-Omni + DART")
    
    parser.add_argument("--model_path", type=str, 
                       default="/data/to/your/Qwen_2.5Omni-3B/Model/folder", 
                       help="Qwen2.5-Omni model path")
    
    parser.add_argument('--sparse', type=str_to_bool, default=True, 
                       help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, 
                       help='prune_layer')
    parser.add_argument('--image_token_start_index', type=int, default=None, 
                       help='image_token_start_index')
    parser.add_argument('--image_token_length', type=int, default=None, 
                       help='image_token_length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, 
                       help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, 
                       help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, 
                       help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, 
                       help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, 
                       help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, 
                       help='pivot_text_token')
    
    parser.add_argument("--data_path", type=str,
                       default="/data/to/your/dataset/path/VoxCeleb/concatenated_audio",
                       help="VoxCeleb dataset directory path")
    parser.add_argument("--result_dir", type=str, default="./Vox_Results",
                       help="Result output directory")
    
    parser.add_argument("--sample_limit", type=int, default=0,
                       help="Limit number of samples to process (0 means all)")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU device ID to use")
    
    parser.add_argument("--attn_implementation", type=str, default="eager",
                       choices=["eager", "flash_attention_2", "sdpa"],
                       help="Attention mechanism implementation type")
    
    return parser.parse_args()

def prepare_audio_for_processor(audio_path: str, 
                               target_sample_rate: int = 16000,
                               target_channels: int = 1) -> np.ndarray:
    """Prepare audio data for processor - following Qwen2.5-Omni requirements"""
    try:
        try:
            audio, sample_rate = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        except Exception as e:
            print(f"librosa loading failed: {e}")
            audio, sample_rate = sf.read(audio_path)
            
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            if sample_rate != target_sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sample_rate, 
                    target_sr=target_sample_rate
                )
        
        if len(audio) == 0:
            print("Warning: Empty audio, creating 3-second silence")
            audio = np.zeros(target_sample_rate * 3, dtype=np.float32)
        
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sample_rate * 3, dtype=np.float32)
        return silence

def load_concatenated_audio_dataset(root_dir: str, sample_limit: int = 0) -> List[Dict[str, Any]]:
    """Load dataset from concatenated_audio directory, based on age_classification_task_meta.json"""
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    print(f"Loading metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    missing_files = 0
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(wav_path):
            missing_files += 1
            continue
        
        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()  
        speaker_age = item.get("speaker_age", 0)  
        
        all_samples.append({
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "choice_e": item["choice_e"],
            "answer_gt": age_group,
            "task": "Speaker_Age_Classification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    if missing_files > 0:
        print(f"Warning: {missing_files} file paths are invalid or don't exist")
    
    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Applying sample limit: randomly selecting {sample_limit} from {len(all_samples)} samples")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Sample count after limit: {len(all_samples)}")
    
    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    random.shuffle(all_samples)
    
    return all_samples

def process_vox_age_sample(sample, processor, model, timing_stats, device, args):
    """Process a single VoxCeleb age classification sample with TAU/SLUE style processing"""
    
    try:
        audio_path_for_inference = sample.get("audio_path_for_inference", sample.get("wav_path", ""))
        age_group = sample.get("age_group", "unknown")
        speaker_id = sample.get("speaker_id", "unknown")
        answer_gt = sample.get("answer_gt", "").strip()
        
        question = sample.get("question", "")
        choice_a = sample.get("choice_a", "")
        choice_b = sample.get("choice_b", "")
        choice_c = sample.get("choice_c", "")
        choice_d = sample.get("choice_d", "")
        choice_e = sample.get("choice_e", "")
        
        instruction = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
E) {choice_e}

Please select the correct answer (A, B, C, D, or E)."""

        sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please listen to this audio and identify the speaker's age group. Choose the most appropriate option: (a) Young Adult (18-30), (b) Early Career (31-40), (c) Mid Career (41-50), (d) Senior (51-70), (e) Elderly (71+). Answer with only the letter (a, b, c, d, or e)."

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

        if args.sparse and audio_detected:
            args.audio_token_start_index = audio_token_start
            args.audio_token_length = audio_token_length
            configure_DART(model, args)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            prefill_start = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)

            prefill_start.record()

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )

        if torch.cuda.is_available():
            decode_start.record()
            torch.cuda.synchronize()
            decode_end.record()
            torch.cuda.synchronize()

            prefill_time = prefill_start.elapsed_time(decode_start) / 1000.0
            decode_time = decode_start.elapsed_time(decode_end) / 1000.0

        output_tokens = generate_ids[:, inputs.input_ids.shape[1]:] if hasattr(inputs, 'input_ids') else generate_ids
        output_text = processor.decode(output_tokens[0], skip_special_tokens=True).strip()
        
        predicted_answer = extract_age_answer(output_text, sample)
        
        is_correct = (predicted_answer.strip() == answer_gt.strip())
        
        input_tokens = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
        output_token_count = output_tokens.shape[1] if hasattr(output_tokens, 'shape') else len(output_tokens[0])
        
        timing_stats.add_record(
            prefill_time=prefill_time if torch.cuda.is_available() else 0.0,
            decode_time=decode_time if torch.cuda.is_available() else 0.0,
            output_tokens=output_token_count,
            input_tokens=input_tokens,
            age_group=age_group
        )

        try:
            if 'inputs' in locals():
                del inputs
            if 'generate_ids' in locals():
                del generate_ids
            if 'output_tokens' in locals():
                del output_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            print(f"Memory cleanup error: {cleanup_error}")

        return {
            "speaker_id": speaker_id,
            "age_group": age_group,
            "wav_path": audio_path_for_inference,
            "question": question,
            "predicted_answer": predicted_answer,
            "answer_gt": answer_gt,
            "is_correct": is_correct,
            "output_text": output_text,
            "audio_detected": audio_detected,
            "audio_token_start": audio_token_start,
            "audio_token_length": audio_token_length,
            "input_tokens": input_tokens,
            "output_tokens": output_token_count
        }
        
    except Exception as e:
        print(f"Error processing VoxAge sample: {e}")
        traceback.print_exc()
        
        try:
            if 'inputs' in locals():
                del inputs
            if 'generate_ids' in locals():
                del generate_ids
            if 'output_tokens' in locals():
                del output_tokens
            torch.cuda.empty_cache()
        except:
            pass
            
        return {
            "speaker_id": sample.get("speaker_id", "unknown"),
            "age_group": sample.get("age_group", "unknown"),
            "wav_path": sample.get("wav_path", "unknown"),
            "question": sample.get("question", ""),
            "predicted_answer": "",
            "answer_gt": sample.get("answer_gt", ""),
            "is_correct": False,
            "output_text": "ERROR",
            "error": str(e)
        }

def create_age_prompt(sample: Dict[str, Any]) -> str:
    """Create age classification task prompt - consistent with original version"""
    question = sample.get("question", "")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")
    choice_e = sample.get("choice_e", "")
    
    prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
E) {choice_e}

Please select the correct answer (A, B, C, D, or E)."""
    
    full_prompt = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
    return full_prompt

def extract_age_answer(text: str, choices: Dict[str, str]) -> str:
    """Extract age group answer from model output text, handling direct a/b/c/d/e responses"""
    text_lower = text.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)') or text_lower.endswith(' a'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)') or text_lower.endswith(' b'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)') or text_lower.endswith(' c'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)') or text_lower.endswith(' d'):
        return choices["choice_d"]
    if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)') or text_lower.endswith(' e'):
        return choices["choice_e"]
    
    if re.search(r'\ba\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['b', 'c', 'd', 'e']):
        return choices["choice_a"]
    if re.search(r'\bb\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'c', 'd', 'e']):
        return choices["choice_b"]
    if re.search(r'\bc\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'd', 'e']):
        return choices["choice_c"]
    if re.search(r'\bd\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'c', 'e']):
        return choices["choice_d"]
    if re.search(r'\be\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'c', 'd']):
        return choices["choice_e"]
        
    for option, choice_text in choices.items():
        option_letter = option[-1].lower()  
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"{option_letter})" in text_lower:
            return choice_text
    
    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    
    if len(choice_matches) == 1:
        return choice_matches[0]
    
    return ""

def calculate_age_classification_metrics(y_true: List[str], y_pred: List[str], age_groups: List[str] = None) -> Dict[str, Any]:
    """Calculate detailed evaluation metrics for age classification"""
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    if age_groups is None:
        age_groups = ['Young Adult (18-30)', 'Early Career (31-40)', 'Mid Career (41-50)', 'Senior (51-70)', 'Elderly (71+)']
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in age_groups and pred_label in age_groups:
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
            'classification_report': "No valid predictions found",
            'valid_samples': 0,
            'total_samples': len(y_true),
            'age_groups': age_groups
        }
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=age_groups, average=None, zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=age_groups, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=age_groups, average='weighted', zero_division=0
    )
    
    per_class_metrics = {}
    for i, age_group in enumerate(age_groups):
        per_class_metrics[age_group] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=age_groups,
        target_names=[f"Age Group: {ag}" for ag in age_groups],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(f1_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true),
        'age_groups': age_groups
    }

def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Calculate classification metrics: accuracy, precision, recall and F1 score - consistent with original"""
    valid_age_groups = ['Young Adult (18-30)', 'Early Career (31-40)', 'Mid Career (41-50)', 'Senior (51-70)', 'Elderly (71+)']
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in valid_age_groups and t in valid_age_groups]
    
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
    
    label_map = {
        'Young Adult (18-30)': 0,
        'Early Career (31-40)': 1,
        'Mid Career (41-50)': 2,
        'Senior (51-70)': 3,
        'Elderly (71+)': 4
    }
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
        'total_samples': len(predictions)
    }

def main():
    """Main function"""
    args = parse_arguments()
    
    gpu_id = args.gpu_id
    torch.cuda.set_device(gpu_id)
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    print(f"\n=== VoxCeleb Age Classification with Qwen2.5-Omni + DART ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Model Path: {args.model_path}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"Data directory: {args.data_path}")
    if args.sample_limit > 0:
        print(f"Sample limit: {args.sample_limit}")
    print("=" * 60)

    method_name = "sparse" if args.sparse else "base"
    ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    output_file = f'{args.result_dir}/VoxCeleb_age_qwen_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{args.result_dir}/VoxCeleb_age_timing_qwen_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    print("\nLoading Qwen2.5-Omni model...")
    device_map = {"": 0}  
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.disable_talker()
    model.eval()
    
    configure_DART(model, args)
    print("Model loaded successfully")
    
    initial_memory = get_gpu_memory_usage()
    print(f"Initial GPU memory: {initial_memory['allocated']:.2f}GB (allocated), {initial_memory['reserved']:.2f}GB (reserved)")
    
    timing_stats = VoxAgeTimingStats()
    
    print("\nLoading VoxCeleb age classification dataset...")
    samples = load_concatenated_audio_dataset(args.data_path, args.sample_limit)
    
    age_group_counts = {}
    for s in samples:
        group = s["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age group statistics:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    correct_samples = 0
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    print(f"\nStarting to process {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing VoxCeleb age classification samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        for i, sample in enumerate(samples):
            try:
                wav_path = sample.get("wav_path", "")
                
                if not os.path.exists(wav_path):
                    print(f"Warning: Audio file does not exist: {wav_path}")
                    continue
                
                sample["audio_path_for_inference"] = wav_path
                
                result = process_vox_age_sample(sample, processor, model, timing_stats, "cuda", args)
                
                if result["output_text"] == "ERROR":
                    print(f"Skipping sample {i}: processing failed")
                    continue
                
                all_sample_results.append(result)
                
                speaker_id = result["speaker_id"]
                ground_truth = result["answer_gt"] 
                predicted_answer = result["predicted_answer"]
                is_correct = result["is_correct"]
                
                all_predictions.append(predicted_answer)
                all_ground_truths.append(ground_truth)
                
                if is_correct:
                    correct_samples += 1
                
                if len(all_sample_results) > 0:
                    current_accuracy = correct_samples / len(all_sample_results)
                    pbar.set_postfix({
                        'Acc': f'{current_accuracy:.3f}',
                        'Correct': f'{correct_samples}/{len(all_sample_results)}'
                    })
                
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if 'result' in locals():
                        del result
                    if 'sample' in locals() and 'audio_path_for_inference' in sample:
                        sample.pop('audio_path_for_inference', None)
                
                if (i + 1) % 50 == 0:
                    print(f"\n[Memory cleanup] Processed {i + 1} samples, performing deep cleanup...")
                    current_memory = get_gpu_memory_usage()
                    print(f"Current GPU memory: {current_memory['allocated']:.2f}GB (allocated), {current_memory['reserved']:.2f}GB (reserved)")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    after_memory = get_gpu_memory_usage()
                    print(f"GPU memory after cleanup: {after_memory['allocated']:.2f}GB (allocated), {after_memory['reserved']:.2f}GB (reserved)")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                traceback.print_exc()
                
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass
                    
                continue
            
            pbar.update(1)

    print("\nCalculating final evaluation metrics...")
    metrics_result = calculate_metrics(all_predictions, all_ground_truths)
    detailed_metrics = calculate_age_classification_metrics(all_predictions, all_ground_truths, list(age_group_counts.keys()))
    final_stats = timing_stats.get_summary()
    
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    
    age_group_results = {}
    for result in all_sample_results:
        group = result['age_group']  
        if group not in age_group_results:
            age_group_results[group] = {'total': 0, 'correct': 0}
        age_group_results[group]['total'] += 1
        if result['is_correct']:
            age_group_results[group]['correct'] += 1
    
    final_memory = get_gpu_memory_usage()
    
    results = {
        "task_info": {
            "task_name": "VoxCeleb Age Classification",
            "model_name": "Qwen2.5-Omni",
            "framework": "DART Sparse Attention",
            "dataset_path": args.data_path,
            "total_samples_processed": total_samples
        },
        "samples": all_sample_results,
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": correct_samples / total_samples if total_samples > 0 else 0,
            "age_group_results": {
                group: {
                    "total": stats['total'],
                    "correct": stats['correct'],
                    "accuracy": stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                }
                for group, stats in age_group_results.items()
            },
            "metrics": metrics_result,
            "detailed_metrics": detailed_metrics,
            "timing": final_stats,
            "memory_usage": {
                "initial": initial_memory,
                "final": final_memory
            },
            "config": {
                "gpu_id": gpu_id,
                "sparse": args.sparse,
                "pruned_layer": args.pruned_layer,
                "reduction_ratio": args.reduction_ratio,
                "sample_limit": args.sample_limit,
                "model_path": args.model_path,
                "attn_implementation": args.attn_implementation,
                "audio_token_start_index": args.audio_token_start_index,
                "audio_token_length": args.audio_token_length,
                "pivot_audio_token": args.pivot_audio_token,
                "pivot_text_token": args.pivot_text_token
            }
        }
    }

    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

    print(f"Saving timing statistics to: {timing_output_file}")
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing statistics saved to: {timing_output_file}")
    
    print("\n" + "=" * 60)
    print("=== VoxCeleb Age Classification Evaluation Results Summary ===")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    
    print("\n=== Age Group Evaluation Results ===")
    for group, stats in results['summary']['age_group_results'].items():
        print(f"{group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    if detailed_metrics.get('f1_macro'):
        print(f"\n=== Detailed Evaluation Metrics ===")
        print(f"F1 Score (macro average): {detailed_metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted average): {detailed_metrics['f1_weighted']:.4f}")
        print(f"Precision (macro average): {detailed_metrics['precision_macro']:.4f}")
        print(f"Recall (macro average): {detailed_metrics['recall_macro']:.4f}")
    
    if metrics_result.get('f1_score'):
        print(f"\n=== Traditional Evaluation Metrics ===")
        print(f"F1 Score: {metrics_result['f1_score']:.4f}")
        print(f"Precision: {metrics_result['precision']:.4f}")  
        print(f"Recall: {metrics_result['recall']:.4f}")
    
    if final_stats.get('overall_summary'):
        overall_stats = final_stats['overall_summary']
        print(f"\n=== Timing Performance Statistics ===")
        print(f"Average inference time: {overall_stats['avg_total_time']:.4f}s (excluding first sample)")
        print(f"Average Prefill time: {overall_stats['avg_prefill_time']:.4f}s")
        print(f"Average Decode time: {overall_stats['avg_decode_time']:.4f}s")
        print(f"Average tokens/sec: {overall_stats['avg_tokens_per_sec']:.2f}")
    
    print(f"\nGPU memory usage: {final_memory['allocated']:.2f}GB (allocated), {final_memory['reserved']:.2f}GB (reserved)")
    print("=" * 60)
    
    print("\nAge classification evaluation completed!")

if __name__ == "__main__":
    main()