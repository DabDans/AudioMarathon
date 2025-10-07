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
import librosa
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score, precision_score, recall_score

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")
print(f"CUDA_VISIBLE_DEVICES: {gpu_temp}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="VESUS with Qwen2.5-Omni DART")
    
    parser.add_argument("--model-path", type=str, 
                       default="/data/to/your/Qwen_2.5_Model/path/",
                       help="Qwen2.5-Omni model path")
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2', 
                       help='Attention implementation method')
    
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='Number of pruned layers')
    parser.add_argument('--image_token_start_index', type=int, default=None, help='Image token start index')
    parser.add_argument('--image_token_length', type=int, default=None, help='Image token length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='Audio token start index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='Audio token length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='Retention ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='Key image token count')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='Key audio token count')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='Key text token count')
    
    parser.add_argument('--sample_limit', type=int, default=0, help='Sample limit (0 for unlimited)')
    
    return parser.parse_args()

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
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def calculate_emotion_metrics(predictions, ground_truths, emotion_labels):
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in emotion_labels and t in emotion_labels]
    
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
    
    label_map = {label: idx for idx, label in enumerate(sorted(emotion_labels))}
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

class VESUSTimingStats:
    def __init__(self):
        self.timing_records = []
        self.emotion_stats = defaultdict(list)
        self.person_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.use_cuda_events = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   emotion_label=None, person_id=None):
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
            "emotion_label": emotion_label,
            "person_id": person_id
        }
        
        self.timing_records.append(record)
        
        if emotion_label:
            self.emotion_stats[emotion_label].append(record)
        
        if person_id:
            self.person_stats[person_id].append(record)
    
    def get_summary(self):
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
        
        emotion_summaries = {}
        for emotion, records in self.emotion_stats.items():
            if len(records) > 0:
                emotion_summaries[emotion] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "emotion_summaries": emotion_summaries
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def process_vesus_sample(sample, processor, model, timing_stats, device, args):
    """Process a single VESUS emotion recognition sample with TAU/SLUE style processing"""
    
    try:
        audio_path_for_inference = sample.get("audio_path_for_inference", sample.get("path", ""))
        emotion_label = sample.get("emotion_label", "unknown")
        person_id = sample.get("person_id", "unknown")
        answer_gt = sample.get("answer_gt", "").upper()
        
        question = sample.get("question", "What emotion is expressed in this audio segment?")
        choice_a = sample.get("choice_a", "")
        choice_b = sample.get("choice_b", "")
        choice_c = sample.get("choice_c", "")
        choice_d = sample.get("choice_d", "")
        
        instruction = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D)."""

        qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        task_prompt = "You are a helpful assistant that analyzes audio to answer questions about emotions. Please listen to the audio carefully and select the correct answer."
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
        
        choices = {
            'choice_a': sample.get('choice_a', ''),
            'choice_b': sample.get('choice_b', ''),
            'choice_c': sample.get('choice_c', ''),
            'choice_d': sample.get('choice_d', ''),
        }
        predicted_answer = extract_emotion_answer(output_text, choices)
        
        is_correct = (predicted_answer.upper() == answer_gt.upper())
        
        input_tokens = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
        output_token_count = output_tokens.shape[1] if hasattr(output_tokens, 'shape') else len(output_tokens[0])
        
        timing_stats.add_record(
            prefill_time=prefill_time if torch.cuda.is_available() else 0.0,
            decode_time=decode_time if torch.cuda.is_available() else 0.0,
            output_tokens=output_token_count,
            input_tokens=input_tokens,
            emotion_label=emotion_label,
            person_id=person_id
        )

        return {
            "audio_path": audio_path_for_inference,
            "question": question,
            "predicted_answer": predicted_answer,
            "answer_gt": answer_gt,
            "is_correct": is_correct,
            "emotion_label": emotion_label,
            "person_id": person_id,
            "output_text": output_text,
            "audio_detected": audio_detected,
            "audio_token_start": audio_token_start,
            "audio_token_length": audio_token_length,
            "input_tokens": input_tokens,
            "output_tokens": output_token_count
        }
        
    except Exception as e:
        print(f"Error processing VESUS sample: {e}")
        traceback.print_exc()
        return {
            "audio_path": sample.get("path", "unknown"),
            "question": sample.get("question", ""),
            "predicted_answer": "",
            "answer_gt": sample.get("answer_gt", ""),
            "is_correct": False,
            "emotion_label": sample.get("emotion_label", "unknown"),
            "person_id": sample.get("person_id", "unknown"),
            "output_text": "ERROR",
            "error": str(e)
        }

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    
    try:
        try:
            audio, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)
        except Exception as e:
            print(f"librosa loading failed: {e}, trying soundfile...")
            audio, sample_rate = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sample_rate != target_sr:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        
        if len(audio) == 0:
            print(f"Warning: empty audio file: {audio_path}")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def _normalize_person_id(person_id) -> Optional[int]:
    if person_id is None:
        return None
    try:
        return int(person_id)
    except (ValueError, TypeError):
        s = str(person_id).strip().lower()
        m = re.search(r"(\d+)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def load_vesus_dataset(json_file_path, data_path):
    
    if not os.path.exists(json_file_path):
        print(f"Error: dataset file not found: {json_file_path}")
        return []
    
    print(f"Loading VESUS emotion dataset: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        valid_samples = []
        filtered_count = 0
        for item in data:
            if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
                emotion_label = str(item.get('emotion_label', '')).strip().lower()
                raw_person = item.get('person_id', '')
                pid = _normalize_person_id(raw_person)

                if (pid in {2, 10}) and (emotion_label == 'happy'):
                    filtered_count += 1
                    continue

                valid_samples.append(item)

        print(f"Filtered {filtered_count} samples (person 2 and person 10 happy emotions)")
        print(f"Loaded {len(valid_samples)} valid samples")
        
        emotion_counts = defaultdict(int)
        person_emotion_counts = defaultdict(lambda: defaultdict(int))
        for sample in valid_samples:
            emotion = sample.get('emotion_label', 'unknown')
            pid = _normalize_person_id(sample.get('person_id', 'unknown'))
            person = str(pid) if pid is not None else str(sample.get('person_id', 'unknown'))
            emotion_counts[emotion] += 1
            person_emotion_counts[person][emotion] += 1
        
        print(f"Emotion distribution: {dict(emotion_counts)}")
        print(f"Person emotion distribution:")
        for person, emotions in person_emotion_counts.items():
            tag = " (filtered happy)" if _normalize_person_id(person) in {2, 10} else ""
            print(f"  person {person}: {dict(emotions)}{tag}")
        
        return valid_samples
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []

def extract_emotion_answer(text, choices):
    text_lower = text.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return "A"
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return "B"
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return "C"
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return "D"
    
    option_patterns = {
        'A': ["option a", "choice a", "a)", "(a)"],
        'B': ["option b", "choice b", "b)", "(b)"],
        'C': ["option c", "choice c", "c)", "(c)"],
        'D': ["option d", "choice d", "d)", "(d)"]
    }
    
    for option, patterns in option_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return option
    
    emotion_keywords = {
        'angry': ['anger', 'frustrated', 'mad', 'furious'],
        'happy': ['joy', 'cheerful', 'pleased', 'delighted'],
        'sad': ['sadness', 'melancholy', 'depressed', 'sorrow'],
        'fearful': ['fear', 'anxiety', 'scared', 'afraid'],
        'monotone': ['flat', 'emotionless', 'neutral', 'bland']
    }
    
    for choice_key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
        if choice_key in choices:
            choice_text = choices[choice_key].lower()
            for emotion, keywords in emotion_keywords.items():
                if emotion in choice_text or any(keyword in choice_text for keyword in keywords):
                    if any(keyword in text_lower for keyword in keywords) or emotion in text_lower:
                        return choice_key[-1].upper()
    
    return ""

def create_emotion_prompt(sample):
    question = sample.get("question", "What emotion is expressed in this audio segment?")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")
    
    prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D)."""
    
    return prompt

def calculate_vesus_metrics(y_true, y_pred):
    """
    Calculate detailed evaluation metrics for VESUS emotion recognition task
    
    Args:
        y_true: True label list (A/B/C/D format)
        y_pred: Predicted label list (A/B/C/D format)
        
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
    args = parse_arguments()
    
    env_data_path = os.environ.get('VESUS_DATA_PATH', '').strip()
    probe_paths = []
    if env_data_path:
        probe_paths.append(env_data_path)
    probe_paths.append(os.getcwd())
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        probe_paths.append(script_dir)
        parent_dir = os.path.dirname(script_dir)
        if parent_dir and parent_dir not in probe_paths:
            probe_paths.append(parent_dir)
    except Exception:
        pass
    probe_paths.append('/data/to/your/dataset/path//VESUS')

    data_path = None
    emotion_json_file = None
    for base in probe_paths:
        if not base:
            continue
        candidate_json = os.path.join(base, 'audio_emotion_dataset.json')
        if os.path.isfile(candidate_json):
            data_path = os.path.abspath(base)
            emotion_json_file = candidate_json
            break
    if data_path is None:
        data_path = os.path.abspath(probe_paths[-1])
        emotion_json_file = os.path.join(data_path, 'audio_emotion_dataset.json')
    
    result_dir = './VESUS_Results'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path}")
    print(f"Result directory: {result_dir}")

    print(f"\n=== VESUS DART Evaluation Configuration ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path}")
    print(f"Result directory: {result_dir}")
    print("=" * 50)

    method_name = "sparse" if args.sparse else "base"
    ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    output_file = f'{result_dir}/vesus_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/vesus_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

    timing_stats = VESUSTimingStats()

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
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.disable_talker()
    
    configure_DART(model, args)
    print("Model loaded successfully")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    samples = load_vesus_dataset(emotion_json_file, data_path)
    
    if not samples:
        print("Error: No data samples found")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")

    results = []
    total_correct = 0
    emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    person_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Screen or non-interactive environment detected, using simplified progress display")
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    print(f"Starting evaluation of {len(samples)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), 
                       desc="VESUS Evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            audio_path = sample.get("path", "")
            audio_full_path = os.path.join(data_path, audio_path)
            
            if not os.path.exists(audio_full_path):
                print(f"Warning: Audio file not found: {audio_full_path}")
                continue
            
            sample["audio_path_for_inference"] = audio_full_path
            
            result = process_vesus_sample(sample, processor, model, timing_stats, "cuda", args)
            
            if result["output_text"] == "ERROR":
                print(f"Skipping sample {idx}: processing failed")
                continue
            
            results.append(result)
            
            emotion_label = result.get("emotion_label", "unknown")
            person_id = result.get("person_id", "unknown")
            is_correct = bool(result.get("is_correct", False))
            
            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1
            
            if is_correct:
                total_correct += 1
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1
            
            if len(results) > 0:
                current_accuracy = total_correct / len(results)
                progress_bar.set_postfix({
                    'Acc': f'{current_accuracy:.3f}',
                    'Correct': f'{total_correct}/{len(results)}'
                })
            
            if idx % 50 == 0 and idx > 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            continue

    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    y_true = [r.get("answer_gt", "") for r in results]
    y_pred = [r.get("predicted_answer", "") for r in results]
    
    detailed_metrics = calculate_vesus_metrics(y_true, y_pred)

    all_predictions = [result["predicted_answer"] for result in results]
    all_ground_truths = [result["answer_gt"] for result in results]
    all_emotion_labels = list(set(all_ground_truths))
    
    emotion_metrics = calculate_emotion_metrics(all_predictions, all_ground_truths, all_emotion_labels)

    emotion_accuracies = {}
    for emotion, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracies[emotion] = stats["correct"] / stats["total"]
        else:
            emotion_accuracies[emotion] = 0.0

    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]
        else:
            person_accuracies[person] = 0.0

    summary = {
        "total_samples": total_samples,
        "correct_samples": total_correct,
        "overall_accuracy": overall_accuracy,
        "metrics": emotion_metrics,
        "sklearn_metrics": detailed_metrics,
        "emotion_stats": dict(emotion_stats),
        "emotion_accuracies": emotion_accuracies,
        "person_stats": dict(person_stats),
        "person_accuracies": person_accuracies,
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file,
            "timing_sample_count": min(100, max(0, len(results) - 1))
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== VESUS Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    
    sklearn_metrics = detailed_metrics
    print(f"\n=== Detailed Evaluation Metrics (sklearn) ===")
    print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro Average): {sklearn_metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted Average): {sklearn_metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro Average): {sklearn_metrics['precision_macro']:.4f}")
    print(f"Recall (Macro Average): {sklearn_metrics['recall_macro']:.4f}")
    
    print(f"\n=== Per-Option Evaluation Metrics ===")
    for choice, per_class_metrics in sklearn_metrics['per_class_metrics'].items():
        print(f"Option {choice}:")
        print(f"  Precision: {per_class_metrics['precision']:.4f}")
        print(f"  Recall: {per_class_metrics['recall']:.4f}")
        print(f"  F1 Score: {per_class_metrics['f1_score']:.4f}")
        print(f"  Support: {per_class_metrics['support']}")
    
    print(f"\n=== Traditional Emotion Accuracy Statistics ===")
    print(f"F1 Score: {emotion_metrics['f1_score']:.4f}")
    print(f"Precision: {emotion_metrics['precision']:.4f}")
    print(f"Recall: {emotion_metrics['recall']:.4f}")
    print(f"Valid samples: {emotion_metrics['valid_samples']}/{emotion_metrics['total_samples']}")
    print(f"Per-emotion accuracy:")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\n=== Inference Time Statistics ===")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f}s (first {timing_sample_count} samples, excluding first)")
    print(f"Average prefill time: {overall_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Average decode time: {overall_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/s")
    
    print(f"\n=== Detailed Classification Report ===")
    print(sklearn_metrics['classification_report'])
    
    print(f"\nResults saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()