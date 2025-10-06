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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score, precision_score, recall_score

random.seed(42)
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
_AUDIO_SPECIAL_TOKEN_ID = 151648

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
    """Convert string to boolean"""
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GTZAN with Qwen2.5-Omni DART")
    
    parser.add_argument("--model-path", type=str, 
                       default="/data/to/your/Qwen_2.5Omni-3B/Model/folder",
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
    """Configure DART sparse attention mechanism"""
    if args.sparse:
        DART_config = {
            "K": args.pruned_layer,
            "image_token_start_index": args.image_token_start_index, 
            "image_token_length": args.image_token_length,
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            "reduction_ratio": args.reduction_ratio,
            "pivot_image_token": args.pivot_image_token,
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            "text_length": 1,
        }
        
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
            model.thinker.model.config.DART_config = DART_config
        elif hasattr(model, 'config'):
            model.config.DART_config = DART_config
        else:
            print("Warning: Unable to set DART configuration")
    else:
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
            model.thinker.model.config.DART_config = None
        elif hasattr(model, 'config'):
            model.config.DART_config = None

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

class GTZANTimingStats:
    """Track GTZAN task inference timing statistics using CUDA Events for precise measurement"""
    def __init__(self):
        self.timing_records = []
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration, genre=None):
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
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "genre": genre
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get overall statistical summary"""
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

def calculate_music_metrics(predictions, ground_truths, genre_labels):
    """Calculate music genre classification metrics: accuracy, precision, recall and F1 score"""
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['A', 'B', 'C', 'D'] and t in ['A', 'B', 'C', 'D']]
    
    if not valid_pairs:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'valid_samples': 0,
            'total_samples': len(predictions)
        }
    
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
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

def clean_text_response(response):
    """Clean model response for GTZAN task, keep only first character as option label"""
    return extract_music_genre_answer(response, ['A', 'B', 'C', 'D'])

def extract_music_genre_answer(response, options):
    """Extract music genre answer from model response"""
    if not response:
        return ""
    
    response = response.strip().upper()
    
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    for choice in ['A', 'B', 'C', 'D']:
        if response.startswith(choice) and len(response) <= 3:
            return choice
    
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    
    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)
    
    return ""

def create_gtzan_prompt(question, options):
    """Create GTZAN task prompt (adapted to Qwen2.5 format while maintaining consistent content)"""
    instruction = "Listen to this audio segment and identify the music genre based on what you hear."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted_options += f"{letter}. {opt}\n"
    
    prompt = f"{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}"
    
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

def calculate_gtzan_metrics(y_true, y_pred):
    """
    Calculate detailed evaluation metrics for GTZAN music classification task
    
    Args:
        y_true: Ground truth label list (A/B/C/D format)
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
    
    data_path_root = os.path.abspath('/data/to/your/dataset/path/GTZAN/concatenated_audio')
    metadata_file = os.path.join(data_path_root, 'music_genre_classification_meta.json')
    
    result_dir = './GTZAN_Results'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    print(f"\n=== GTZAN DART Evaluation Configuration ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    method_name = "sparse" if args.sparse else "base"
    ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    output_file = f'{result_dir}/gtzan_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/gtzan_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = GTZANTimingStats()

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading GTZAN metadata: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file does not exist: {metadata_file}")
        return
    
    samples = load_gtzan_metadata(metadata_file)

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")

    genre_stats = {}
    for sample in samples:
        genre = sample.get("genre_label", "unknown")
        genre_stats[genre] = genre_stats.get(genre, 0) + 1
    
    print(f"Genre statistics: {genre_stats}")

    results = []
    correct_count = 0
    genre_correct = {genre: 0 for genre in genre_stats.keys()}
    genre_total = {genre: 0 for genre in genre_stats.keys()}

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    print(f"Starting evaluation on {len(samples)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loaded - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    with tqdm(total=len(samples), desc="Processing GTZAN music genre samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            
            audio_rel = sample["path"]
            audio_full = os.path.join(data_path_root, 'wav', audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"Warning: Audio file does not exist: {audio_full}")
                continue

            audio_path_for_inference = audio_full

            options = [
                sample["choice_a"],
                sample["choice_b"], 
                sample["choice_c"],
                sample["choice_d"]
            ]

            instruction = "Listen to this audio segment and identify the music genre based on what you hear.\n"
            instruction += f"A: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\n"
            instruction += "Respond with only the letter of your answer (A, B, C, or D)."

            qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            task_prompt = "You are a helpful audio analysis assistant."
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
                output = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True)
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

            current_genre = sample.get("genre_label", "unknown")
            genre_total[current_genre] = genre_total.get(current_genre, 0) + 1

            predicted_answer = extract_music_genre_answer(output_text, options)
            ground_truth = sample["answer_gt"].upper()

            is_correct = (predicted_answer == ground_truth)

            if is_correct:
                correct_count += 1
                genre_correct[current_genre] = genre_correct.get(current_genre, 0) + 1

            if idx > 0 and idx <= 100:
                timing_stats.add_record(prefill_time, decode_time, output_tokens, input_token_length, 
                                      sample.get("duration", 0), current_genre)

            result = {
                "idx": idx,
                "uniq_id": sample.get("uniq_id", idx),
                "genre_label": current_genre,
                "path": audio_rel,
                "question": sample["question"],
                "options": options,
                "predicted_answer": predicted_answer,
                "correct_answer": ground_truth,
                "correct": is_correct,
                "response_text": output_text,
                "gt": ground_truth,
                "pred": predicted_answer,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "total_time": total_time
            }

            results.append(result)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            pbar.update(1)

    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    genre_accuracies = {}
    for genre in genre_stats.keys():
        if genre_total.get(genre, 0) > 0:
            genre_accuracies[genre] = genre_correct.get(genre, 0) / genre_total[genre] * 100

    y_true = [r["gt"] for r in results]
    y_pred = [r["pred"] for r in results]
    
    detailed_metrics = calculate_gtzan_metrics(y_true, y_pred)

    predictions = [r["predicted_answer"] for r in results]
    ground_truths = [r["correct_answer"] for r in results]
    metrics = calculate_music_metrics(predictions, ground_truths, list(genre_stats.keys()))

    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "genre_stats": genre_stats,
        "genre_accuracies": genre_accuracies,
        "genre_correct": genre_correct,
        "genre_total": genre_total,
        "metrics": metrics,
        "sklearn_metrics": detailed_metrics,
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "data_path": data_path_root,
            "timing_sample_count": min(100, max(0, len(results) - 1))
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== GTZAN Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    
    sklearn_metrics = detailed_metrics
    print(f"\n=== Detailed Evaluation Metrics (sklearn) ===")
    print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {sklearn_metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {sklearn_metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro): {sklearn_metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {sklearn_metrics['recall_macro']:.4f}")
    
    print(f"\n=== Per Option Evaluation Metrics ===")
    for choice, per_class_metrics in sklearn_metrics['per_class_metrics'].items():
        print(f"Option {choice}:")
        print(f"  Precision: {per_class_metrics['precision']:.4f}")
        print(f"  Recall: {per_class_metrics['recall']:.4f}")
        print(f"  F1 Score: {per_class_metrics['f1_score']:.4f}")
        print(f"  Support: {per_class_metrics['support']}")
    
    print(f"\n=== Genre Accuracy Statistics ===")
    for genre, acc in genre_accuracies.items():
        correct_num = genre_correct.get(genre, 0)
        total_num = genre_total.get(genre, 0)
        print(f"  {genre}: {acc:.2f}% ({correct_num}/{total_num})")
    
    timing_summary = timing_stats.get_summary()
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\n=== Inference Time Statistics ===")
    print(f"Average inference time: {timing_summary.get('avg_total_time', 0):.4f}s (first {timing_sample_count} samples, excluding first)")
    print(f"Average prefill time: {timing_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Average decode time: {timing_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Average throughput: {timing_summary.get('avg_decode_tokens_per_sec', 0):.2f} tokens/s")
    
    print(f"\n=== Detailed Classification Report ===")
    print(sklearn_metrics['classification_report'])
    
    print(f"\nResults saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()