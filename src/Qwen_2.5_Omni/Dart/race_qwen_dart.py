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
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="RACE with Qwen2.5-Omni DART")
    
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
    parser.add_argument('--pivot_image_token', type=int, default=None, help='Number of key image tokens')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='Number of key audio tokens')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='Number of key text tokens')
    
    parser.add_argument('--sample_limit', type=int, default=0, help='Sample limit (0 for unlimited)')
    
    return parser.parse_args()

def configure_DART(model, args):
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
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

class RaceTimingStats:
    def __init__(self):
        self.timing_records = []
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration):
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
            "audio_duration": audio_duration,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        if self.total_samples == 0:
            return {}
        
        return {
            "total_samples": self.total_samples,
            "avg_prefill_time": self.total_prefill_time / self.total_samples,
            "avg_decode_time": self.total_decode_time / self.total_samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def clean_text_response(response):
    if not response:
        return ""
    
    resp = response.strip().upper()
    
    for ch in resp:
        if ch in ["A", "B", "C", "D"]:
            return ch
    
    words = resp.split()
    for word in words:
        for ch in word:
            if ch in ["A", "B", "C", "D"]:
                return ch
    
    return words[0] if words else ""

def calculate_race_metrics(y_true, y_pred, subset_labels=None):
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    clean_subset_labels = [] if subset_labels is not None else None
    
    valid_labels = ['A', 'B', 'C', 'D']
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in valid_labels and pred_label in valid_labels:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
            if subset_labels is not None:
                clean_subset_labels.append(subset_labels[i])
    
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
            'subset_metrics': {},
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
    
    subset_metrics = {}
    if clean_subset_labels is not None:
        unique_subsets = list(set(clean_subset_labels))
        for subset in unique_subsets:
            subset_indices = [i for i, label in enumerate(clean_subset_labels) if label == subset]
            if subset_indices:
                subset_true = [clean_y_true[i] for i in subset_indices]
                subset_pred = [clean_y_pred[i] for i in subset_indices]
                
                subset_accuracy = accuracy_score(subset_true, subset_pred)
                subset_precision, subset_recall, subset_f1, _ = precision_recall_fscore_support(
                    subset_true, subset_pred, average='macro', zero_division=0
                )
                
                subset_metrics[subset] = {
                    'accuracy': float(subset_accuracy),
                    'precision': float(subset_precision),
                    'recall': float(subset_recall),
                    'f1_score': float(subset_f1),
                    'samples': len(subset_indices)
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
        'subset_metrics': subset_metrics,
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true),
        'class_labels': valid_labels
    }

def main():
    args = parse_arguments()
    
    data_path_root = '/data/to/your/dataset/path//race_audio'
    
    result_dir = './Race_Results'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    print(f"\n=== RACE DART Evaluation Configuration ===")
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
    output_file = f'{result_dir}/race_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/race_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

    timing_stats = RaceTimingStats()

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

    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: Benchmark file not found: {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f"Sample count limited to: {len(benchmark)}")

    results = []

    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,        
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(benchmark)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc="RACE Evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            audio_rel = sample["audio_path"]
            audio_path_for_inference = os.path.join(data_path_root, audio_rel)
            
            if not os.path.exists(audio_path_for_inference):
                print(f"Warning: Audio file not found: {audio_path_for_inference}")
                continue

            question = sample["question"]
            options = sample["options"]
            
            instruction = f"Question: {question}\n\nOptions:\n"
            for i, opt in enumerate(options):
                letter = chr(65 + i)  
                instruction += f"{letter}. {opt}\n"
            instruction += "\nRespond with only the letter of the correct option (A, B, C, or D)."

            qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            task_prompt = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
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

            predicted_choice = clean_text_response(output_text)
            ground_truth_choice = sample["answer"]

            is_correct = predicted_choice == ground_truth_choice

            if is_correct:
                correct_count += 1

            if "high" in audio_rel:
                total_high += 1
                if is_correct:
                    correct_high += 1
                difficulty = "high"
            elif "middle" in audio_rel:
                total_middle += 1
                if is_correct:
                    correct_middle += 1
                difficulty = "middle"
            else:
                difficulty = "unknown"

            result = {
                "id": sample.get("id", f"sample_{idx}"),
                "question": sample["question"],
                "options": sample["options"],
                "correct_answer": ground_truth_choice,
                "predicted_answer": predicted_choice,
                "correct": is_correct,
                "difficulty": difficulty,
                "audio_path": sample["audio_path"],
                "response_text": output_text,
                "gt": ground_truth_choice,  
                "pred": predicted_choice,   
                "subset": difficulty,  
                "timing": {
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": total_time,
                    "input_tokens": input_token_length,
                    "output_tokens": output_tokens,
                    "tokens_per_sec": output_tokens/decode_time if decode_time > 0 else 0
                }
            }

            results.append(result)
            
            if idx > 0 and idx <= 100:
                timing_stats.add_record(prefill_time, decode_time, output_tokens, input_token_length, 0)

            current_acc = correct_count / (idx + 1) if idx >= 0 else 0
            progress_bar.set_postfix({
                'Acc': f"{current_acc:.3f}",
                'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
            })

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    y_true = [r["gt"] for r in results]
    y_pred = [r["pred"] for r in results]
    subset_labels = [r["subset"] for r in results]
    
    detailed_metrics = calculate_race_metrics(y_true, y_pred, subset_labels)

    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "high_accuracy": correct_high / total_high * 100 if total_high > 0 else 0,
        "middle_accuracy": correct_middle / total_middle * 100 if total_middle > 0 else 0,
        "high_correct": correct_high,
        "high_total": total_high,
        "middle_correct": correct_middle,
        "middle_total": total_middle,
        "sklearn_metrics": detailed_metrics,  
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "timing_sample_count": min(100, max(0, len(results) - 1))
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    print(f"Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"Saving timing stats to: {timing_output_file}")
    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== RACE Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    
    metrics = detailed_metrics
    print(f"\n=== Detailed Evaluation Metrics (sklearn) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (macro average): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted average): {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro average): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro average): {metrics['recall_macro']:.4f}")
    
    print(f"\n=== Per-Option Evaluation Metrics ===")
    for choice, per_class_metrics in metrics['per_class_metrics'].items():
        print(f"Option {choice}:")
        print(f"  Precision: {per_class_metrics['precision']:.4f}")
        print(f"  Recall: {per_class_metrics['recall']:.4f}")
        print(f"  F1 Score: {per_class_metrics['f1_score']:.4f}")
        print(f"  Samples: {per_class_metrics['support']}")
    
    print(f"\n=== Subset Evaluation Metrics ===")
    for subset, subset_metrics in metrics['subset_metrics'].items():
        print(f"{subset.upper()} set:")
        print(f"  Accuracy: {subset_metrics['accuracy']:.4f}")
        print(f"  Precision: {subset_metrics['precision']:.4f}")
        print(f"  Recall: {subset_metrics['recall']:.4f}")
        print(f"  F1 Score: {subset_metrics['f1_score']:.4f}")
        print(f"  Samples: {subset_metrics['samples']}")
    
    print(f"\n=== Traditional Accuracy Statistics ===")
    if total_high > 0:
        print(f"HIGH set accuracy: {correct_high/total_high*100:.2f}% ({correct_high}/{total_high})")
    if total_middle > 0:
        print(f"MIDDLE set accuracy: {correct_middle/total_middle*100:.2f}% ({correct_middle}/{total_middle})")
    
    timing_summary = timing_stats.get_summary()
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\n=== Inference Time Statistics ===")
    print(f"Average inference time: {timing_summary.get('avg_total_time', 0):.4f} seconds (first {timing_sample_count} samples, excluding first sample)")
    print(f"Average prefill time: {timing_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average decode time: {timing_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    
    print(f"\n=== Detailed Classification Report ===")
    print(metrics['classification_report'])
    
    print(f"\nResults saved to: {output_file}")
    print(f"Timing stats saved to: {timing_output_file}")

if __name__ == "__main__":
    main()