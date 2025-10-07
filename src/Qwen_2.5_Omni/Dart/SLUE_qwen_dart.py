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

slue_json_file = "/data/to/your/dataset/path/SLUE/merged_audio_data.json"
audio_base_dir = "/data/to/your/dataset/path/SLUE"
result_dir = os.environ.get("RESULTS_DIR", './SLUE_Results')

def str_to_bool(value):
    """Convert string to boolean"""
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
    parser.add_argument('--reduction_ratio', type=float, default=0.3, help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    return parser.parse_args()

def configure_DART(model, args):
    """Configure DART sparse attention mechanism - Adapted for Qwen2.5-Omni"""
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

    
    print(f"Qwen2.5-Omni DART configuration: sparse={args.sparse}, "
          f"reduction_ratio={args.reduction_ratio}, "
          f"pruned_layer={args.pruned_layer}")


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
        """Add a timing record, limited to first 100 samples (excluding first one)"""
        if self.total_samples < self.max_timing_samples:
            record = {
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
                "audio_duration": audio_duration,
                "task_type": task_type
            }
            self.timing_records.append(record)
            
            if task_type:
                self.task_type_stats[task_type].append(record)
        
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        if audio_duration:
            self.total_audio_duration += audio_duration
    
    def get_summary(self):
        """Get overall statistics summary"""
        if self.total_samples == 0:
            return {"error": "No timing records available"}
        
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
            if records:
                task_df = pd.DataFrame(records)
                task_summaries[task_type] = {
                    "count": len(records),
                    "avg_total_time": task_df["total_time"].mean(),
                    "avg_prefill_time": task_df["prefill_time"].mean(),
                    "avg_decode_time": task_df["decode_time"].mean(),
                    "avg_tokens_per_sec": task_df["decode_tokens_per_sec"].mean()
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
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

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
        print(f"Error: JSON file format incorrect, expected list format")
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
            print(f"Warning: Task missing 'path' key, skipped: {task}")
            continue
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
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
            print(f"Warning: Cannot get audio info: {full_audio_path}, error: {e}")
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

def extract_answer_choice(response):
    """Extract answer choice (A, B, C, D) from model response, handle various output formats"""
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
        return {"accuracy": accuracy, "predicted_choice": pred, "ground_truth_choice": gt}
    except Exception as e:
        print(f"Error evaluating accuracy: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt}

def calculate_slue_metrics(predictions, ground_truths):
    """Calculate F1 score and other metrics for SLUE tasks"""
    try:
        valid_indices = [i for i, (pred, gt) in enumerate(zip(predictions, ground_truths)) 
                        if pred and gt]
        
        if not valid_indices:
            return {
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "macro_f1": 0.0,
                "valid_samples": 0
            }
        
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_ground_truths = [ground_truths[i] for i in valid_indices]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_ground_truths, valid_predictions, average='weighted', zero_division=0
        )
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            valid_ground_truths, valid_predictions, average='macro', zero_division=0
        )
        
        return {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1,
            "valid_samples": len(valid_predictions)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
            "valid_samples": 0
        }

def main():
    args = parse_arguments()
    
    print(f"\n=== SLUE DART NER Task Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"SLUE JSON file: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)

    method_name = "sparse" if args.sparse else "base"
    ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    output_file = f'{result_dir}/slue_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/slue_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = SLUETimingStats()

    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    
    os.makedirs(result_dir, exist_ok=True)

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

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")

    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    for sample in samples:
        task_type_stats[sample.get("task_name", "unknown")] += 1
        dataset_stats[sample.get("dataset_name", "unknown")] += 1

    print(f"Task type distribution: {dict(task_type_stats)}")
    print(f"Dataset distribution: {dict(dataset_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0
    
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)
    dataset_correct = defaultdict(int)
    dataset_total = defaultdict(int)

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    print(f"Starting evaluation of {len(samples)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    with tqdm(total=len(samples), desc="Processing SLUE NER samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            
            audio_path_for_inference = sample.get("path")
            if not audio_path_for_inference:
                print(f"Sample {idx} missing audio_path")
                continue

            question = sample.get("question", "")
            choice_a = sample.get("choice_a", "")
            choice_b = sample.get("choice_b", "")
            choice_c = sample.get("choice_c", "")
            choice_d = sample.get("choice_d", "")

            instruction = f"{question}\n"
            instruction += f"A: {choice_a}\nB: {choice_b}\nC: {choice_c}\nD: {choice_d}\n"

            qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            task_prompt = "You are a helpful assistant that analyzes audio to answer questions about named entities. Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."
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

            predicted_choice = extract_answer_choice(output_text)
            ground_truth_choice = sample["answer_gt"]

            metrics = evaluate_slue_accuracy(predicted_choice, ground_truth_choice)
            is_correct = metrics["accuracy"] == 1.0

            if is_correct:
                total_accuracy += 1
                task_type_correct[sample["task_name"]] += 1
                dataset_correct[sample["dataset_name"]] += 1

            processed_samples += 1
            task_type_total[sample["task_name"]] += 1
            dataset_total[sample["dataset_name"]] += 1

            timing_stats.add_record(prefill_time, decode_time, output_tokens, input_token_length, 
                                  sample.get("duration", 0), sample["task_name"])

            result = {
                "id": sample["id"],
                "audio_path": sample["path"],
                "task_name": sample["task_name"],
                "dataset_name": sample["dataset_name"],
                "question": sample["question"],
                "choices": {
                    "A": sample["choice_a"],
                    "B": sample["choice_b"],
                    "C": sample["choice_c"],
                    "D": sample["choice_d"]
                },
                "ground_truth_choice": ground_truth_choice,
                "predicted_choice": predicted_choice,
                "raw_response": output_text,
                "is_correct": is_correct,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "total_time": total_time,
                "audio_duration": sample.get("duration", 0)
            }

            results.append(result)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            pbar.update(1)

    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0

    all_predictions = [sample["predicted_choice"] for sample in results]
    all_ground_truths = [sample["ground_truth_choice"] for sample in results]
    
    overall_metrics = calculate_slue_metrics(all_predictions, all_ground_truths)

    task_type_accuracies = {}
    task_type_metrics = {}
    for task_name in task_type_stats.keys():
        if task_type_total[task_name] > 0:
            task_type_accuracies[task_name] = task_type_correct[task_name] / task_type_total[task_name]
            
            task_predictions = [r["predicted_choice"] for r in results if r["task_name"] == task_name]
            task_ground_truths = [r["ground_truth_choice"] for r in results if r["task_name"] == task_name]
            task_type_metrics[task_name] = calculate_slue_metrics(task_predictions, task_ground_truths)

    dataset_accuracies = {}
    dataset_metrics = {}
    for dataset_name in dataset_stats.keys():
        if dataset_total[dataset_name] > 0:
            dataset_accuracies[dataset_name] = dataset_correct[dataset_name] / dataset_total[dataset_name]
            
            dataset_predictions = [r["predicted_choice"] for r in results if r["dataset_name"] == dataset_name]
            dataset_ground_truths = [r["ground_truth_choice"] for r in results if r["dataset_name"] == dataset_name]
            dataset_metrics[dataset_name] = calculate_slue_metrics(dataset_predictions, dataset_ground_truths)

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
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
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
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== SLUE DART Evaluation Results Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Valid samples: {overall_metrics['valid_samples']}")
    print(f"Overall accuracy: {final_accuracy:.3f}")
    print(f"F1 score: {overall_metrics['f1_score']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"Macro F1: {overall_metrics['macro_f1']:.4f}")
    print(f"Number of task types: {len(task_type_stats)}")
    print(f"Number of datasets: {len(dataset_stats)}")
    
    print(f"\nDetailed metrics by task type:")
    for task_name, acc in task_type_accuracies.items():
        f1 = task_type_metrics[task_name]["f1_score"]
        print(f"  {task_name}: Accuracy={acc:.3f}, F1={f1:.4f} ({task_type_correct[task_name]}/{task_type_total[task_name]})")
    
    print(f"\nDetailed metrics by dataset:")
    for dataset_name, acc in dataset_accuracies.items():
        f1 = dataset_metrics[dataset_name]["f1_score"]
        print(f"  {dataset_name}: Accuracy={acc:.3f}, F1={f1:.4f} ({dataset_correct[dataset_name]}/{dataset_total[dataset_name]})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\nTiming statistics (based on first {timing_sample_count} samples, excluding 1st):")
    print(f"Statistical samples: {overall_summary.get('total_samples', 0)}")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/sec")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()