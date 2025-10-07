#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support, classification_report
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
import gc
import re
import traceback

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

sys.path.append("/data/to/your/Qwen_2.5/folder")
from modeling_qwen2_5_omni_dart import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/to/your/Qwen_2.5Omni-3B/Model/folder")
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2', help='attn_implementation')
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
    parser.add_argument('--sample_limit', type=int, default=0, help='Limit number of samples for testing')
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

def load_desed_qa_dataset(json_file, audio_base_dir):
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []
    
    print(f"Loading DESED task JSON: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    
    if isinstance(data, dict) and 'tasks' in data:
        tasks = data['tasks']
        print(f"Loaded {len(tasks)} tasks from JSON (dictionary format)")
    elif isinstance(data, list):
        tasks = data
        print(f"Loaded {len(tasks)} tasks from JSON (list format)")
    else:
        print(f"Error: Incorrect JSON file format, expected list format or dictionary with 'tasks' field")
        return []
    
    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        relative_path = task.get("path", "")
        if not relative_path:
            if "audio" in task and "path" in task["audio"]:
                relative_path = task["audio"]["path"]
            else:
                print(f"Warning: Task {i} missing audio path information, skipped")
                continue
        
        full_audio_path = os.path.join(audio_base_dir, relative_path)
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue
        
        task_type = task.get("task_type", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        
        if "choices" in task and isinstance(task["choices"], dict):
            choices = task["choices"]
            choice_a = choices.get("A", "")
            choice_b = choices.get("B", "")
            choice_c = choices.get("C", "")
            choice_d = choices.get("D", "")
        else:
            choice_a = task.get("choice_a", "")
            choice_b = task.get("choice_b", "")
            choice_c = task.get("choice_c", "")
            choice_d = task.get("choice_d", "")
        
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"Unable to read audio file info {full_audio_path}: {e}")
            continue
        
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_type": task_type,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "original_events": task.get("all_events", task.get("original_events", [])),
            "all_events": task.get("all_events", []),
            "primary_event": task.get("primary_event", ""),
            "correct_event": task.get("correct_event", ""),
            "path_extracted_event": task.get("path_extracted_event", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"qa_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[task_type] += 1
    
    if missing_files > 5:
        print(f"Total of {missing_files} audio files do not exist")
    
    print(f"Successfully loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    
    return dataset

def extract_answer_choice(response):
    if not response:
        return ""
    
    if "assistant\n" in response:
        assistant_parts = response.split("assistant\n")
        if len(assistant_parts) > 1:
            assistant_response = assistant_parts[-1].strip()
            response = assistant_response
    
    response = response.strip().upper()
    
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    
    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)
    
    return ""

def calculate_desed_metrics(y_true, y_pred):
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
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=labels,
        target_names=['Choice A', 'Choice B', 'Choice C', 'Choice D'],
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
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true)
    }

class DESEDTimingStats:
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
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
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "task_type": task_type
        }
        
        self.timing_records.append(record)
        
        if task_type:
            self.task_type_stats[task_type].append(record)
    
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
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_audio_duration": self.total_audio_duration,
            "avg_audio_duration": self.total_audio_duration / self.total_samples if self.total_samples > 0 else 0
        }
        
        task_summaries = {}
        for task_type, records in self.task_type_stats.items():
            if len(records) > 0:
                task_summaries[task_type] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "task_summaries": task_summaries
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def main():
    args = parse_arguments()
    
    random.seed(42)
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")
    
    sample_limit = args.sample_limit if args.sample_limit > 0 else int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    
    qa_json_file = "/data/to/your/dataset/path/DESED/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json"
    audio_base_dir = "/data/to/your/dataset/path/DESED/DESED_dataset/concatenated_audio"
    result_dir = os.environ.get("RESULTS_DIR", './DESED_Results')
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n=== DESED Sound Event Detection Evaluation Configuration (Qwen2.5-Omni + DART) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Model path: {args.model_path}")
    print(f"DART sparse attention: {args.sparse}")
    if args.sparse:
        print(f"Pruned layer: {args.pruned_layer}")
        print(f"Retention ratio: {args.reduction_ratio}")
        print(f"Audio token start index: {args.audio_token_start_index}")
        print(f"Audio token length: {args.audio_token_length}")
    print(f"Task JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)

    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)
    if not samples:
        print("Unable to load dataset, exiting")
        return
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, processing {len(samples)} samples")

    method_name = f"dart_{args.reduction_ratio}" if args.sparse else "base"
    output_file = f'{result_dir}/desed_sound_event_detection_results_qwen25_dart_{method_name}.json'
    timing_output_file = f'{result_dir}/desed_sound_event_detection_timing_stats_qwen25_dart_{method_name}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = DESEDTimingStats()

    print("Loading Qwen2.5-Omni model...")
    device_map = {"": 0}
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    
    configure_DART(model, args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Using dataset with {len(samples)} samples")

    task_type_stats = defaultdict(int)
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_stats[task_type] += 1
    
    print(f"Task type statistics: {dict(task_type_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0
    
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    print(f"Starting evaluation of {len(samples)} samples...")
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="DESED Sound Event Detection Evaluation (Qwen2.5+DART)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            audio_path_for_inference = sample["audio"]["path"] if "audio" in sample and "path" in sample["audio"] else sample.get("path", "")
            if not os.path.exists(audio_path_for_inference):
                print(f"Audio file does not exist: {audio_path_for_inference}")
                continue

            ground_truth_choice = sample.get("answer_gt", "")
            task_type = sample.get("task_type", "unknown")

            question = sample.get("question", "")
            choice_a = sample.get("choice_a", "")
            choice_b = sample.get("choice_b", "")
            choice_c = sample.get("choice_c", "")
            choice_d = sample.get("choice_d", "")
            instruction = f"Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D). {question}\nA: {choice_a}\nB: {choice_b}\nC: {choice_c}\nD: {choice_d}\n"

            qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            task_prompt = "You are a helpful assistant that analyzes audio to detect and classify sound events. Please listen carefully and select the most appropriate answer from the given choices."
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

            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            prefill_start_event.record()
            with torch.no_grad():
                prefill_output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            prefill_end_event.record()

            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
            decode_start_event.record()
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            decode_end_event.record()
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0

            resp = processor.batch_decode(
                out_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if "assistant\n" in resp:
                resp = resp.split("assistant\n")[-1].strip()
            
            output_tokens = out_ids.shape[1] - inputs.input_ids.shape[1]
            predicted_choice = extract_answer_choice(resp)

            is_correct = predicted_choice.upper() == ground_truth_choice.upper()
            accuracy = 1.0 if is_correct else 0.0
            total_accuracy += accuracy
            processed_samples += 1
            task_type_total[task_type] += 1
            if is_correct:
                task_type_correct[task_type] += 1

            timing_stats.add_record(
                prefill_time=prefill_time,
                decode_time=decode_time,
                output_tokens=output_tokens,
                input_tokens=input_token_length,
                audio_duration=sample.get("duration", 0),
                task_type=task_type
            )

            result = {
                "id": sample.get("id", f"sample_{idx}"),
                "audio_path": audio_path_for_inference,
                "task_type": task_type,
                "question": question,
                "choices": {
                    "A": choice_a,
                    "B": choice_b,
                    "C": choice_c,
                    "D": choice_d
                },
                "ground_truth_choice": ground_truth_choice,
                "predicted_choice": predicted_choice,
                "raw_response": resp,
                "is_correct": is_correct,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "audio_duration": sample.get("duration", 0)
            }
            results.append(result)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            current_avg_acc = total_accuracy / processed_samples
            progress_bar.set_description(f"DESED Evaluation - Accuracy: {current_avg_acc:.3f}")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            
            result = {
                "id": sample.get("id", f"sample_{idx}"),
                "audio_path": sample.get("path", ""),
                "task_type": sample.get("task_type", "unknown"),
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "ground_truth_choice": sample.get("answer_gt", ""),
                "predicted_choice": "error",
                "raw_response": f"ERROR: {str(e)}",
                "is_correct": False,
                "input_tokens": 0,
                "audio_tokens": 0,
                "output_tokens": 0,
                "prefill_time": 0.0,
                "decode_time": 0.0,
                "total_time": 0.0,
                "audio_duration": sample.get("duration", 0)
            }
            results.append(result)
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            continue

    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0

    task_type_accuracies = {}
    for task_type in task_type_stats.keys():
        if task_type_total[task_type] > 0:
            task_type_accuracies[task_type] = task_type_correct[task_type] / task_type_total[task_type]

    y_true = [sample["ground_truth_choice"] for sample in results if sample["predicted_choice"] != "error"]
    y_pred = [sample["predicted_choice"] for sample in results if sample["predicted_choice"] != "error"]
    
    detailed_metrics = calculate_desed_metrics(y_true, y_pred)

    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "task_type_stats": dict(task_type_stats),
        "task_type_accuracies": task_type_accuracies,
        "task_type_correct": dict(task_type_correct),
        "task_type_total": dict(task_type_total),
        "sklearn_metrics": detailed_metrics,
        "config": {
            "gpu_id": gpu_id,
            "model_path": args.model_path,
            "sparse_enabled": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            "sample_limit": sample_limit,
            "task_json_file": qa_json_file,
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
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== DESED Sound Event Detection Evaluation Results Summary (Qwen2.5-Omni + DART) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Overall accuracy: {final_accuracy:.3f}")
    print(f"Number of task types: {len(task_type_stats)}")
    
    print(f"\nAccuracy by task type:")
    for task_type, acc in task_type_accuracies.items():
        correct_num = task_type_correct[task_type]
        total_num = task_type_total[task_type]
        print(f"  {task_type}: {acc:.3f} ({correct_num}/{total_num})")
    
    metrics = detailed_metrics
    print(f"\n=== Detailed Evaluation Metrics (sklearn) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro Average): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted Average): {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro Average): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro Average): {metrics['recall_macro']:.4f}")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\n=== Inference Timing Statistics ===")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds (first {timing_sample_count} samples, excluding first one)")
    print(f"Average prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    
    print(f"\n=== Detailed Classification Report ===")
    print(metrics['classification_report'])
    
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()