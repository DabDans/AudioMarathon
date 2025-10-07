import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm
from collections import defaultdict
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
import librosa

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attn_implementation')
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    return parser.parse_args()

def configure_DART_for_aero1(model, args, audio_token_start=None, audio_token_length=None):
    """Configure DART for Aero-1 model - using audio related config"""
    if args.sparse and hasattr(model.config, 'audio_layer_idx'):
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = 1.0 - args.reduction_ratio
        
        if audio_token_start is not None:
            model.config.audio_token_start = audio_token_start
        if audio_token_length is not None:
            model.config.audio_token_num = audio_token_length
            
        print(f"Configure Aero-1 DART: layer={args.pruned_layer}, prune_ratio={1.0-args.reduction_ratio:.3f}")
        print(f"Audio token config: start={audio_token_start}, length={audio_token_length}")
    else:
        if not hasattr(model.config, 'audio_layer_idx'):
            print("Warning: Aero-1 model does not support audio pruning config")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

class DESEDTimingStats:
    """Track timing stats for DESED sound event detection task inference"""
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration=None, task_type=None):
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

class CudaEventTimingStats:
    """CUDA Event batch timing stats class"""
    
    def __init__(self):
        self.timing_records = []
        self.prefill_times = []
        self.decode_times = []
        self.total_times = []
    
    def add_timing_record(self, prefill_time, decode_time, total_time):
        self.prefill_times.append(prefill_time)
        self.decode_times.append(decode_time)
        self.total_times.append(total_time)
        
        self.timing_records.append({
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time
        })
    
    def get_time_statistics(self, times_list, name=""):
        if not times_list:
            return {}
        
        stats = {
            f"{name}_avg": sum(times_list) / len(times_list),
            f"{name}_count": len(times_list)
        }
        return stats
    
    def get_full_statistics(self):
        stats = {}
        stats.update(self.get_time_statistics(self.prefill_times, "prefill"))
        stats.update(self.get_time_statistics(self.decode_times, "decode"))
        stats.update(self.get_time_statistics(self.total_times, "total"))
        return stats
    
    def print_statistics(self):
        if not self.timing_records:
            print("No timing stats")
            return
        
        print("\n=== CUDA Event Timing Stats ===")
        
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill timing:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode timing:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total timing:")
        print(f"  Average: {total_stats['total_avg']:.6f}s")
        print(f"  Samples: {total_stats['total_count']}")

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference function using CUDA Event API for precise GPU timing
    """
    torch.cuda.synchronize()
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        event_start.record()
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
        event_prefill_end.record()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True
            )
        event_total_end.record()
        event_start.synchronize()
        event_prefill_end.synchronize()
        event_total_end.synchronize()
        prefill_time = event_start.elapsed_time(event_prefill_end) / 1000.0
        total_time = event_start.elapsed_time(event_total_end) / 1000.0
        decode_time = event_prefill_end.elapsed_time(event_total_end) / 1000.0
        if hasattr(out_ids, 'sequences'):
            tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
        else:
            tokens = out_ids[:, inputs['input_ids'].shape[1]:]
        output_tokens = len(tokens[0])
        response_text = processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return {
            'response_text': response_text,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'output_tokens': output_tokens,
            'generated_ids': out_ids,
            'tokens': tokens,
            'outputs': outputs,
            'tokens_per_second': output_tokens / decode_time if decode_time > 0 else 0
        }
    finally:
        pass

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio via librosa and split, adapted for Aero-1 model"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:
            audio_chunks = split_audio(audio)
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    Load data from new DESED task JSON file
    """
    dataset = []
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        return []
    print(f"Loading DESED task JSON: {json_file}")
    print(f"Audio base dir: {audio_base_dir}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    if not isinstance(data, dict) or 'tasks' not in data:
        print(f"Error: JSON format invalid, expected dict with 'tasks' field")
        return []
    tasks = data['tasks']
    print(f"Loaded {len(tasks)} tasks from JSON")
    task_type_stats = defaultdict(int)
    missing_files = 0
    for i, task in enumerate(tasks):
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task missing audio path: {task}")
            continue
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file not found: {full_audio_path}")
            continue
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"Cannot read audio file info {full_audio_path}: {e}")
            continue
        choices = task.get("choices", {})
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_type": task.get("task_type", "unknown"),
            "question": task.get("question", ""),
            "choice_a": choices.get("A", ""),
            "choice_b": choices.get("B", ""),
            "choice_c": choices.get("C", ""),
            "choice_d": choices.get("D", ""),
            "answer_gt": task.get("answer_gt", ""),
            "original_events": task.get("all_events", []),
            "all_events": task.get("all_events", []),
            "primary_event": task.get("primary_event", ""),
            "correct_event": task.get("correct_event", ""),
            "path_extracted_event": task.get("path_extracted_event", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"qa_task_{task.get('uniq_id', i)}"
        }
        dataset.append(item)
        task_type_stats[item["task_type"]] += 1
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files not found")
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type stats: {dict(task_type_stats)}")
    return dataset

def create_aero_qa_prompt(doc):
    """Generate QA format prompt for Aero-1 model"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    prompt_text = f"""Listen to the audio carefully and answer the following question.

{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please select the correct answer and respond with only the letter (A, B, C, or D)."""
    return prompt_text

def extract_answer_choice(response):
    """Extract answer choice (A, B, C, D) from model response"""
    if not response:
        return ""
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

def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate accuracy for sound event detection task"""
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
        print(f"Error evaluating sound event detection accuracy: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def calculate_desed_metrics(predictions, ground_truths):
    """Calculate F1 score etc for DESED sound event detection task"""
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) if p in ['A', 'B', 'C', 'D'] and t in ['A', 'B', 'C', 'D']]
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
        'total_samples': len(predictions)
    }

def generate_sklearn_desed_dart_evaluation_report(y_true, y_pred, task_types=None, labels=None):
    """
    Generate detailed evaluation report for DESED sound event detection task (DART version) using sklearn
    
    Args:
        y_true: true labels list (e.g. ['A', 'B', 'C', 'D'])
        y_pred: predicted labels list (e.g. ['A', 'B', 'C', 'D'])
        task_types: list of task types for per-type analysis
        labels: label names list for classification report
    
    Returns:
        dict: dictionary with various evaluation metrics
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"error": "Invalid input data for evaluation"}
    valid_indices = []
    valid_y_true = []
    valid_y_pred = []
    valid_label_set = {'A', 'B', 'C', 'D'}
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in valid_label_set and pred_label in valid_label_set:
            valid_indices.append(i)
            valid_y_true.append(true_label)
            valid_y_pred.append(pred_label)
    if not valid_y_true:
        return {"error": "No valid labels for evaluation"}
    accuracy = accuracy_score(valid_y_true, valid_y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='micro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='weighted', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(valid_y_true, valid_y_pred, average=None, labels=['A', 'B', 'C', 'D'], zero_division=0)
    if labels is None:
        target_names = ['A', 'B', 'C', 'D']
    else:
        target_names = labels
    classification_rep = classification_report(valid_y_true, valid_y_pred, target_names=target_names, output_dict=True, zero_division=0)
    evaluation_report = {
        "overall_metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted
        },
        "per_choice_metrics": {},
        "classification_report": classification_rep,
        "sample_statistics": {
            "total_samples": len(y_true),
            "valid_samples": len(valid_y_true),
            "invalid_samples": len(y_true) - len(valid_y_true),
            "correct_predictions": sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == p),
            "unique_true_labels": list(set(valid_y_true)),
            "unique_pred_labels": list(set(valid_y_pred))
        }
    }
    choice_labels = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choice_labels):
        if i < len(precision_per_class):
            evaluation_report["per_choice_metrics"][choice] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1_score": f1_per_class[i],
                "support": int(support_per_class[i]) if i < len(support_per_class) else 0
            }
    if task_types and len(task_types) == len(y_true):
        task_type_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, task_type in enumerate(task_types):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                task_type_analysis[task_type]["y_true"].append(valid_y_true[valid_index])
                task_type_analysis[task_type]["y_pred"].append(valid_y_pred[valid_index])
        task_summaries = {}
        for task_type, data in task_type_analysis.items():
            if len(data["y_true"]) > 0:
                task_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    task_precision, task_recall, task_f1, _ = precision_recall_fscore_support(data["y_true"], data["y_pred"], average='macro', zero_division=0)
                except:
                    task_precision = task_recall = task_f1 = 0.0
                task_summaries[task_type] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": task_accuracy,
                    "precision_macro": task_precision,
                    "recall_macro": task_recall,
                    "f1_macro": task_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        evaluation_report["task_type_analysis"] = task_summaries
    return evaluation_report

def main():
    import random
    random.seed(42)
    args = parse_arguments()
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    print(f"Using GPU ID: {gpu_id}")
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0.0))
    prune_method = os.environ.get("PRUNE_METHOD", "fast_v")
    use_random = (prune_method == "random")
    use_frame = (prune_method == "frame")
    if use_random == False and use_frame == False:
        prune_method = "fast_v"
    if prune_ratio == 0:
        method_is = "base"
    else:
        method_is = prune_method
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    if debug_mode:
        print("Debug mode enabled - detailed output will be shown")
    qa_json_file = "/data/to/your/eval_DESED_dataset_concatenated_audio_desed_sound_event_detection_task.json"
    audio_base_dir = "/data/to/your/eval_DESED_dataset_concatenated_audio"
    print(f"QA JSON file: {qa_json_file}")
    print(f"Audio base dir: {audio_base_dir}")
    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/DESED_Aero1_DART_Results')
    os.makedirs(result_dir, exist_ok=True)
    output_file = f'{result_dir}/DESED_Aero1_DART_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    timing_output_file = f'{result_dir}/DESED_Aero1_DART_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    cuda_event_output_file = f'{result_dir}/DESED_Aero1_DART_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")
    print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")
    _AUDIO_SPECIAL_TOKEN_ID = 151667
    timing_stats = DESEDTimingStats()
    cuda_event_stats = CudaEventTimingStats()
    print(f"\n=== DESED Sound Event Detection Evaluation Config (Aero-1 + DART) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer idx: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Task JSON file: {qa_json_file}")
    print(f"Audio base dir: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    if debug_mode:
        print("Debug mode: enabled")
    print("=" * 50)
    print("Loading Aero-1 model...")
    sys.stdout.flush()
    model_name = "lmms-lab/Aero-1-Audio-1.5B"
    processor = AutoProcessor.from_pretrained(
        model_name, 
        revision="main", 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    print("Aero-1 model loaded successfully")
    sys.stdout.flush()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {len(samples)} samples")
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, processing {len(samples)} samples")
    task_type_stats = defaultdict(int)
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_stats[task_type] += 1
    print(f"Task type stats: {dict(task_type_stats)}")
    results = []
    total_accuracy = 0
    processed_samples = 0
    all_predictions = []
    all_ground_truths = []
    all_task_types = []
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, simplified progress display")
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout,
        'mininterval': 0.1,
        'maxinterval': 1.0,
        'disable': False,
        'leave': True,
        'position': 0
    }
    if is_screen_env:
        tqdm_kwargs['mininterval'] = 0.05
        tqdm_kwargs['maxinterval'] = 0.5
    print(f"Start evaluating {len(samples)} samples...")
    allocated, reserved = get_gpu_memory_usage()
    print(f"After model loading GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="DESED Sound Event Detection Evaluation (Aero-1+DART)", **tqdm_kwargs)
    for idx, sample in progress_bar:
        try:
            audio_path = sample["audio"]["path"]
            audio_chunks, sample_rate = prepare_audio_for_processor(audio_path)
            ground_truth_choice = sample.get("answer_gt", "")
            task_type = sample.get("task_type", "unknown")
            prompt_text = create_aero_qa_prompt(sample)
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            for chunk in audio_chunks:
                messages[0]["content"].append({
                    "type": "audio",
                    "audio": "placeholder",
                })
            messages[0]["content"].append({
                "type": "text",
                "text": prompt_text
            })
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                audios=audio_chunks,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to("cuda")
            audio_token_length = 0
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end - audio_token_start + 1
                configure_DART_for_aero1(model, args, audio_token_start, audio_token_length)
                if hasattr(model.config, 'audio_layer_idx'):
                    model.config.random = use_random
                    model.config.frame = use_frame
            if debug_mode:
                print(f"Processing audio: {os.path.basename(audio_path)}")
                print(f"Task type: {task_type}")
                print(f"Audio chunks: {len(audio_chunks)}")
                print(f"Total audio length: {sum(len(chunk) for chunk in audio_chunks)}, sample rate: {sample_rate}")
                print(f"Question: {sample.get('question', '')}")
                print(f"Correct answer: {ground_truth_choice}")
                print(f"Input IDs shape: {inputs.input_ids.shape}")
                print(f"Estimated audio token length: {audio_token_length}")
                sys.stdout.flush()
            result = cuda_timing_inference(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=10
            )
            output = result['response_text']
            prefill_time = result['prefill_time']
            decode_time = result['decode_time']
            total_time = result['total_time']
            output_tokens = result['output_tokens']
            predicted_choice = extract_answer_choice(output)
            metrics = evaluate_qa_accuracy(predicted_choice, ground_truth_choice)
            accuracy = metrics["accuracy"]
            is_correct = metrics["is_correct"]
            all_predictions.append(predicted_choice)
            all_ground_truths.append(ground_truth_choice)
            all_task_types.append(task_type)
            total_accuracy += accuracy
            processed_samples += 1
            task_type_total[task_type] += 1
            if is_correct:
                task_type_correct[task_type] += 1
            current_avg_acc = total_accuracy / processed_samples
            if debug_mode:
                print(f"Model output: '{output}'")
                print(f"Inference time: total={total_time:.3f}s, prefill={prefill_time:.3f}s, decode={decode_time:.3f}s")
                print(f"Output tokens: {output_tokens}")
                print(f"Extracted answer: '{predicted_choice}'")
                print(f"Ground truth: '{ground_truth_choice}'")
                print(f"Answer correct: {is_correct}")
                print("=" * 50)
                sys.stdout.flush()
            update_interval = 50 if is_screen_env else 20
            sample_count = idx + 1
            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_type[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })
                if is_screen_env:
                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), accuracy: {current_avg_acc:.3f}")
            else:
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_type[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })
            results.append({
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "filename": sample.get("filename", ""),
                "task_type": task_type,
                "path": sample.get("path", ""),
                "duration":