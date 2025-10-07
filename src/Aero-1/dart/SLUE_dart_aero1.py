import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
import re
import traceback
import subprocess
import tempfile
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import librosa

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

os.environ['PYTHONUNBUFFERED'] = '1'

_AUDIO_SPECIAL_TOKEN_ID = 151667

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/Aero-1-Audio-1.5B")
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attn_implementation')
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--image_token_start_index', type=int, default=None, help='image_token_start_index')
    parser.add_argument('--image_token_length', type=int, default=None, help='image_token_length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    parser.add_argument('--prune_method', type=str, default='fast_v', help='Pruning method: fast_v, random, frame')
    return parser.parse_args()

def configure_DART_for_Aero1(model, args, audio_token_start=None, audio_token_length=None):
    """Configure DART sparse attention mechanism for Aero-1"""
    if args.sparse:
        model.config.image_layer_idx = None
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_token_num = audio_token_length or args.audio_token_length
        model.config.audio_token_start = audio_token_start or args.audio_token_start_index
        model.config.audio_prune_ratio = args.reduction_ratio
        model.config.random = (args.prune_method == "random")
        model.config.frame = (args.prune_method == "frame")
        print(f"DART configuration enabled - Layer:{args.pruned_layer}, Ratio:{args.reduction_ratio}, Method:{args.prune_method}")
    else:
        model.config.image_layer_idx = None
        model.config.audio_layer_idx = None
        model.config.audio_token_num = None
        model.config.audio_token_start = None
        model.config.audio_prune_ratio = 0
        model.config.random = False
        model.config.frame = False

gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit is set to: {sample_limit}")

debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
if debug_mode:
    print("Debug mode enabled - verbose output will be shown")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

class SLUETimingStats:
    """Track SLUE inference timing stats using CUDA Event for precision"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
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
    
    def get_summary(self):
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
        summary = self.get_summary()
        data = {
            "summary": summary,
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

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
            print("No timing data available")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill timing stats:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode timing stats:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total timing stats:")
        print(f"  Average: {total_stats['total_avg']:.6f}s")
        print(f"  Sample count: {total_stats['total_count']}")

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sampling rate"""
    if original_sr == target_sr:
        return audio_array
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    audio_splits = []
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio using librosa and split, compatible with Aero-1 official example"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30s limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_slue_dataset(json_file, audio_base_dir):
    """
    Load SLUE task data from JSON file (robust version from SLUE_test.py)
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
        print(f"Error: Incorrect JSON file format, expected list format")
        return []
    print(f"Loaded {len(data)} tasks from JSON")
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0
    for i, task in enumerate(data):
        relative_path = task.get("path", "")
        if not relative_path:
            print(f"Warning: Task missing 'path' key, skipped: {task}")
            continue
        full_audio_path = os.path.join(audio_base_dir, relative_path)
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file not found: {full_audio_path}")
            continue
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
        except Exception:
            duration = 0
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "task_name": task.get("task_name", "unknown"),
            "dataset_name": task.get("dataset_name", "unknown"),
            "question": task.get("question", ""),
            "choice_a": task.get("choice_a", ""),
            "choice_b": task.get("choice_b", ""),
            "choice_c": task.get("choice_c", ""),
            "choice_d": task.get("choice_d", ""),
            "answer_gt": task.get("answer_gt", ""),
            "duration": duration,
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }
        dataset.append(item)
        task_type_stats[item["task_name"]] += 1
        dataset_stats[item["dataset_name"]] += 1
    if missing_files > 5:
        print(f"Warning: {missing_files} audio files not found")
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type stats: {dict(task_type_stats)}")
    print(f"Dataset stats: {dict(dataset_stats)}")
    return dataset

def create_aero1_slue_prompt(doc):
    """Create SLUE formatted task prompt for Aero-1 model"""
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
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]
    return messages, prompt_text

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

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference with precise GPU timing using CUDA Event API
    Follows NVIDIA CUDA best practices for timing:
    1. Record start event on GPU stream
    2. Run actual computation task on GPU stream  
    3. Record end event on GPU stream
    4. Use event sync to wait for completion
    """
    torch.cuda.synchronize()
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    try:
        event_start.record()
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, 
                           output_hidden_states=False, return_dict=True)
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

def evaluate_slue_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate SLUE task accuracy, return detailed dictionary"""
    try:
        pred = predicted_choice.strip().upper() if predicted_choice else ""
        gt = ground_truth_choice.strip().upper() if ground_truth_choice else ""
        is_correct = pred == gt
        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "predicted_choice": pred,
            "ground_truth_choice": gt,
            "is_correct": is_correct
        }
    except Exception:
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def generate_sklearn_slue_dart_evaluation_report(y_true, y_pred, task_types=None, dataset_names=None, labels=None):
    """
    Generate detailed evaluation report using sklearn for SLUE NER task (DART version)
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
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average=None, labels=['A', 'B', 'C', 'D'], zero_division=0
    )
    if labels is None:
        target_names = ['A', 'B', 'C', 'D']
    else:
        target_names = labels
    classification_rep = classification_report(
        valid_y_true, valid_y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
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
        task_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, task_type in enumerate(task_types):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                task_analysis[task_type]["y_true"].append(valid_y_true[valid_index])
                task_analysis[task_type]["y_pred"].append(valid_y_pred[valid_index])
        task_summaries = {}
        for task_type, data in task_analysis.items():
            if len(data["y_true"]) > 0:
                task_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    task_precision, task_recall, task_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
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
    if dataset_names and len(dataset_names) == len(y_true):
        dataset_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, dataset_name in enumerate(dataset_names):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                dataset_analysis[dataset_name]["y_true"].append(valid_y_true[valid_index])
                dataset_analysis[dataset_name]["y_pred"].append(valid_y_pred[valid_index])
        dataset_summaries = {}
        for dataset_name, data in dataset_analysis.items():
            if len(data["y_true"]) > 0:
                dataset_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    dataset_precision, dataset_recall, dataset_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    dataset_precision = dataset_recall = dataset_f1 = 0.0
                dataset_summaries[dataset_name] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": dataset_accuracy,
                    "precision_macro": dataset_precision,
                    "recall_macro": dataset_recall,
                    "f1_macro": dataset_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        evaluation_report["dataset_analysis"] = dataset_summaries
    return evaluation_report

def main():
    random.seed(42)
    args = parse_arguments()

    # Path configs
    slue_json_file = "/data/to/your/eval/SLUE/merged_audio_data.json"
    audio_base_dir = "/data/to/your/eval/SLUE"
    print(f"SLUE JSON file: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/SLUE_Results')
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n=== SLUE DART Aero-1 Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Prune method: {args.prune_method}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    sparse_suffix = f"_sparse_{args.prune_method}" if args.sparse else "_base"
    output_file = f'{result_dir}/slue_aero1_results_dart{sparse_suffix}.json'
    timing_output_file = f'{result_dir}/slue_aero1_timing_stats_dart{sparse_suffix}.json'
    cuda_event_output_file = f'{result_dir}/slue_aero1_cuda_event_stats_dart{sparse_suffix}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")
    print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")

    timing_stats = SLUETimingStats()
    cuda_event_stats = CudaEventTimingStats()

    print("Loading Aero-1 model...")
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision="main",
        trust_remote_code=True
    )
    print("Successfully loaded Aero-1 processor")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    print("Successfully loaded Aero-1 model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    correct_count = 0
    total_count = 0
    all_predictions = []
    all_ground_truths = []
    all_task_types = []
    all_dataset_names = []
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Screen/non-interactive env detected, using simplified progress display")
        sys.stdout.flush()
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    with tqdm(enumerate(samples), total=len(samples), desc="SLUE Evaluation (Aero-1)", **tqdm_kwargs) as pbar:
        for idx, sample in pbar:
            try:
                audio_path = sample["path"]
                audio_chunks, sample_rate = prepare_audio_for_processor(audio_path)
                if audio_chunks is None:
                    continue
                messages, prompt_text = create_aero1_slue_prompt(sample)
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
                ).to(device)
                audio_token_start = None
                audio_token_length = 0
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    rev_ids = token_ids[::-1]
                    audio_token_end = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    configure_DART_for_Aero1(model, args, audio_token_start, audio_token_length)
                    if debug_mode:
                        print(f"Audio token start: {audio_token_start}")
                        print(f"Audio token length: {audio_token_length}")
                result = cuda_timing_inference(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=10
                )
                response = result['response_text']
                prefill_time = result['prefill_time']
                decode_time = result['decode_time']
                total_time = result['total_time']
                output_tokens = result['output_tokens']
                input_tokens = inputs['input_ids'].shape[1]
                predicted_choice = extract_answer_choice(response)
                ground_truth_choice = sample["answer_gt"]
                all_predictions.append(predicted_choice)
                all_ground_truths.append(ground_truth_choice)
                all_task_types.append(sample.get("task_name", "unknown"))
                all_dataset_names.append(sample.get("dataset_name", "unknown"))
                metrics = evaluate_slue_accuracy(predicted_choice, ground_truth_choice)
                is_correct = metrics["is_correct"]
                if is_correct:
                    correct_count += 1
                total_count += 1
                result_data = {
                    "id": sample["id"],
                    "audio_path": sample["path"],
                    "question": sample["question"],
                    "predicted_choice": predicted_choice,
                    "ground_truth_choice": ground_truth_choice,
                    "correct": is_correct,
                    "task_name": sample.get("task_name", "unknown"),
                    "dataset_name": sample.get("dataset_name", "unknown"),
                    "response_text": response,
                    "audio_chunks": len(audio_chunks),
                    "audio_token_length": audio_token_length,
                    "timing": {
                        "prefill_time": prefill_time, "decode_time": decode_time, "total_time": total_time,
                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                        "tokens_per_sec": output_tokens/decode_time if decode_time > 0 else 0
                    }
                }
                results.append(result_data)
                timing_stats.add_record(
                    prefill_time, decode_time, output_tokens, input_tokens,
                    audio_duration=sample.get("duration", 0),
                    task_type=sample.get("task_name", "unknown")
                )
                cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)
                current_acc = correct_count / total_count if total_count > 0 else 0
                pbar.set_postfix({'Acc': f"{current_acc:.3f}", 'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"})
                if 'inputs' in locals():
                    del inputs
                if 'audio_chunks' in locals():
                    del audio_chunks
                if 'result' in locals():
                    del result
                torch.cuda.empty_cache()
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                if debug_mode:
                    traceback.print_exc()
                all_predictions.append("")
                all_ground_truths.append(sample.get("answer_gt", ""))
                all_task_types.append(sample.get("task_name", "unknown"))
                all_dataset_names.append(sample.get("dataset_name", "unknown"))
                continue
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    timing_summary = timing_stats.get_summary()
    class_labels = ['A', 'B', 'C', 'D']
    f1 = f1_score(all_ground_truths, all_predictions, labels=class_labels, average='macro', zero_division=0)
    precision = precision_score(all_ground_truths, all_predictions, labels=class_labels, average='macro', zero_division=0)
    recall = recall_score(all_ground_truths, all_predictions, labels=class_labels, average='macro', zero_division=0)
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn SLUE DART evaluation report ===")
        print(f"Total samples: {len(all_predictions)} (predictions), {len(all_ground_truths)} (ground truth)")
        sklearn_evaluation = generate_sklearn_slue_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            task_types=all_task_types,
            dataset_names=all_dataset_names,
            labels=['A', 'B', 'C', 'D']
        )
        print(f"sklearn SLUE DART evaluation report generated:")
        print(f"  Overall accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  F1 macro: {sklearn_evaluation['overall_metrics']['f1_macro']:.4f}")
        print(f"  F1 micro: {sklearn_evaluation['overall_metrics']['f1_micro']:.4f}")
        print(f"  F1 weighted: {sklearn_evaluation['overall_metrics']['f1_weighted']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
    else:
        print("Warning: No valid predictions, unable to generate sklearn evaluation report")
        sklearn_evaluation = {"error": "No valid predictions for evaluation"}
    summary = {
        "model_name": "Aero-1-Audio-1.5B",
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": final_accuracy,
        "f1_score_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "task_type_stats": dict(task_type_stats),
        "dataset_stats": dict(dataset_stats),
        "sklearn_evaluation": sklearn_evaluation,
        "config": {
            "gpu_id": gpu_id, "sparse": args.sparse, "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio, "prune_method": args.prune_method, "sample_limit": sample_limit
        },
        "timing": timing_summary
    }
    final_results = {"summary": summary, "samples": results}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing stats saved to: {timing_output_file}")
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)
    print(f"CUDA Event stats saved to: {cuda_event_output_file}")
    print(f"\n=== SLUE DART Aero-1 Evaluation Results ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Processed samples: {total_count}")
    print(f"Overall accuracy: {final_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    if "sklearn_evaluation" in summary and "error" not in summary["sklearn_evaluation"]:
        sklearn_metrics = summary["sklearn_evaluation"]["overall_metrics"]
        print(f"\n=== Sklearn Evaluation Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision - Macro: {sklearn_metrics['precision_macro']:.4f}")
        print(f"Recall - Macro: {sklearn_metrics['recall_macro']:.4f}")
        print(f"F1 score - Macro: {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 score - Micro: {sklearn_metrics['f1_micro']:.4f}")
        print(f"F1 score - Weighted: {sklearn_metrics['f1_weighted']:.4f}")
        print(f"\nPer choice detailed metrics:")
        per_choice_metrics = summary["sklearn_evaluation"]["per_choice_metrics"]
        for choice in sorted(per_choice_metrics.keys()):
            metrics_detail = per_choice_metrics[choice]
            print(f"  Choice {choice}: Precision={metrics_detail['precision']:.