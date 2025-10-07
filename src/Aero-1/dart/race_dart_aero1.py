import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
import traceback
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
import librosa

# Disable transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Audio special token ID (used by Aero-1)
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
    parser.add_argument('--min_audio_duration', type=float, default=120.0, help='Minimum audio duration in seconds (default: 120s = 2 min)')
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
            "pivot_image_token": args.pivot_audio_token,
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            "text_length": 1,
        }
        model.config.DART_config = DART_config
        print(f"DART configuration enabled: pruned layers={args.pruned_layer}, retained ratio={args.reduction_ratio}")
    else:
        model.config.DART_config = None
        print("DART configuration disabled")

# Get GPU ID
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

# Sample limit (if provided)
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

class RaceTimingStats:
    """Track inference timing statistics for RACE task using CUDA Event for precision"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration):
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
            "audio_duration": audio_duration
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get overall statistic summary"""
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

class CudaEventTimingStats:
    """CUDA Event batch timing statistics class"""
    
    def __init__(self):
        self.timing_records = []
        self.prefill_times = []
        self.decode_times = []
        self.total_times = []
    
    def add_timing_record(self, prefill_time, decode_time, total_time):
        """Add a timing measurement record"""
        self.prefill_times.append(prefill_time)
        self.decode_times.append(decode_time)
        self.total_times.append(total_time)
        
        self.timing_records.append({
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time
        })
    
    def get_time_statistics(self, times_list, name=""):
        """Calculate time statistics (average only)"""
        if not times_list:
            return {}
        
        stats = {
            f"{name}_avg": sum(times_list) / len(times_list),
            f"{name}_count": len(times_list)
        }
        return stats
    
    def get_full_statistics(self):
        """Get full timing statistics"""
        stats = {}
        stats.update(self.get_time_statistics(self.prefill_times, "prefill"))
        stats.update(self.get_time_statistics(self.decode_times, "decode"))
        stats.update(self.get_time_statistics(self.total_times, "total"))
        return stats
    
    def print_statistics(self):
        """Print timing statistics (average only)"""
        if not self.timing_records:
            print("No timing statistics data")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        
        # Prefill statistics
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill timing statistics:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        
        # Decode statistics
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode timing statistics:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        
        # Total statistics
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total timing statistics:")
        print(f"  Average: {total_stats['total_avg']:.6f}s")
        print(f"  Sample count: {total_stats['total_count']}")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sampling rate"""
    if original_sr == target_sr:
        return audio_array
    
    # Use librosa for resampling
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30-second chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def get_audio_duration(audio_path, target_sr=16000):
    """Get audio duration (seconds)"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Failed to get audio duration: {audio_path}, Error: {e}")
        return 0

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load and split audio using librosa, compatible with Aero-1 official example"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:  # 30 seconds @ 16kHz
            audio_chunks = split_audio(audio)
            print(f"Audio length {len(audio)} exceeds 30 seconds limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def clean_text_response(response):
    """Clean model response for RACE task, keep only the first character as option label"""
    if not response:
        return ""
    resp = response.strip().upper()
    for ch in resp:
        if ch in ['A', 'B', 'C', 'D']:
            return ch
    return resp.split()[0] if resp.split() else ""

def create_race_prompt_aero1(question, options):
    """Create Aero-1 format RACE task prompt"""
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options += f"{letter}. {opt}\n"
    
    instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    return {
        "instruction": instruction,
        "question": question,
        "formatted_options": formatted_options.strip(),
        "format_text": format_text
    }

def cuda_timing_inference(model, processor, inputs, max_new_tokens=3):
    """
    Inference function with precise GPU timing using CUDA Event API
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

def filter_samples_by_duration(benchmark, data_path_root, min_duration=120.0):
    """
    Filter samples with audio length >= specified duration
    
    Args:
        benchmark: original dataset sample list
        data_path_root: data root directory
        min_duration: minimum audio duration (seconds), default 120s (2 minutes)
    
    Returns:
        filtered_samples: filtered sample list
        duration_stats: duration statistics
    """
    print(f"\n=== Start filtering samples with audio length >= {min_duration/60:.1f} minutes ===")
    filtered_samples = []
    duration_stats = {
        "total_samples": len(benchmark),
        "filtered_samples": 0,
        "duration_distribution": {
            "< 1min": 0,
            "1-2min": 0,
            "2-3min": 0,
            "3-5min": 0,
            "> 5min": 0
        },
        "audio_durations": [],
        "filtered_durations": [],
        "skipped_files": []
    }
    print(f"Checking audio duration for {len(benchmark)} samples...")
    for idx, sample in enumerate(tqdm(benchmark, desc="Checking audio duration")):
        audio_rel = sample["audio_path"]
        audio_full = os.path.join(data_path_root, audio_rel)
        if not os.path.exists(audio_full):
            print(f"Warning: Audio file not found: {audio_full}")
            duration_stats["skipped_files"].append(audio_rel)
            continue
        duration = get_audio_duration(audio_full)
        duration_stats["audio_durations"].append({
            "audio_path": audio_rel,
            "duration": duration
        })
        if duration < 60:
            duration_stats["duration_distribution"]["< 1min"] += 1
        elif duration < 120:
            duration_stats["duration_distribution"]["1-2min"] += 1
        elif duration < 180:
            duration_stats["duration_distribution"]["2-3min"] += 1
        elif duration < 300:
            duration_stats["duration_distribution"]["3-5min"] += 1
        else:
            duration_stats["duration_distribution"]["> 5min"] += 1
        if duration >= min_duration:
            sample_copy = sample.copy()
            sample_copy["audio_duration"] = duration
            filtered_samples.append(sample_copy)
            duration_stats["filtered_durations"].append({
                "audio_path": audio_rel,
                "duration": duration
            })
    duration_stats["filtered_samples"] = len(filtered_samples)
    print(f"\n=== Audio duration filtering statistics ===")
    print(f"Original sample count: {duration_stats['total_samples']}")
    print(f"Filtered sample count: {duration_stats['filtered_samples']}")
    print(f"Filtering ratio: {duration_stats['filtered_samples']/duration_stats['total_samples']*100:.2f}%")
    print(f"Skipped files count: {len(duration_stats['skipped_files'])}")
    print(f"\nDuration distribution:")
    for range_name, count in duration_stats["duration_distribution"].items():
        percentage = count / duration_stats['total_samples'] * 100 if duration_stats['total_samples'] > 0 else 0
        print(f"  {range_name}: {count} samples ({percentage:.1f}%)")
    if duration_stats["filtered_durations"]:
        durations = [item["duration"] for item in duration_stats["filtered_durations"]]
        print(f"\nFiltered samples audio duration statistics:")
        print(f"  Min duration: {min(durations):.2f}s ({min(durations)/60:.2f}min)")
        print(f"  Max duration: {max(durations):.2f}s ({max(durations)/60:.2f}min)")
        print(f"  Avg duration: {sum(durations)/len(durations):.2f}s ({sum(durations)/len(durations)/60:.2f}min)")
    return filtered_samples, duration_stats

def calculate_race_metrics(predictions, ground_truths):
    """Calculate F1 score and metrics for RACE task"""
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['A', 'B', 'C', 'D'] and t in ['A', 'B', 'C', 'D']]
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

def generate_sklearn_race_dart_evaluation_report(y_true, y_pred, difficulties=None, labels=None):
    """
    Generate detailed evaluation report for RACE reading comprehension task (DART version) using sklearn
    
    Args:
        y_true: list of true labels (e.g. ['A', 'B', 'C', 'D'])
        y_pred: list of predicted labels (e.g. ['A', 'B', 'C', 'D'])
        difficulties: list of difficulty levels for analysis
        labels: list of label names for classification report
    
    Returns:
        dict: dictionary containing various evaluation metrics
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
    if difficulties and len(difficulties) == len(y_true):
        difficulty_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, difficulty in enumerate(difficulties):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                difficulty_analysis[difficulty]["y_true"].append(valid_y_true[valid_index])
                difficulty_analysis[difficulty]["y_pred"].append(valid_y_pred[valid_index])
        difficulty_summaries = {}
        for difficulty, data in difficulty_analysis.items():
            if len(data["y_true"]) > 0:
                difficulty_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    difficulty_precision, difficulty_recall, difficulty_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    difficulty_precision = difficulty_recall = difficulty_f1 = 0.0
                difficulty_summaries[difficulty] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": difficulty_accuracy,
                    "precision_macro": difficulty_precision,
                    "recall_macro": difficulty_recall,
                    "f1_macro": difficulty_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        evaluation_report["difficulty_level_analysis"] = difficulty_summaries
    return evaluation_report

def main():
    random.seed(42)
    args = parse_arguments()
    data_path_root = os.environ.get("RACE_DATA_PATH", "/data/to/your/race_audio/path")
    results_dir_name = os.environ.get("RESULTS_DIR", "Race_Results")
    if not os.path.isabs(results_dir_name):
        result_dir = os.path.abspath(results_dir_name)
    else:
        result_dir = results_dir_name
    os.makedirs(result_dir, exist_ok=True)
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    print(f"\n=== RACE DART Evaluation Configuration (Aero-1, Long Audio Filtering) ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Minimum audio duration: {args.min_audio_duration}s ({args.min_audio_duration/60:.1f}min)")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    sparse_suffix = "_sparse" if args.sparse else "_base"
    duration_label = f"min{args.min_audio_duration/60:.0f}min"
    output_file = os.path.join(result_dir, f'race_aero1_results_dart{sparse_suffix}_{duration_label}.json')
    timing_output_file = os.path.join(result_dir, f'race_aero1_timing_stats_dart{sparse_suffix}_{duration_label}.json')
    cuda_event_output_file = os.path.join(result_dir, f'race_aero1_cuda_event_stats_dart{sparse_suffix}_{duration_label}.json')
    duration_stats_file = os.path.join(result_dir, f'race_aero1_duration_stats_{duration_label}.json')
    
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    print(f"CUDA Event statistics will be saved to: {cuda_event_output_file}")
    print(f"Duration statistics will be saved to: {duration_stats_file}")

    timing_stats = RaceTimingStats()
    cuda_event_stats = CudaEventTimingStats()

    model_path = args.model_path
    print(f"Loading Aero-1 model: {model_path}")
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
        
    generation_config = GenerationConfig.from_pretrained(model_path)
    configure_DART(model, args)
    print("DART configuration complete")

    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: Benchmark file not found {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        original_benchmark = json.load(f)

    print(f"Original dataset contains {len(original_benchmark)} samples")

    benchmark, duration_stats = filter_samples_by_duration(
        original_benchmark, 
        data_path_root, 
        min_duration=args.min_audio_duration
    )
    
    with open(duration_stats_file, "w", encoding="utf-8") as f:
        json.dump(duration_stats, f, ensure_ascii=False, indent=2)
    print(f"Duration statistics saved to: {duration_stats_file}")

    if len(benchmark) == 0:
        print(f"Error: No samples found with audio duration >= {args.min_audio_duration}s!")
        return

    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f"Filtered sample count limited to: {sample_limit}")

    results = []
    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0
    all_predictions = []
    all_ground_truths = []
    all_difficulties = []

    print(f"Starting evaluation of {len(benchmark)} long audio samples (>= {args.min_audio_duration/60:.1f}min)...")
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
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

    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc=f"RACE Evaluation (Aero-1+DART, >={args.min_audio_duration/60:.1f}min)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        prefill_time = 0
        decode_time = 0
        output_tokens = 0
        audio_token_length = 0
        pred_choice = ""
        is_correct = False
        try:
            audio_rel = sample["audio_path"]
            audio_full = os.path.join(data_path_root, audio_rel)
            if not os.path.exists(audio_full):
                print(f"Warning: Audio file not found: {audio_full}")
                continue
            actual_duration = sample.get("audio_duration", get_audio_duration(audio_full))
            if "high" in audio_rel:
                total_high += 1
                difficulty = "high"
            elif "middle" in audio_rel:
                total_middle += 1
                difficulty = "middle"
            else:
                difficulty = "unknown"
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            audio_chunks, sample_rate = prepare_audio_for_processor(audio_full)
            for chunk in audio_chunks:
                messages[0]["content"].append({
                    "type": "audio",
                    "audio": "placeholder",
                })
            prompt_data = create_race_prompt_aero1(sample['question'], sample['options'])
            messages[0]["content"].append({
                "type": "text",
                "text": f"{prompt_data['instruction']}\n\nQuestion: {prompt_data['question']}\n\nOptions:\n{prompt_data['formatted_options']}\n\n{prompt_data['format_text']}"
            })
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                audios=audio_chunks,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to("cuda")
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start_index = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end_index = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end_index - audio_token_start_index + 1
                if args.sparse:
                    model.config.DART_config['audio_token_start_index'] = audio_token_start_index
                    model.config.DART_config['audio_token_length'] = audio_token_length
                model.config.image_layer_idx = None
                model.config.audio_layer_idx = args.pruned_layer
                model.config.audio_token_num = audio_token_length
                model.config.audio_token_start = audio_token_start_index
                model.config.audio_prune_ratio = args.reduction_ratio
                model.config.random = False
                model.config.frame = False
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
                pred_choice = clean_text_response(response)
                correct_choice = sample["answer"]
                is_correct = pred_choice == correct_choice
                all_predictions.append(pred_choice)
                all_ground_truths.append(correct_choice)
                all_difficulties.append(difficulty)
                if is_correct:
                    correct_count += 1
                    if "high" in audio_rel:
                        correct_high += 1
                    elif "middle" in audio_rel:
                        correct_middle += 1
                input_tokens = inputs['input_ids'].shape[1]
                result = {
                    "id": sample.get("id", f"sample_{idx}"),
                    "question": sample["question"],
                    "options": sample["options"],
                    "correct_answer": correct_choice,
                    "predicted_answer": pred_choice,
                    "correct": is_correct,
                    "difficulty": difficulty,
                    "audio_path": sample["audio_path"],
                    "audio_duration": actual_duration,
                    "audio_chunks": len(audio_chunks),
                    "response_text": response,
                    "timing": {
                        "prefill_time": prefill_time,
                        "decode_time": decode_time,
                        "total_time": prefill_time + decode_time,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_sec": output_tokens/decode_time if decode_time > 0 else 0
                    }
                }
                results.append(result)
                timing_stats.add_record(
                    prefill_time,
                    decode_time,
                    output_tokens,
                    input_tokens,
                    audio_duration=actual_duration
                )
                cuda_event_stats.add_timing_record(prefill_time, decode_time, prefill_time + decode_time)
                current_acc = correct_count / (idx + 1) if idx >= 0 else 0
                progress_bar.set_postfix({
                    'Acc': f"{current_acc:.3f}",
                    'Ans': f"{pred_choice}/{correct_choice}",
                    'Duration': f"{actual_duration:.1f}s",
                    'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                })
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
                if (idx + 1) % 50 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            pred_choice = "ERROR"
            is_correct = False
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            actual_duration = sample.get("audio_duration", 0)
            difficulty = "high" if "high" in sample.get("audio_path", "") else "middle"
            all_predictions.append("")
            all_ground_truths.append(sample.get("answer", ""))
            all_difficulties.append(difficulty)
            results.append({
                "id": sample.get("id", f"sample_{idx}"),
                "question": sample["question"],
                "options": sample["options"],
                "correct_answer": sample["answer"],
                "predicted_answer": pred_choice,
                "correct": is_correct,
                "difficulty": difficulty,
                "audio_path": sample.get("audio_path", ""),
                "audio_duration": actual_duration,
                "audio_chunks": 1,
                "response_text": "ERROR",
                "timing": {
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time,
                    "input_tokens": 0,
                    "output_tokens": output_tokens,
                    "tokens_per_sec": 0
                }
            })
            current_acc = correct_count / (idx + 1) if idx >= 0 else 0
            continue

    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0
    all_predictions_for_metrics = [r["predicted_answer"] for r in results]
    all_ground_truths_for_metrics = [r["correct_answer"] for r in results]
    metrics = calculate_race_metrics(all_predictions_for_metrics, all_ground_truths_for_metrics)

    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn RACE DART evaluation report ===")
        print(f"Total samples: {len(all_predictions)} (pred), {len(all_ground_truths)} (truth)")
        sklearn_evaluation = generate_sklearn_race_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            difficulties=all_difficulties,
            labels=['A', 'B', 'C', 'D']
        )
        print(f"sklearn RACE DART evaluation report generated:")
        print(f"  Overall accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Macro F1: {sklearn_evaluation['overall_metrics']['f1_macro']:.4f}")
        print(f"  Micro F1: {sklearn_evaluation['overall_metrics']['f1_micro']:.4f}")
        print(f"  Weighted F1: {sklearn_evaluation['overall_metrics']['f1_weighted']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
    else:
        print("Warning: No valid predictions to generate sklearn evaluation report")
        sklearn_evaluation = {"error": "No valid predictions for evaluation"}

    audio_durations = [r["audio_duration"] for r in results if r["audio_duration"] > 0]
    avg_audio_duration = sum(audio_durations) / len(audio_durations) if audio_durations else 0
    min_audio_duration_actual = min(audio_durations) if audio_durations else 0
    max_audio_duration_actual = max(audio_durations) if audio_durations else 0

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
        "metrics": metrics,
        "sklearn_evaluation": sklearn_evaluation,
        "audio_duration_stats": {
            "min_duration_filter": args.min_audio_duration,
            "avg_duration": avg_audio_duration,
            "min_duration_actual": min