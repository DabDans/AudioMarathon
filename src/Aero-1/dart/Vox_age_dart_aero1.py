import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
import gc
import io
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np
import librosa
import traceback
import glob
from collections import defaultdict

# Disable warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Disable transformers warnings
from transformers import logging
logging.set_verbosity_error()

# Set random seed
random.seed(42)

# Audio special token ID (Aero-1 usage)
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
    else:
        model.config.DART_config = None

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30-second chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Use librosa to load and split audio, compatible with Aero-1 official example"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def calculate_metrics(predictions, ground_truths):
    """Calculate classification metrics: accuracy, precision, recall, F1 score"""
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

def generate_sklearn_voxceleb_age_dart_evaluation_report(y_true, y_pred, speaker_ids=None, labels=None):
    """
    Generate detailed evaluation report for VoxCeleb age classification task (DART version) using sklearn
    
    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        speaker_ids: List of speaker IDs for speaker-based analysis
        labels: List of label names for classification report
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"error": "Invalid input data for evaluation"}
    valid_indices = []
    valid_y_true = []
    valid_y_pred = []
    valid_age_groups = {
        'Young Adult (18-30)', 'Early Career (31-40)', 
        'Mid Career (41-50)', 'Senior (51-70)', 'Elderly (71+)'
    }
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if (true_label and pred_label and 
            true_label in valid_age_groups and 
            pred_label in valid_age_groups and
            pred_label != "error"):
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
    unique_labels = sorted(list(set(valid_y_true + valid_y_pred)))
    try:
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            valid_y_true, valid_y_pred, average=None, labels=unique_labels, zero_division=0
        )
    except Exception as e:
        print(f"Error calculating per-class metrics: {e}")
        precision_per_class = recall_per_class = f1_per_class = support_per_class = []
    if labels is None:
        target_names = unique_labels
    else:
        target_names = labels
    try:
        classification_rep = classification_report(
            valid_y_true, valid_y_pred, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        print(f"Error generating classification report: {e}")
        classification_rep = {}
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
        "per_age_group_metrics": {},
        "classification_report": classification_rep,
        "sample_statistics": {
            "total_samples": len(y_true),
            "valid_samples": len(valid_y_true),
            "invalid_samples": len(y_true) - len(valid_y_true),
            "correct_predictions": sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == p),
            "unique_true_labels": list(set(valid_y_true)),
            "unique_pred_labels": list(set(valid_y_pred)),
            "age_group_distribution": {}
        }
    }
    from collections import Counter
    true_label_counts = Counter(valid_y_true)
    for age_group, count in true_label_counts.items():
        evaluation_report["sample_statistics"]["age_group_distribution"][age_group] = count
    for i, age_group in enumerate(unique_labels):
        if i < len(precision_per_class):
            evaluation_report["per_age_group_metrics"][age_group] = {
                "precision": precision_per_class[i] if i < len(precision_per_class) else 0.0,
                "recall": recall_per_class[i] if i < len(recall_per_class) else 0.0,
                "f1_score": f1_per_class[i] if i < len(f1_per_class) else 0.0,
                "support": int(support_per_class[i]) if i < len(support_per_class) else 0
            }
    if speaker_ids and len(speaker_ids) == len(y_true):
        speaker_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, speaker_id in enumerate(speaker_ids):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                speaker_analysis[speaker_id]["y_true"].append(valid_y_true[valid_index])
                speaker_analysis[speaker_id]["y_pred"].append(valid_y_pred[valid_index])
        speaker_summaries = {}
        for speaker_id, data in speaker_analysis.items():
            if len(data["y_true"]) > 0:
                speaker_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    speaker_precision, speaker_recall, speaker_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    speaker_precision = speaker_recall = speaker_f1 = 0.0
                speaker_age_groups = Counter(data["y_true"])
                speaker_summaries[speaker_id] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": speaker_accuracy,
                    "precision_macro": speaker_precision,
                    "recall_macro": speaker_recall,
                    "f1_macro": speaker_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p),
                    "age_group_distribution": dict(speaker_age_groups)
                }
        evaluation_report["speaker_level_analysis"] = speaker_summaries
    return evaluation_report

class VoxAgeTimingStats:
    """Global timing stats class, not grouped by folder"""
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_tokens = 0
        self.timing_records = []
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens=0, age_group=None):
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        self.timing_records.append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        })
    def get_summary(self):
        """Get timing summary"""
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "avg_tokens_per_sec": 0
            }
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_decode_time": self.total_decode_time / self.samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        }
    def export_to_json(self, output_file):
        """Export stats to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return output_file
    def print_summary(self):
        """Print timing summary"""
        summary = self.get_summary()
        print(f"\n=== Timing Summary ===")
        print(f"Valid samples: {summary['samples']}")
        print(f"Average Prefill time: {summary['avg_prefill_time']:.4f} seconds")
        print(f"Average Decode time: {summary['avg_decode_time']:.4f} seconds")
        print(f"Average Total time: {summary['avg_total_time']:.4f} seconds")
        print(f"Average tokens/sec: {summary['avg_tokens_per_sec']:.2f}")

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
            print("No timing statistics data")
            return
        print("\n=== CUDA Event Timing Statistics ===")
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill Timing:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode Timing:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total Timing:")
        print(f"  Average: {total_stats['total_avg']:.6f}s")
        print(f"  Samples: {total_stats['total_count']}")

gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

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
    print("Debug mode enabled")

data_path_root = os.environ.get("VOXCELEB_AGE_DATA_PATH", 
    '/data/to/your/voxceleb_age_data_path')
result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/results_dir')
os.makedirs(result_dir, exist_ok=True)

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    """Load dataset from concatenated_audio directory, based on age_classification_task_meta.json"""
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found: {meta_file}")
        return []
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        if not os.path.exists(wav_path):
            if debug_mode:
                print(f"Warning: File does not exist {wav_path}")
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
            "choice_e": item.get("choice_e", ""),
            "answer_gt": age_group,
            "task": "Speaker_Age_Classification"
        })
    print(f"Loaded {len(all_samples)} valid audio samples")
    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Applying sample limit: randomly select {sample_limit} from {len(all_samples)} samples")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Sample count after limiting: {len(all_samples)}")
    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    print("Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    random.shuffle(all_samples)
    return all_samples

def extract_age_answer(text, choices):
    """Extract age group answer from model output text"""
    text_lower = text.lower().strip()
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return choices["choice_d"]
    if choices.get("choice_e") and (text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)')):
        return choices["choice_e"]
    for option, choice_text in choices.items():
        if not choice_text:
            continue
        option_letter = option[-1].lower()
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"{option_letter})" in text_lower:
            return choice_text
    choice_matches = []
    for choice_text in choices.values():
        if choice_text and choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    if len(choice_matches) == 1:
        return choice_matches[0]
    return ""

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
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

def main():
    args = parse_arguments()
    sparse_suffix = f"_gpu{gpu_id}_{method_is}_prune_{prune_ratio}" if args.sparse else f"_gpu{gpu_id}_base"
    output_file = f'{result_dir}/VoxCeleb_age_Aero1_results{sparse_suffix}.json'
    timing_output_file = f'{result_dir}/VoxCeleb_age_Aero1_timing_stats{sparse_suffix}.json'
    cuda_event_output_file = f'{result_dir}/VoxCeleb_age_Aero1_cuda_event_stats{sparse_suffix}.json'
    print(f"\n=== Vox Age DART Age Classification Evaluation Config (Aero-1) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Data directory: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")
    print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")
    print("=" * 50)
    print("Loading Aero-1 model...")
    sys.stdout.flush()
    model_path = args.model_path
    print(f"Using Aero-1 model: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision="main",
        trust_remote_code=True
    )
    print("Successfully loaded Aero processor")
    sys.stdout.flush()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    configure_DART(model, args)
    print("Successfully loaded Aero-1 model and configured DART")
    sys.stdout.flush()
    timing_stats = VoxAgeTimingStats()
    cuda_event_stats = CudaEventTimingStats()
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    if not samples:
        print("Error: Failed to load any samples")
        return
    all_predictions = []
    all_ground_truths = []
    all_speaker_ids = []
    all_sample_results = []
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    print(f"Start processing {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing VoxCeleb age classification samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, sample in enumerate(samples):
            wav_path = sample['wav_path']
            speaker_id = sample["speaker_id"]
            ground_truth = sample["age_group"].strip()
            choices = {
                "choice_a": sample["choice_a"],
                "choice_b": sample["choice_b"],
                "choice_c": sample["choice_c"],
                "choice_d": sample["choice_d"]
            }
            if sample.get("choice_e"):
                choices["choice_e"] = sample["choice_e"]
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            predicted_age_group = ""
            is_correct = False
            try:
                messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                audio_chunks, sample_rate = prepare_audio_for_processor(wav_path)
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",
                    })
                instruction = "Listen to this audio and identify the speaker's age group. Choose the most appropriate option: (a) Young Adult (18-30), (b) Early Career (31-40), (c) Mid Career (41-50), (d) Senior (51-70), (e) Elderly (71+). Answer with only the letter (a, b, c, d, or e)."
                messages[0]["content"].append({
                    "type": "text",
                    "text": instruction
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
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    if args.sparse:
                        model.config.DART_config['audio_token_start_index'] = audio_token_start
                        model.config.DART_config['audio_token_length'] = audio_token_length
                    model.config.image_layer_idx = None
                    model.config.audio_layer_idx = prune_layer_idx
                    model.config.audio_token_num = audio_token_length
                    model.config.audio_token_start = audio_token_start
                    model.config.audio_prune_ratio = prune_ratio
                    model.config.random = use_random
                    model.config.frame = use_frame
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
                predicted_age_group = extract_age_answer(output, choices)
                all_predictions.append(predicted_age_group)
                all_ground_truths.append(ground_truth)
                all_speaker_ids.append(speaker_id)
                is_correct = predicted_age_group == ground_truth
                if i > 0 and i <= 100:
                    timing_stats.add_record(prefill_time, decode_time, output_tokens, inputs['input_ids'].shape[1], age_group=ground_truth)
                    cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)
            except Exception as e:
                print(f"Inference error: {e}")
                if debug_mode:
                    traceback.print_exc()
                output = "ERROR"
                predicted_age_group = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                all_predictions.append(predicted_age_group)
                all_ground_truths.append(ground_truth)
                all_speaker_ids.append(speaker_id)
            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_age_group,
                "is_correct": is_correct,
                "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            }
            all_sample_results.append(sample_result)
            if 'inputs' in locals():
                del inputs
            if 'audio_chunks' in locals():
                del audio_chunks
            if 'result' in locals():
                del result
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p != "error" and t != "error") / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p != "error" and t != "error"))
            pbar.set_postfix({
                'Samples': f'{i+1}/{len(samples)}',
                'Accuracy': f'{current_accuracy:.3f}',
                'Speaker': speaker_id[:8] + '...' if len(speaker_id) > 8 else speaker_id
            })
            pbar.update()
    metrics_result = calculate_metrics(all_predictions, all_ground_truths)
    final_stats = timing_stats.get_summary()
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn VoxCeleb Age Classification DART Evaluation Report ===")
        print(f"Total samples: {len(all_predictions)} (predicted), {len(all_ground_truths)} (ground truth)")
        sklearn_evaluation = generate_sklearn_voxceleb_age_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            speaker_ids=all_speaker_ids,
            labels=None
        )
        print(f"sklearn VoxCeleb Age Classification DART Evaluation Report generated:")
        print(f"  Overall Accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Macro F1: {sklearn_evaluation['overall_metrics']['f1_macro']:.4f}")
        print(f"  Micro F1: {sklearn_evaluation['overall_metrics']['f1_micro']:.4f}")
        print(f"  Weighted F1: {sklearn_evaluation['overall_metrics']['f1_weighted']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
    else:
        print("Warning: No valid predictions, cannot generate sklearn evaluation report")
        sklearn_evaluation = {"error": "No valid predictions for evaluation"}
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    age_group_results = {}
    for result in all_sample_results:
        group = result['ground_truth']
        if group not in age_group_results:
            age_group_results[group] = {'total': 0, 'correct': 0}
        age_group_results[group]['total'] += 1
        if result['is_correct']:
            age_group_results[group]['correct'] += 1
    results = {
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
            "sklearn_evaluation": sklearn_evaluation,
            "timing": final_stats,
            "config": {
                "model_name": "Aero-1-Audio-1.5B",
                "gpu_id": gpu_id,
                "sparse": args.sparse,
                "pruned_layer": args.pruned_layer,
                "reduction_ratio": args.reduction_ratio,
                "prune_method": method_is,
                "sample_limit": sample_limit
            }
        }
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing stats saved to: {timing_output_file}")
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)
    print(f"CUDA Event stats saved to: {cuda_event_output_file}")
    print("\n=== Evaluation Summary (Aero-1 + DART) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Total samples: {total_samples}")
    print(f"Total accuracy: {results['summary']['accuracy']:.2%}")
    for group, stats in results['summary']['age_group_results'].items():
        print(f"{group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"F1 Score: {metrics_result['f1_score']:.4f}")
    print(f"Precision: {metrics_result['precision']:.4f}")  
    print(f"Recall: {metrics_result['recall']:.4f}")
    if "sklearn_evaluation" in results["summary"] and "error" not in results["summary"]["sklearn_evaluation"]:
        sklearn_metrics = results["summary"]["sklearn_evaluation"]["overall_metrics"]
        sample_stats = results["summary"]["sklearn_evaluation"]["sample_statistics"]
        print(f"\n=== Sklearn Evaluation Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision - Macro: {sklearn_metrics['precision_macro']:.4f}")
        print(f"Recall - Macro: {sklearn_metrics['recall_macro']:.4f}")
        print(f"F1 Score - Macro: {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 Score - Micro: {sklearn_metrics['f1_micro']:.4f}")
        print(f"F1 Score - Weighted: {sklearn_metrics['f1_weighted']:.4f}")
        print(f"\nAge group distribution:")
        age_group_dist = sample_stats.get("age_group_distribution", {})
        for age_group, count in age_group_dist.items():
            percentage = count / sample_stats['valid_samples'] * 100 if sample_stats['valid_samples'] > 0 else 0
            print(f"  {age_group}: {count} samples ({percentage:.1f}%)")
        print(f"\nDetailed metrics per age group:")
        per_age_group_metrics = results["summary"]["sklearn_evaluation"]["per_age_group_metrics"]
        for age_group in sorted(per_age_group_metrics.keys()):
            metrics = per_age_group_metrics[age_group]
            print(f"  {age_group}: Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}, "
                  f"Support={metrics['support']}")
        if "speaker_level_analysis" in results["summary"]["sklearn_evaluation"]:
            speaker_analysis = results["summary"]["sklearn_evaluation"]["speaker_level_analysis"]
            print(f"\nTop 5 speaker-level analysis:")
            sorted_speaker_analysis = sorted(speaker_analysis.items(), 
                                           key=lambda x: x[1]['accuracy'], reverse=True)
            for speaker_id, analysis in sorted_speaker_analysis[:5]:
                age_dist = analysis['age_group_distribution']
                age_groups = ", ".join([f"{group}({count})" for group, count in