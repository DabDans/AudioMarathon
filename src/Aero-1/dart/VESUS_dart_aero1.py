import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
import gc
import traceback
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict
import soundfile as sf
import pandas as pd

# Environment config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Disable transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Audio special token ID for Aero-1
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
    parser.add_argument('--prune_method', type=str, default="fast_v", 
                       choices=["fast_v", "random", "frame", "base"], help='pruning method')
    return parser.parse_args()

def configure_DART_for_Aero1(model, args):
    """Configure DART sparse attention for Aero-1 model"""
    if args.sparse and args.reduction_ratio > 0:
        # Set Aero-1 specific pruning config
        prune_ratio = 1.0 - args.reduction_ratio  # Convert to prune ratio
        
        use_random = (args.prune_method == "random")
        use_frame = (args.prune_method == "frame")
        
        model.config.image_layer_idx = None
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = prune_ratio
        model.config.random = use_random
        model.config.frame = use_frame
        
        print(f"Configure Aero-1 DART: Prune Layer={args.pruned_layer}, Prune Ratio={prune_ratio:.3f}, Method={args.prune_method}")
    else:
        model.config.audio_layer_idx = None
        model.config.audio_prune_ratio = 0
        model.config.random = False
        model.config.frame = False
        print("Disable DART pruning")

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
    
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples@16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, data_path, target_sr=16000, debug_mode=False):
    """Load audio with librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Build full audio path
        full_audio_path = os.path.join(data_path, audio_path)
        
        if not os.path.exists(full_audio_path):
            if debug_mode:
                print(f"Audio file does not exist: {full_audio_path}")
            return None
        
        # Load audio (officially recommended way)
        audio, sample_rate = librosa.load(full_audio_path, sr=target_sr)
        
        audio = audio.astype(np.float32)
        
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30s, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        if debug_mode:
            traceback.print_exc()
        return None

def calculate_emotion_metrics(predictions, ground_truths, emotion_labels):
    """Calculate emotion classification metrics: accuracy, precision, recall, F1"""
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

def generate_sklearn_vesus_dart_evaluation_report(y_true, y_pred, emotion_labels=None, person_ids=None):
    """
    Generate detailed evaluation report for VESUS emotion recognition task (DART version) using sklearn
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
    
    classification_rep = classification_report(
        valid_y_true, valid_y_pred,
        target_names=['A', 'B', 'C', 'D'],
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
    
    if emotion_labels and len(emotion_labels) == len(y_true):
        emotion_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        
        for i, emotion_label in enumerate(emotion_labels):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                emotion_analysis[emotion_label]["y_true"].append(valid_y_true[valid_index])
                emotion_analysis[emotion_label]["y_pred"].append(valid_y_pred[valid_index])
        
        emotion_summaries = {}
        for emotion_label, data in emotion_analysis.items():
            if len(data["y_true"]) > 0:
                emotion_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    emotion_precision = emotion_recall = emotion_f1 = 0.0
                
                emotion_summaries[emotion_label] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": emotion_accuracy,
                    "precision_macro": emotion_precision,
                    "recall_macro": emotion_recall,
                    "f1_macro": emotion_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        
        evaluation_report["emotion_level_analysis"] = emotion_summaries
    
    if person_ids and len(person_ids) == len(y_true):
        person_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        
        for i, person_id in enumerate(person_ids):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                person_analysis[person_id]["y_true"].append(valid_y_true[valid_index])
                person_analysis[person_id]["y_pred"].append(valid_y_pred[valid_index])
        
        person_summaries = {}
        for person_id, data in person_analysis.items():
            if len(data["y_true"]) > 0:
                person_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    person_precision, person_recall, person_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    person_precision = person_recall = person_f1 = 0.0
                
                person_summaries[person_id] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": person_accuracy,
                    "precision_macro": person_precision,
                    "recall_macro": person_recall,
                    "f1_macro": person_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        
        evaluation_report["person_level_analysis"] = person_summaries
    
    return evaluation_report

class VESUSTimingStats:
    """Track inference timing statistics for VESUS emotion recognition, supports CUDA Event measurement"""
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
            "emotion_label": emotion_label,
            "person_id": person_id
        }
        
        self.timing_records.append(record)
        
        if emotion_label:
            self.emotion_stats[emotion_label].append(record)
        
        if person_id:
            self.person_stats[person_id].append(record)
    
    def get_summary(self):
        """Get overall timing summary"""
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
        """Export timing stats to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def load_vesus_dataset(json_file_path):
    """Load VESUS emotion recognition dataset"""
    if not os.path.exists(json_file_path):
        print(f"Error: Dataset file does not exist: {json_file_path}")
        return []
    
    print(f"Loading VESUS emotion dataset: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        valid_samples = []
        filtered_count = 0
        for item in data:
            if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
                person_id = item.get('person_id', '')
                emotion_label = item.get('emotion_label', '').lower()
                
                if (person_id in ['person2', 'person10'] and emotion_label == 'happy'):
                    filtered_count += 1
                    print(f"Filtered sample: {person_id} - {emotion_label} - {item.get('path', '')}")
                    continue
                
                valid_samples.append(item)
        
        print(f"Filtered {filtered_count} samples (person2 and person10 happy emotion)")
        print(f"Loaded {len(valid_samples)} valid samples")
        
        emotion_counts = defaultdict(int)
        person_emotion_counts = defaultdict(lambda: defaultdict(int))
        for sample in valid_samples:
            emotion = sample.get('emotion_label', 'unknown')
            person = sample.get('person_id', 'unknown')
            emotion_counts[emotion] += 1
            person_emotion_counts[person][emotion] += 1
        
        print(f"Emotion distribution: {dict(emotion_counts)}")
        print(f"Per person emotion distribution:")
        for person, emotions in person_emotion_counts.items():
            if person in ['person2', 'person10']:
                print(f"  {person}: {dict(emotions)} (happy samples filtered)")
            else:
                print(f"  {person}: {dict(emotions)}")
        
        return valid_samples
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []

def extract_emotion_answer(text, choices):
    """Extract emotion answer from model output text"""
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

def cuda_timing_inference(model, processor, inputs, max_new_tokens=64):
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
    
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
    
    # Path config
    data_path = os.environ.get("VESUS_DATA_PATH", "/data/to/your/project/Phi-4-multimodal-instruct/eval/VESUS")
    emotion_json_file = os.path.join(data_path, "audio_emotion_dataset.json")
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/VESUS_Results')
    os.makedirs(result_dir, exist_ok=True)
    
    prune_ratio = 1.0 - args.reduction_ratio if args.sparse else 0
    method_name = args.prune_method if prune_ratio > 0 else "base"
    
    output_file = f'{result_dir}/VESUS_Aero1_DART_results_gpu{gpu_id}_{method_name}_prune_{prune_ratio:.3f}.json'
    timing_output_file = f'{result_dir}/VESUS_Aero1_DART_timing_stats_gpu{gpu_id}_{method_name}_prune_{prune_ratio:.3f}.json'
    
    print(f"\n=== VESUS Emotion Recognition Evaluation Config (Aero-1 DART) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Model Path: {args.model_path}")
    print(f"DART Sparse Mode: {args.sparse}")
    print(f"Prune Layer Index: {args.pruned_layer}")
    print(f"Retained Ratio: {args.reduction_ratio}")
    print(f"Prune Ratio: {prune_ratio:.3f}")
    print(f"Prune Method: {args.prune_method}")
    print(f"Data Path: {data_path}")
    print(f"JSON File: {emotion_json_file}")
    if sample_limit > 0:
        print(f"Sample Limit: {sample_limit}")
    print("=" * 50)
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

    print("Loading Aero-1 model...")
    sys.stdout.flush()
    
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        revision="main",
        trust_remote_code=True
    )
    print("Successfully loaded Aero processor")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    print("Successfully loaded Aero-1 model")
    sys.stdout.flush()

    configure_DART_for_Aero1(model, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_vesus_dataset(emotion_json_file)
    
    if not samples:
        print("Error: No data samples found")
        return
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, processing {len(samples)} samples")

    allocated, reserved = get_gpu_memory_usage()
    print(f"After model load GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    timing_stats = VESUSTimingStats()

    results = []
    total_correct = 0
    emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    person_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    all_predictions = []
    all_ground_truths = []
    all_emotion_labels = []
    all_person_ids = []

    print(f"Start evaluating {len(samples)} samples...")
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()

    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="VESUS Evaluation (Aero-1 DART)")

    for idx, sample in progress_bar:
        prefill_time = 0
        decode_time = 0
        output_tokens = 0
        audio_token_length = 0
        predicted_answer = ""
        is_correct = False
        resp = ""
        
        try:
            audio_path = sample.get("path", "")
            audio_result = prepare_audio_for_processor(audio_path, data_path, debug_mode=debug_mode)
            
            if audio_result is None:
                continue
            
            audio_chunks, sample_rate = audio_result
            
            emotion_label = sample.get("emotion_label", "unknown")
            person_id = sample.get("person_id", "unknown")
            answer_gt = sample.get("answer_gt", "").upper()
            
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
            
            question = sample.get("question", "What emotion is expressed in this audio segment?")
            choice_a = sample.get("choice_a", "")
            choice_b = sample.get("choice_b", "")
            choice_c = sample.get("choice_c", "")
            choice_d = sample.get("choice_d", "")
            
            prompt_text = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D)."""
            
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
            
            audio_token_length = 0
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end - audio_token_start + 1
                
                if args.sparse:
                    model.config.audio_token_num = audio_token_length
                    model.config.audio_token_start = audio_token_start
            
            if debug_mode:
                print(f"Processing audio: {os.path.basename(audio_path)}")
                print(f"Emotion Label: {emotion_label}")
                print(f"Person ID: {person_id}")
                print(f"Number of audio chunks: {len(audio_chunks)}")
                print(f"Estimated audio token length: {audio_token_length}")
                sys.stdout.flush()
            
            result = cuda_timing_inference(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=64
            )
            
            resp = result['response_text']
            prefill_time = result['prefill_time']
            decode_time = result['decode_time']
            total_time = result['total_time']
            output_tokens = result['output_tokens']

            predicted_answer = extract_emotion_answer(resp, sample)
            is_correct = (predicted_answer == answer_gt)

            all_predictions.append(predicted_answer)
            all_ground_truths.append(answer_gt)
            all_emotion_labels.append(emotion_label)
            all_person_ids.append(person_id)

            if is_correct:
                total_correct += 1
            
            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1
            
            if is_correct:
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1

            current_accuracy = total_correct / (idx + 1)
            progress_bar.set_postfix({
                'Acc': f'{current_accuracy:.3f}',
                'Emotion': emotion_label[:8],
                'Person': person_id
            })

            results.append({
                "idx": idx,
                "path": audio_path,
                "emotion_label": emotion_label,
                "person_id": person_id,
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "answer_gt": answer_gt,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response_text": resp,
                "audio_chunks": len(audio_chunks),
                "audio_tokens": audio_token_length,
                "timing": {
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": total_time
                }
            })

            if idx > 0 and idx <= 100:
                timing_stats.add_record(
                    prefill_time, decode_time, 
                    output_tokens,
                    inputs["input_ids"].shape[1],
                    emotion_label, person_id
                )

            del inputs, result
            if 'audio_chunks' in locals():
                del audio_chunks
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
            all_ground_truths.append(sample.get("answer_gt", "").upper())
            all_emotion_labels.append(sample.get("emotion_label", "unknown"))
            all_person_ids.append(sample.get("person_id", "unknown"))
            continue

    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    all_predictions_for_metrics = [result["predicted_answer"] for result in results]
    all_ground_truths_for_metrics = [result["answer_gt"] for result in results]
    all_emotion_labels_for_metrics = list(set(all_ground_truths_for_metrics))
    
    emotion_metrics = calculate_emotion_metrics(all_predictions_for_metrics, all_ground_truths_for_metrics, all_emotion_labels_for_metrics)

    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn VESUS DART evaluation report ===")
        print(f"Total samples: {len(all_predictions)} (pred), {len(all_ground_truths)} (true labels)")
        
        sklearn_evaluation = generate_sklearn_vesus_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            emotion_labels=all_emotion_labels,
            person_ids=all_person_ids
        )
        
        print(f"sklearn VESUS DART evaluation report generated:")
        print(f"  Overall accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Macro F1: {sklearn_evaluation['overall_metrics']['f1_macro']:.4f}")
        print(f"  Micro F1: {sklearn_evaluation['overall_metrics']['f1_micro']:.4f}")
        print(f"  Weighted F1: {sklearn_evaluation['overall_metrics']['f1_weighted']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
    else:
        print("Warning: No valid predictions, cannot generate sklearn evaluation report")
        sklearn_evaluation = {"error": "No valid predictions for evaluation"}

    emotion_accuracies = {}
    for emotion, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracies[emotion] = stats["correct"] / stats["total"]

    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]

    summary = {
        "total_samples": total_samples,
        "correct_samples": total_correct,
        "overall_accuracy": overall_accuracy,
        "metrics": emotion_metrics,
        "sklearn_evaluation": sklearn_evaluation,
        "emotion_stats": dict(emotion_stats),
        "emotion_accuracies": emotion_accuracies,
        "person_stats": dict(person_stats),
        "person_accuracies": person_accuracies,
        "config": {
            "model_name": args.model_path,
            "gpu_id": gpu_id,
            "dart_sparse": args.sparse,
            "prune_layer_idx": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "prune_ratio": prune_ratio,
            "prune_method": args.prune_method,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file
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

    print(f"\n=== VESUS Emotion Recognition Evaluation Summary (Aero-1 DART) ===")
    print(f"Model: {args.model_path}")
    print(f"DART Config: Sparse={args.sparse}, Layer={args.pruned_layer}, Retained Ratio={args.reduction_ratio}, Method={args.prune_method}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"F1 Score: {emotion_metrics['f1_score']:.4f}")
    print(f"Precision: {emotion_metrics['precision']:.4f}")
    print(f"Recall: {emotion_metrics['recall']:.4f}")
    print(f"Valid samples: {emotion_metrics['valid_samples']}/{emotion_metrics['total_samples']}")
    
    if "sklearn_evaluation" in summary and "error" not in summary["sklearn_evaluation"]:
        sklearn_metrics = summary["sklearn_evaluation"]["overall_metrics"]
        
        print(f"\n=== Sklearn Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision - Macro Avg: {sklearn_metrics['precision_macro']:.4f}")
        print(f"Recall - Macro Avg: {sklearn_metrics['recall_macro']:.4f}")
        print(f"F1 Score - Macro Avg: {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 Score - Micro Avg: {sklearn_metrics['f1_micro']:.4f}")
        print(f"F1 Score - Weighted Avg: {sklearn_metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-choice Metrics:")
        per_choice_metrics = summary["sklearn_evaluation"]["per_choice_metrics"]
        for choice in sorted(per_choice_metrics.keys()):
            metrics_detail = per_choice_metrics[choice]
            print(f"  Choice {choice}: Precision={metrics_detail['precision']:.4f}, "
                  f"Recall={metrics_detail['recall']:.4f}, F1={metrics_detail['f1_score']:.4f}, "
                  f"Support={metrics_detail['support']}")
        
        if "emotion_level_analysis" in summary["sklearn_evaluation"]:
            emotion_analysis = summary["sklearn_evaluation"]["emotion_level_analysis"]
            print(f"\nEmotion-level Analysis:")
            for emotion, analysis in emotion_analysis.items():
                print(f"  {emotion}: Accuracy={analysis['accuracy']:.4f}, "
                      f"F1={analysis['f1_macro']:.4f}, "
                      f"Samples={analysis['sample_count']}, "
                      f"Correct={analysis['correct_count']}")
        
        if "person_level_analysis" in summary["sklearn_evaluation"]:
            person_analysis = summary["sklearn_evaluation"]["person_level_analysis"]
            print(f"\nPerson-level Analysis:")
            for person_id, analysis in person_analysis.items():
                print(f"  {person_id}: Accuracy={analysis['accuracy']:.4f}, "
                      f"F1={analysis['f1_macro']:.4f}, "
                      f"Samples={analysis['sample_count']}, "
                      f"Correct={analysis['correct_count']}")
    
    print(f"\nEmotion accuracies (traditional calculation):")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== Timing Stats (CUDA Events Measurement)===")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f}s")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/s")
    print(f"Results saved to: {output_file}")
    print(f"Timing stats saved to: {timing_output_file}")

if __name__ == "__main__":
    main()