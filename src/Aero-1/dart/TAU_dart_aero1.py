import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import gc
import traceback
import glob

# Disable warnings and set environment
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'

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
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable DART sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio for DART')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    parser.add_argument('--use_random', type=str_to_bool, default=False, help='Use random pruning')
    parser.add_argument('--use_frame', type=str_to_bool, default=False, help='Use frame pruning')
    return parser.parse_args()

def configure_DART(model, args):
    """Configure DART sparse attention mechanism for Aero-1"""
    if args.sparse:
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = args.reduction_ratio
        model.config.random = args.use_random
        model.config.frame = args.use_frame
        DART_config = {
            "K": args.pruned_layer,
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            "reduction_ratio": args.reduction_ratio,
            "pivot_audio_token": args.pivot_audio_token,
            "pivot_text_token": args.pivot_text_token,
            "text_length": 1,
        }
        model.config.DART_config = DART_config
        print(f"DART sparse attention enabled: layer={args.pruned_layer}, ratio={args.reduction_ratio}")
    else:
        model.config.DART_config = None
        model.config.audio_layer_idx = None
        print("Using base model (no DART optimization)")

# Get GPU ID and config
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]

print(f"Using GPU ID: {gpu_id}")
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")
if debug_mode:
    print("Debug mode enabled")

# Data path config - change to TAU dataset path
data_path_root = '/data/to/your/tau/concatenated_resampled'
result_dir = '/data/to/your/tau_dart_results'
os.makedirs(result_dir, exist_ok=True)

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class TAUTimingStats:
    """Track inference timing statistics for TAU acoustic scene classification task using CUDA Event precision"""
    def __init__(self):
        self.timing_records = []
        self.scene_stats = defaultdict(list)
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, scene_label=None):
        """Add timing record"""
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "scene_label": scene_label
        }
        self.timing_records.append(record)
        
        if scene_label:
            self.scene_stats[scene_label].append(record)
    
    def get_summary(self):
        """Get overall timing summary"""
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
        scene_summaries = {}
        for scene_label, records in self.scene_stats.items():
            if len(records) > 0:
                scene_df = pd.DataFrame(records)
                scene_summaries[scene_label] = {
                    "samples": len(records),
                    "avg_prefill_time": scene_df["prefill_time"].mean(),
                    "avg_decode_time": scene_df["decode_time"].mean(),
                    "avg_total_time": scene_df["total_time"].mean(),
                    "avg_tokens_per_sec": scene_df["decode_tokens_per_sec"].mean()
                }
        return {
            "overall_summary": summary,
            "scene_summaries": scene_summaries
        }
    
    def export_to_json(self, output_file):
        """Export timing stats to JSON file"""
        summary = self.get_summary()
        data = {
            "summary": summary,
            "detailed_records": self.timing_records
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples @ 16kHz)"""
    CHUNK_LIM = 480000
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

def load_tau_acoustic_scene_dataset(root_dir):
    """Load TAU dataset for acoustic scene classification task"""
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found: {meta_file}")
        return [], {}
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    scene_counts = {}
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(wav_path):
            print(f"Warning: File not found {wav_path}")
            continue
        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"] # A, B, C, D
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        all_samples.append({
            "scene_label": scene_label,
            "audio_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification",
            "id": f"tau_scene_{len(all_samples)}"
        })
    print(f"Total loaded {len(all_samples)} valid audio samples")
    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count} samples ({count/len(all_samples)*100:.1f}%)")
    random.shuffle(all_samples)
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """Extract acoustic scene answer option (A/B/C/D) from model output text"""
    import re
    text_lower = text.lower().strip()
    options = ['a', 'b', 'c', 'd']
    if text_lower in options:
        return text_lower.upper()
    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    match = re.search(r'\b([ABCD])\b', text.upper())
    if match:
        return match.group(1)
    match = re.search(r'[(\[]?([ABCD])[)\].]?', text.upper())
    if match:
        return match.group(1)
    if choices:
        best_match = None
        max_overlap = 0
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()
            if choice_lower in text_lower:
                return chr(65 + i)
            keywords = choice_lower.split(' - ')[0].split()
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        if best_match and max_overlap > 1:
            return best_match
    return ""

def calculate_tau_metrics(predictions, ground_truths):
    """Calculate F1 score and other metrics for TAU task"""
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
    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    precision = precision_score(valid_ground_truths, valid_predictions, average='weighted', zero_division=0)
    recall = recall_score(valid_ground_truths, valid_predictions, average='weighted', zero_division=0)
    f1 = f1_score(valid_ground_truths, valid_predictions, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions)
    }

def generate_sklearn_tau_dart_evaluation_report(y_true, y_pred, scene_labels=None, labels=None):
    """
    Generate detailed evaluation report for TAU acoustic scene classification task (DART version) using sklearn
    
    Args:
        y_true: List of true labels (e.g. ['A', 'B', 'C', 'D'])
        y_pred: List of predicted labels (e.g. ['A', 'B', 'C', 'D'])
        scene_labels: List of scene labels (optional), for scene analysis
        labels: List of label names for classification report
    Returns:
        dict: Dictionary containing various evaluation metrics
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
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average=None, labels=['A', 'B', 'C', 'D'], zero_division=0
    )
    if labels is None:
        labels = ['A', 'B', 'C', 'D']
    classification_rep = classification_report(
        valid_y_true, valid_y_pred, 
        target_names=labels, 
        output_dict=True,
        zero_division=0
    )
    correct_predictions = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == p)
    total_predictions = len(valid_y_true)
    evaluation_report = {
        "overall_metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted
        },
        "per_class_metrics": {},
        "classification_report": classification_rep,
        "sample_statistics": {
            "total_samples": len(y_true),
            "valid_samples": len(valid_y_true),
            "invalid_samples": len(y_true) - len(valid_y_true),
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions
        }
    }
    class_labels = ['A', 'B', 'C', 'D']
    for i, class_label in enumerate(class_labels):
        if i < len(precision_per_class):
            evaluation_report["per_class_metrics"][class_label] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1_score": f1_per_class[i],
                "support": int(support_per_class[i]) if i < len(support_per_class) else 0
            }
    option_stats = {}
    for option in ['A', 'B', 'C', 'D']:
        option_stats[option] = {
            "true_count": sum(1 for label in valid_y_true if label == option),
            "pred_count": sum(1 for label in valid_y_pred if label == option)
        }
    evaluation_report["option_distribution"] = option_stats
    if scene_labels and len(scene_labels) == len(y_true):
        scene_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, scene_label in enumerate(scene_labels):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                scene_analysis[scene_label]["y_true"].append(valid_y_true[valid_index])
                scene_analysis[scene_label]["y_pred"].append(valid_y_pred[valid_index])
        scene_summaries = {}
        for scene_label, data in scene_analysis.items():
            if len(data["y_true"]) > 0:
                scene_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    scene_precision, scene_recall, scene_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    scene_precision = scene_recall = scene_f1 = 0.0
                scene_summaries[scene_label] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": scene_accuracy,
                    "precision_macro": scene_precision,
                    "recall_macro": scene_recall,
                    "f1_macro": scene_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        evaluation_report["scene_analysis"] = scene_summaries
    return evaluation_report

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference function with precise GPU timing using CUDA Event API
    Optimized for Aero-1 model, supports DART sparse attention
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

def create_tau_prompt_aero1(sample):
    """Create Aero-1 format prompt for TAU acoustic scene classification task"""
    instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option."
    option_text = f"A: {sample['choice_a']}\nB: {sample['choice_b']}\nC: {sample['choice_c']}\nD: {sample['choice_d']}"
    format_instruction = "Respond with only the letter of your answer (A, B, C, or D)."
    return f"{instruction}\n\n{option_text}\n\n{format_instruction}"

def main():
    random.seed(42)
    args = parse_arguments()
    print(f"\n=== TAU Acoustic Scene Classification Aero-1+DART Evaluation Config ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Model: {args.model_path}")
    print(f"DART Sparse Mode: {args.sparse}")
    print(f"Pruned Layers: {args.pruned_layer}")
    print(f"Retained Ratio: {args.reduction_ratio}")
    print(f"Data Directory: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample Limit: {sample_limit}")
    print("=" * 40)
    if args.sparse:
        if args.use_random:
            method_name = "random"
        elif args.use_frame:
            method_name = "frame"
        else:
            method_name = "fast_v"
        method_suffix = f"_{method_name}_prune{args.reduction_ratio}"
    else:
        method_suffix = "_base"
    output_file = f'{result_dir}/TAU_Aero1_DART_results_gpu{gpu_id}{method_suffix}.json'
    timing_output_file = f'{result_dir}/TAU_Aero1_DART_timing_stats_gpu{gpu_id}{method_suffix}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")
    timing_stats = TAUTimingStats()
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
    configure_DART(model, args)
    samples, scene_counts = load_tau_acoustic_scene_dataset(data_path_root)
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample number limited to: {len(samples)}")
    grouped_samples = defaultdict(list)
    for sample in samples:
        grouped_samples[sample["scene_label"]].append(sample)
    for scene, scene_samples in grouped_samples.items():
        print(f"Scene {scene}: {len(scene_samples)} samples")
    all_predictions = []
    all_ground_truths = []
    all_scene_labels = []
    results = {
        "model_name": "Aero-1-Audio-1.5B",
        "task": "Acoustic_Scene_Classification",
        "dart_config": {
            "enabled": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "use_random": args.use_random,
            "use_frame": args.use_frame,
            "method": method_name if args.sparse else "base"
        },
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "scene_stats": {scene: {"total": 0, "correct": 0} for scene in scene_counts},
            "metrics": {},
            "timing": {}
        }
    }
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loaded - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    with tqdm(total=len(samples), desc="Processing TAU acoustic scene samples (Aero-1+DART)", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                audio_chunks, sample_rate = prepare_audio_for_processor(sample["audio_path"])
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",
                    })
                text_prompt = create_tau_prompt_aero1(sample)
                messages[0]["content"].append({
                    "type": "text",
                    "text": text_prompt
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
                        model.config.audio_token_num = audio_token_length
                        model.config.audio_token_start = audio_token_start
                        if hasattr(model.config, 'DART_config') and model.config.DART_config:
                            model.config.DART_config['audio_token_start_index'] = audio_token_start
                            model.config.DART_config['audio_token_length'] = audio_token_length
                if debug_mode:
                    print(f"Processing audio: {os.path.basename(sample['audio_path'])}")
                    print(f"Scene: {sample['scene_label']}")
                    print(f"Audio chunk count: {len(audio_chunks)}")
                    print(f"Audio token length: {audio_token_length}")
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
                input_tokens = inputs['input_ids'].shape[1]
                choices = [sample['choice_a'], sample['choice_b'], sample['choice_c'], sample['choice_d']]
                predicted_label = extract_acoustic_scene_answer(output, choices)
                true_label = sample["answer_gt"]
                is_correct = predicted_label == true_label
                all_predictions.append(predicted_label)
                all_ground_truths.append(true_label)
                all_scene_labels.append(sample["scene_label"])
                sample_result = {
                    "id": sample["id"],
                    "audio_path": sample["audio_path"],
                    "scene_label": sample["scene_label"],
                    "question": sample["question"],
                    "choice_a": sample["choice_a"],
                    "choice_b": sample["choice_b"],
                    "choice_c": sample["choice_c"],
                    "choice_d": sample["choice_d"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "response": output,
                    "is_correct": is_correct,
                    "audio_chunks": len(audio_chunks),
                    "audio_tokens": audio_token_length,
                    "timing": {
                        "prefill_time": prefill_time,
                        "decode_time": decode_time,
                        "total_time": total_time,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_sec": output_tokens/decode_time if decode_time > 0 else 0
                    }
                }
                results["samples"].append(sample_result)
                if idx > 0 and idx <= 100:
                    timing_stats.add_record(
                        prefill_time,
                        decode_time,
                        output_tokens,
                        input_tokens,
                        scene_label=sample["scene_label"]
                    )
                results["summary"]["total_samples"] += 1
                if is_correct:
                    results["summary"]["correct_samples"] += 1
                scene_label = sample["scene_label"]
                results["summary"]["scene_stats"][scene_label]["total"] += 1
                if is_correct:
                    results["summary"]["scene_stats"][scene_label]["correct"] += 1
                pbar.set_postfix({
                    'Acc': f"{results['summary']['correct_samples']/results['summary']['total_samples']:.3f}",
                    'Scene': scene_label[:15] + '...' if len(scene_label) > 15 else scene_label,
                    'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                })
                del inputs
                del audio_chunks
                del result
                torch.cuda.empty_cache()
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                pbar.update(1)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                if debug_mode:
                    traceback.print_exc()
                all_predictions.append("")
                all_ground_truths.append(sample.get("answer_gt", "A"))
                all_scene_labels.append(sample.get("scene_label", "unknown"))
                pbar.update(1)
                continue
    if len(all_predictions) > 0:
        metrics = calculate_tau_metrics(all_predictions, all_ground_truths)
        results["summary"]["metrics"] = metrics
        print(f"\n=== TAU Acoustic Scene Classification Evaluation Results (Aero-1+DART) ===")
        print(f"Model: {args.model_path}")
        print(f"DART Config: {results['dart_config']}")
        print(f"Total Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")
    for scene_label, stats in results["summary"]["scene_stats"].items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn TAU DART Evaluation Report ===")
        print(f"Total samples: {len(all_predictions)} (predicted), {len(all_ground_truths)} (ground truth)")
        sklearn_evaluation = generate_sklearn_tau_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            scene_labels=all_scene_labels,
            labels=['A', 'B', 'C', 'D']
        )
        print(f"sklearn TAU DART evaluation report generated:")
        print(f"  Overall accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Precision (macro): {sklearn_evaluation['overall_metrics']['precision_macro']:.4f}")
        print(f"  Recall (macro): {sklearn_evaluation['overall_metrics']['recall_macro']:.4f}")
        print(f"  F1 score (macro): {sklearn_evaluation['overall_metrics']['f1_macro']:.4f}")
        print(f"  F1 score (weighted): {sklearn_evaluation['overall_metrics']['f1_weighted']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
        results["summary"]["sklearn_evaluation"] = sklearn_evaluation
    else:
        print("Warning: No valid prediction results, cannot generate sklearn evaluation report")
        results["summary"]["sklearn_evaluation"] = {"error": "No valid predictions for evaluation"}
    timing_summary = timing_stats.get_summary()
    results["summary"]["timing"] = timing_summary
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing stats saved to: {timing_output_file}")
    if "sklearn_evaluation" in results["summary"] and "error" not in results["summary"]["sklearn_evaluation"]:
        sklearn_metrics = results["summary"]["sklearn_evaluation"]["overall_metrics"]
        print(f"\n=== Sklearn Evaluation Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {sklearn_metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {sklearn_metrics['recall_macro']:.4f}")
        print(f"F1 Score (Macro): {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {sklearn_metrics['f1_weighted']:.4f}")
        print(f"\nDetailed metrics per option:")
        per_class_metrics = results["summary"]["sklearn_evaluation"]["per_class_metrics"]
        for option in sorted(per_class_metrics.keys()):
            metrics_detail = per_class_metrics[option]
            print(f"  Option {option}: Precision={metrics_detail['precision']:.4f}, "
                  f"Recall={metrics_detail['recall']:.4f}, F1={metrics_detail['f1_score']:.4f}, "
                  f"Support={metrics_detail['support']}")
        print(f"\nOption distribution stats:")
        option_dist = results["summary"]["sklearn_evaluation"]["option_distribution"]
        for option in sorted(option_dist.keys()):
            stats = option_dist[option]
            print(f"  Option {option}: Ground Truth={stats['true_count']}, Prediction={stats['pred_count']}")
        if "scene_analysis" in results["summary"]["sklearn_evaluation"]:
            scene_analysis = results["summary"]["sklearn_evaluation"]["scene_analysis"]
            print(f"\nTop 5 scenes analysis:")
            for scene_label, analysis in list(scene_analysis.items())[:5]:
                print(f"  {scene_label}: Accuracy={analysis['accuracy']:.4f}, "
                      f"F1(Macro)={analysis['f1_macro']:.4f}, "
                      f"Sample count={analysis['sample_count']}, "
                      f"Correct count={analysis['correct_count']}")

if __name__ == "__main__":
    main()