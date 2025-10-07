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

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Disable transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Audio special token ID (Aero-1 model)
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
    return parser.parse_args()

def configure_DART_aero1(model, args):
    """Configure DART sparse attention mechanism - Aero-1 version"""
    if args.sparse:
        # Configure audio pruning parameters for Aero-1 model
        model.config.image_layer_idx = None
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = 1 - args.reduction_ratio  # Pruning ratio instead of retained ratio
        model.config.random = False  # Default to fast_v method
        model.config.frame = False
        
        # DART configuration for compatibility
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
        model.config.DART_config = DART_config
        print(f"DART sparse attention enabled - Layer: {args.pruned_layer}, Retained ratio: {args.reduction_ratio}")
    else:
        model.config.DART_config = None
        model.config.audio_prune_ratio = 0
        print("Using base mode (no DART sparse attention)")

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

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio using librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Load audio using librosa (official recommended way)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure data type is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sampling rate does not match
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split if audio longer than 30s
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            return audio_chunks, sample_rate
        else:
            # Return single chunk in list if not longer than 30s
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        # Return silence chunk list
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

# Get GPU ID
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

# Sample limit (if provided)
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

class GTZANTimingStats:
    """Track GTZAN task inference timing statistics, using CUDA Event for precise measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration, genre=None):
        """Add a timing record"""
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
        """Get overall statistics summary"""
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
    """Calculate music genre classification metrics: accuracy, precision, recall, F1 score"""
    # Filter out invalid predictions and ground truths
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
    
    # Convert to numeric labels
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    
    # Calculate metrics
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

def generate_sklearn_gtzan_dart_evaluation_report(y_true, y_pred, genres=None, labels=None):
    """
    Generate detailed evaluation report for GTZAN music genre classification task (DART version) using sklearn
    
    Args:
        y_true: list of ground truth labels (e.g. ['A', 'B', 'C', 'D'])
        y_pred: list of predicted labels (e.g. ['A', 'B', 'C', 'D'])
        genres: list of music genres, for genre-level analysis
        labels: list of label names, for classification report
    
    Returns:
        dict: dictionary containing various evaluation metrics
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"error": "Invalid input data for evaluation"}
    
    # Filter valid predictions and ground truths
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
    
    # Calculate accuracy
    accuracy = accuracy_score(valid_y_true, valid_y_pred)
    
    # Calculate precision, recall, F1 (macro/micro/weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average=None, labels=['A', 'B', 'C', 'D'], zero_division=0
    )
    
    # Classification report
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
    
    # Build complete evaluation report
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
    
    # Add per-class metrics
    choice_labels = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choice_labels):
        if i < len(precision_per_class):
            evaluation_report["per_choice_metrics"][choice] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1_score": f1_per_class[i],
                "support": int(support_per_class[i]) if i < len(support_per_class) else 0
            }
    
    # If genres provided, add genre-level analysis
    if genres and len(genres) == len(y_true):
        genre_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        
        for i, genre in enumerate(genres):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                genre_analysis[genre]["y_true"].append(valid_y_true[valid_index])
                genre_analysis[genre]["y_pred"].append(valid_y_pred[valid_index])
        
        genre_summaries = {}
        for genre, data in genre_analysis.items():
            if len(data["y_true"]) > 0:
                genre_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                try:
                    genre_precision, genre_recall, genre_f1, _ = precision_recall_fscore_support(
                        data["y_true"], data["y_pred"], average='macro', zero_division=0
                    )
                except:
                    genre_precision = genre_recall = genre_f1 = 0.0
                
                genre_summaries[genre] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": genre_accuracy,
                    "precision_macro": genre_precision,
                    "recall_macro": genre_recall,
                    "f1_macro": genre_f1,
                    "correct_count": sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                }
        
        evaluation_report["genre_level_analysis"] = genre_summaries
    
    return evaluation_report

def clean_text_response(response):
    """Clean model response for GTZAN task, return only the first character as option label"""
    if not response:
        return ""
    resp = response.strip().upper()
    # Return first non-empty character
    for ch in resp:
        if ch in ["A", "B", "C", "D"]:
            return ch
    return resp.split()[0] if resp.split() else ""

def load_gtzan_metadata(metadata_path):
    """Load GTZAN metadata file"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Filter out incomplete entries
    valid_samples = []
    for item in metadata:
        if all(key in item for key in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            valid_samples.append(item)
    
    print(f"Loaded {len(valid_samples)} valid samples from {len(metadata)} entries")
    return valid_samples

def cuda_timing_inference_aero1(model, processor, inputs, max_new_tokens=3):
    """
    Inference function using CUDA Event API for precise GPU timing - Aero-1 version
    """
    
    # Synchronize GPU, prepare for timing
    torch.cuda.synchronize()
    
    # Create CUDA events
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        # === Phase 1: Prefill timing ===
        event_start.record()
        
        # Prefill computation
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, 
                           output_hidden_states=False, return_dict=True)
        
        event_prefill_end.record()
        
        # === Phase 2: Full Generation timing ===
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
        
        # === Synchronize events ===
        event_start.synchronize()
        event_prefill_end.synchronize()
        event_total_end.synchronize()
        
        # === Calculate precise time differences ===
        prefill_time = event_start.elapsed_time(event_prefill_end) / 1000.0
        total_time = event_start.elapsed_time(event_total_end) / 1000.0
        decode_time = event_prefill_end.elapsed_time(event_total_end) / 1000.0
        
        # Decode output
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
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse arguments
    args = parse_arguments()
    
    # Data path configuration
    data_path_root = os.environ.get("GTZAN_DATA_PATH", 
        "/data/to/your/gtzan/concatenated_audio")
    metadata_file = os.path.join(data_path_root, 'music_genre_classification_meta.json')
    
    # Simplify results directory configuration
    results_dir_name = os.environ.get("RESULTS_DIR", "GTZAN_Results")
    if not os.path.isabs(results_dir_name):
        result_dir = os.path.abspath(results_dir_name)
    else:
        result_dir = results_dir_name
    
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    print(f"\n=== GTZAN DART Evaluation Config (Aero-1) ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    # Output file path and naming
    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = os.path.join(result_dir, f'gtzan_results_aero1_dart{sparse_suffix}.json')
    timing_output_file = os.path.join(result_dir, f'gtzan_timing_stats_aero1_dart{sparse_suffix}.json')
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    # Create timing stats object
    timing_stats = GTZANTimingStats()

    # Model path configuration
    model_path = args.model_path
    print(f"Loading Aero-1 model: {model_path}")
    
    # Load Aero-1 processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision="main",
        trust_remote_code=True
    )
    print("Successfully loaded Aero-1 processor")
    
    # Load Aero-1 model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    
    # Configure DART
    configure_DART_aero1(model, args)
    print("Aero-1 model loaded successfully")

    # Load GTZAN metadata
    print(f"Loading GTZAN metadata: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found: {metadata_file}")
        return
    
    samples = load_gtzan_metadata(metadata_file)
    
    # Apply sample limit
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")

    # Count number of each genre
    genre_stats = {}
    for sample in samples:
        genre = sample.get("genre_label", "unknown")
        genre_stats[genre] = genre_stats.get(genre, 0) + 1
    
    print(f"Genre statistics: {genre_stats}")

    # Print initial memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after loading model - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    results = []
    correct_count = 0
    genre_correct = {genre: 0 for genre in genre_stats.keys()}
    genre_total = {genre: 0 for genre in genre_stats.keys()}
    
    # Collect all predictions and ground truths for sklearn evaluation
    all_predictions = []
    all_ground_truths = []
    all_genres = []

    print(f"Starting evaluation for {len(samples)} samples...")
    
    # Detect if running in screen or non-interactive environment
    is_screen_env = not os.sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    # Set tqdm parameters
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': os.sys.stdout
    }

    progress_bar = tqdm(enumerate(samples), total=len(samples), 
                       desc="GTZAN Evaluation(Aero-1+DART)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            # Build audio file path
            audio_rel = sample["path"]
            audio_full = os.path.join(data_path_root, audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"Warning: Audio file not found: {audio_full}")
                continue

            # Count current genre
            current_genre = sample.get("genre_label", "unknown")
            genre_total[current_genre] = genre_total.get(current_genre, 0) + 1

            # Use Aero-1 official message format - supports multiple audio chunks
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Prepare audio input - now returns list of audio chunks
            audio_chunks, sample_rate = prepare_audio_for_processor(audio_full)
            
            # Add each audio chunk as audio content in message
            for chunk in audio_chunks:
                messages[0]["content"].append({
                    "type": "audio",
                    "audio": "placeholder",  # This will be replaced with actual audio
                })

            # Prepare options list
            options = [
                sample["choice_a"],
                sample["choice_b"], 
                sample["choice_c"],
                sample["choice_d"]
            ]

            # Build music genre classification prompt
            instruction = "Listen to this audio segment and identify the music genre based on what you hear."
            format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
            
            # Format options
            formatted_options = ""
            for i, opt in enumerate(options):
                letter = chr(65 + i)  # A, B, C, D...
                formatted_options += f"{letter}. {opt}\n"
            
            # Add text content
            messages[0]["content"].append({
                "type": "text",
                "text": f"{instruction}\n\nQuestion: {sample['question']}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}"
            })
            
            # Use chat template to process messages
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Use processor to process text and audio chunks
            inputs = processor(
                text=prompt,
                audios=audio_chunks,  # Pass list of audio chunks
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to("cuda")
            
            # Get audio special token position info and configure DART
            audio_token_length = 0
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start_index = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end_index = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end_index - audio_token_start_index + 1
                
                # Update DART config and Aero-1 specific config
                if args.sparse:
                    # DART config
                    model.config.DART_config['audio_token_start_index'] = audio_token_start_index
                    model.config.DART_config['audio_token_length'] = audio_token_length
                    
                    # Aero-1 specific config
                    model.config.audio_token_num = audio_token_length
                    model.config.audio_token_start = audio_token_start_index
                
                # Inference with precise CUDA timing
                result = cuda_timing_inference_aero1(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=3
                )
                
                # Get result
                response = result['response_text']
                prefill_time = result['prefill_time']
                decode_time = result['decode_time']
                total_time = result['total_time']
                output_tokens = result['output_tokens']
                
                # Get token count
                input_tokens = inputs['input_ids'].shape[1]
                
                # Clean response
                pred_choice = clean_text_response(response)
                correct_choice = sample["answer_gt"]
                is_correct = pred_choice == correct_choice
                
                # Collect predictions and ground truths for sklearn evaluation
                all_predictions.append(pred_choice)
                all_ground_truths.append(correct_choice)
                all_genres.append(current_genre)
                
                if is_correct:
                    correct_count += 1
                    genre_correct[current_genre] = genre_correct.get(current_genre, 0) + 1
                
                # Record result
                result_record = {
                    "idx": idx,
                    "uniq_id": sample.get("uniq_id", idx),
                    "genre_label": current_genre,
                    "path": audio_rel,
                    "question": sample["question"],
                    "options": options,
                    "predicted_answer": pred_choice,
                    "correct_answer": correct_choice,
                    "correct": is_correct,
                    "response_text": response,
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
                
                results.append(result_record)
                
                # Add timing record (only for first 100 samples, excluding first sample)
                if idx > 0 and idx <= 100:
                    audio_duration = sum(len(chunk) for chunk in audio_chunks) / sample_rate
                    timing_stats.add_record(
                        prefill_time,
                        decode_time,
                        output_tokens,
                        input_tokens,
                        audio_duration=audio_duration,
                        genre=current_genre
                    )
                
                # Memory cleanup
                del inputs
                del audio_chunks
                del result
                torch.cuda.empty_cache()
                
                # Deep cleanup every 10 samples
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Update progress bar
                current_acc = correct_count / (idx + 1) if idx >= 0 else 0
                progress_bar.set_postfix({
                    'Acc': f"{current_acc:.3f}",
                    'Genre': current_genre,
                    'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            if hasattr(args, 'debug') and args.debug:
                traceback.print_exc()
            
            # Add error label for sklearn evaluation
            all_predictions.append("")  # empty string means invalid prediction
            all_ground_truths.append(sample.get("answer_gt", ""))
            all_genres.append(sample.get("genre_label", "unknown"))
            continue

    # Final accuracy and metrics calculation
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    # Per-genre accuracy calculation
    genre_accuracies = {}
    for genre in genre_stats.keys():
        if genre_total.get(genre, 0) > 0:
            genre_accuracies[genre] = genre_correct.get(genre, 0) / genre_total[genre] * 100

    # Music classification metrics
    predictions = [r["predicted_answer"] for r in results]
    ground_truths = [r["correct_answer"] for r in results]
    metrics = calculate_music_metrics(predictions, ground_truths, list(genre_stats.keys()))

    # sklearn GTZAN DART evaluation report (new)
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn GTZAN DART evaluation report ===")
        print(f"Total samples: {len(all_predictions)} (predictions), {len(all_ground_truths)} (ground truth)")
        
        # Generate sklearn GTZAN DART evaluation report
        sklearn_evaluation = generate_sklearn_gtzan_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            genres=all_genres,
            labels=['A', 'B', 'C', 'D']
        )
        
        print(f"sklearn GTZAN DART evaluation report generated:")
        print(f"  Overall accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Macro F1: {sklearn_evaluation['overall_metrics']['f1_macro']:.4f}")
        print(f"  Micro F1: {sklearn_evaluation['overall_metrics']['f1_micro']:.4f}")
        print(f"  Weighted F1: {sklearn_evaluation['overall_metrics']['f1_weighted']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
    else:
        print("Warning: No valid predictions, unable to generate sklearn evaluation report")
        sklearn_evaluation = {"error": "No valid predictions for evaluation"}

    # Create results summary
    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "genre_stats": genre_stats,
        "genre_accuracies": genre_accuracies,
        "genre_correct": genre_correct,
        "genre_total": genre_total,
        "metrics": metrics,
        "sklearn_evaluation": sklearn_evaluation,
        "config": {
            "model_name": "Aero-1-Audio-1.5B",
            "gpu_id": gpu_id,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "data_path": data_path_root
        },
        "timing": timing_stats.get_summary()
    }

    # Save results
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Save timing statistics
    timing_stats.export_to_json(timing_output_file)
    
    print(f"\n=== GTZAN DART Evaluation Results (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    print(f"\nPer-genre accuracy:")
    for genre, acc in genre_accuracies.items():
        correct_num = genre_correct.get(genre, 0)
        total_num = genre_total.get(genre, 0)
        print(f"  {genre}: {acc:.2f}% ({correct_num}/{total_num})")

    # Print sklearn evaluation summary
    if "sklearn_evaluation" in summary and "error" not in summary["sklearn_evaluation"]:
        sklearn_metrics = summary["sklearn_evaluation"]["overall_metrics"]
        
        print(f"\n=== Sklearn Evaluation Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision - Macro: {sklearn_metrics['precision_macro']:.4f}")
        print(f"Recall - Macro: {sklearn_metrics['recall_macro']:.4f}")
        print(f"F1 Score - Macro: {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 Score - Micro: {sklearn_metrics['f1_micro']:.4f}")
        print(f"F1 Score - Weighted: {sklearn_metrics['f1_weighted']:.4f}")
        
        # Print per-option metrics
        print(f"\nPer-option detailed metrics:")
        per_choice_metrics = summary["sklearn_evaluation"]["per_choice_metrics"]
        for choice in sorted(per_choice_metrics.keys()):
            metrics_detail = per_choice_metrics[choice]
            print(f"  Option {choice}: Precision={metrics_detail['precision']:.4f}, "
                  f"Recall={metrics_detail['recall']:.4f}, F1={metrics_detail['f1_score']:.4f}, "
                  f"Support={metrics_detail['support']}")
        
        # If genre-level analysis is available, print top 5 genres
        if "genre_level_analysis" in summary["sklearn_evaluation"]:
            genre_analysis = summary["sklearn_evaluation"]["genre_level_analysis"]
            print(f"\nTop 5 genre-level analysis:")
            sorted_genre_analysis = sorted(genre_analysis.items(), 
                                         key=lambda x: x[1]['accuracy'], reverse=True)
            for genre, analysis in sorted_genre_analysis[:5]:
                print(f"  {genre}: Accuracy={analysis['accuracy']:.4f}, "
                      f"F1={analysis['f1_macro']:.4f}, "
                      f"Sample count={analysis['sample_count']}, "
                      f"Correct count={analysis['correct_count']}")

    timing_summary = timing_stats.get_summary()
    print(f"\n=== Timing statistics ===")
    print(f"Average inference time: {timing_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average Prefill time: {timing_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average Decode time: {timing_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {timing_summary.get('avg_decode_tokens_per_sec', 0):.2f} tokens/sec")

    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()