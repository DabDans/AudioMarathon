#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GTZAN Music Genre Classification DART Version Evaluation Script
Integrated DART sparse attention mechanism, supports audio pruning
"""

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
from transformers import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

_AUDIO_SPECIAL_TOKEN_ID = 200011

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
            "pivot_image_token": args.pivot_image_token,
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            "text_length": 1,
        }
        model.config.DART_config = DART_config
    else:
        model.config.DART_config = None

gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

class GTZANTimingStats:
    """Track inference timing statistics for GTZAN task, using CUDA Event for precision"""
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
    """Calculate music genre classification metrics: Accuracy, Precision, Recall and F1 score"""

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
        'total_samples': len(predictions),
        'label_mapping': label_map
    }

def clean_text_response(response):
    """Clean model response for GTZAN task, keep only the first character as option label"""
    if not response:
        return ""
    resp = response.strip().upper()

    for ch in resp:
        if ch in ["A", "B", "C", "D"]:
            return ch
    return resp.split()[0] if resp.split() else ""

def load_audio_for_gtzan(audio_path, audio_cache=None):
    """
    Load audio file, return format matching processor
    Return: ([audio_array], sampling_rate)
    """
    if audio_cache is not None and audio_path in audio_cache:
        return audio_cache[audio_path]
    else:
        try:
            audio_data, sr = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                sr = 16000
            result = ([audio_data], sr)
            if audio_cache is not None:
                audio_cache[audio_path] = result
            return result
        except Exception as e:
            print(f"Failed to load audio {audio_path}: {e}")
            return None

def prepare_audio_for_processor(audio_data, target_sr=16000):
    """Convert audio to format expected by processor"""
    if isinstance(audio_data, list):
        return [(audio, target_sr) for audio in audio_data]
    else:
        return [(audio_data, target_sr)]

def create_gtzan_prompt(question, options):
    """Create prompt for GTZAN task"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio segment and identify the music genre based on what you hear."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options += f"{letter}. {opt}\n"
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    return prompt

def load_gtzan_metadata(metadata_path):
    """Load GTZAN metadata file"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    valid_samples = []
    for item in metadata:
        if all(key in item for key in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            valid_samples.append(item)
    print(f"Loaded {len(valid_samples)} valid samples from {len(metadata)} entries")
    return valid_samples

def main():
    args = parse_arguments()
    data_path_root = os.environ.get("GTZAN_DATA_PATH", 
        "/data/to/your/gtzan_audio/path")
    metadata_file = "/data/to/your/gtzan_metadata/path/music_genre_classification_meta.json"
    results_dir_name = os.environ.get("RESULTS_DIR", "GTZAN_Results")
    if not os.path.isabs(results_dir_name):
        result_dir = os.path.abspath(results_dir_name)
    else:
        result_dir = results_dir_name
    os.makedirs(result_dir, exist_ok=True)
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    print(f"\n=== GTZAN DART Evaluation Config ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = os.path.join(result_dir, f'gtzan_results_dart{sparse_suffix}.json')
    timing_output_file = os.path.join(result_dir, f'gtzan_timing_stats_dart{sparse_suffix}.json')
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = GTZANTimingStats()
    model_path = "microsoft/Phi-4-multimodal-instruct"
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="balanced_low_0",
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    configure_DART(model, args)
    print("Model loaded successfully")
    generation_config = GenerationConfig.from_pretrained(model_path)

    print(f"Loading GTZAN metadata: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file does not exist: {metadata_file}")
        return
    samples = load_gtzan_metadata(metadata_file)
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample number limited to: {len(samples)}")

    genre_stats = {}
    for sample in samples:
        genre = sample.get("genre_label", "unknown")
        genre_stats[genre] = genre_stats.get(genre, 0) + 1
    print(f"Genre statistics: {genre_stats}")

    audio_cache = {}
    results = []
    correct_count = 0
    genre_correct = {genre: 0 for genre in genre_stats.keys()}
    genre_total = {genre: 0 for genre in genre_stats.keys()}

    print(f"Starting evaluation for {len(samples)} samples...")

    is_screen_env = not os.sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': os.sys.stdout
    }

    progress_bar = tqdm(enumerate(samples), total=len(samples), 
                       desc="GTZAN Evaluation", **tqdm_kwargs)
    for idx, sample in progress_bar:
        audio_rel = sample["path"]
        audio_full = os.path.join(data_path_root, audio_rel)
        if not os.path.exists(audio_full):
            print(f"Warning: Audio file does not exist: {audio_full}")
            continue
        audio_raw = load_audio_for_gtzan(audio_full, audio_cache)
        if audio_raw is None:
            print(f"Skipping sample {idx}: Audio load failed")
            continue
        audio = prepare_audio_for_processor(audio_raw[0])
        options = [
            sample["choice_a"],
            sample["choice_b"], 
            sample["choice_c"],
            sample["choice_d"]
        ]
        prompt = create_gtzan_prompt(sample['question'], options)
        current_genre = sample.get("genre_label", "unknown")
        genre_total[current_genre] = genre_total.get(current_genre, 0) + 1
        inputs = processor(
            text=prompt,
            audios=audio,
            return_tensors="pt"
        ).to("cuda")
        inputs['input_mode'] = torch.tensor([2])
        audio_token_length = 0
        if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
            token_ids = inputs.input_ids[0].tolist()
            audio_token_start_index = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
            rev_ids = token_ids[::-1]
            audio_token_end_index = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
            audio_token_length = audio_token_end_index - audio_token_start_index + 1
            print(audio_token_start_index)
            print(audio_token_length)
            print(audio_token_end_index)
            model.config.DART_config['audio_token_start_index'] = audio_token_start_index
            model.config.DART_config['audio_token_length'] = audio_token_length
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                prefill_start_event = torch.cuda.Event(enable_timing=True)
                prefill_end_event = torch.cuda.Event(enable_timing=True)
                generation_start_event = torch.cuda.Event(enable_timing=True)
                generation_end_event = torch.cuda.Event(enable_timing=True)
                prefill_start_event.record()
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True
                    )
                prefill_end_event.record()
                generation_start_event.record()
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    generation_config=generation_config,
                    do_sample=False,
                    use_cache=True
                )
                generation_end_event.record()
                torch.cuda.synchronize()
                prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
                full_generation_time = generation_start_event.elapsed_time(generation_end_event) / 1000.0
                decode_time = full_generation_time - prefill_time
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = generate_ids.shape[1] - input_tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            pred_choice = clean_text_response(response)
            correct_choice = sample["answer_gt"]
            is_correct = pred_choice == correct_choice
            if is_correct:
                correct_count += 1
                genre_correct[current_genre] = genre_correct.get(current_genre, 0) + 1
            result = {
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
                audio_duration=sample.get("duration", 0),
                genre=current_genre
            )
            current_acc = correct_count / (idx + 1) if idx >= 0 else 0
            progress_bar.set_postfix({
                'Acc': f"{current_acc:.3f}",
                'Genre': current_genre,
                'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
            })
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0
    genre_accuracies = {}
    for genre in genre_stats.keys():
        if genre_total.get(genre, 0) > 0:
            genre_accuracies[genre] = genre_correct.get(genre, 0) / genre_total[genre] * 100
    predictions = [r["predicted_answer"] for r in results]
    ground_truths = [r["correct_answer"] for r in results]
    metrics = calculate_music_metrics(predictions, ground_truths, list(genre_stats.keys()))
    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "genre_stats": genre_stats,
        "genre_accuracies": genre_accuracies,
        "genre_correct": genre_correct,
        "genre_total": genre_total,
        "metrics": metrics,
        "config": {
            "gpu_id": gpu_id,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "data_path": data_path_root
        },
        "timing": timing_stats.get_summary()
    }
    final_results = {
        "summary": summary,
        "samples": results
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    timing_stats.export_to_json(timing_output_file)
    print(f"\n=== GTZAN DART Evaluation Results ===")
    print(f"Total Accuracy: {overall_acc:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"\nAccuracy by genre:")
    for genre, acc in genre_accuracies.items():
        correct_num = genre_correct.get(genre, 0)
        total_num = genre_total.get(genre, 0)
        print(f"  {genre}: {acc:.2f}% ({correct_num}/{total_num})")

if __name__ == "__main__":
    main()