#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RACE Reading Comprehension DART Evaluation Script
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
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    parser.add_argument("--model-path", type=str, default="microsoft/Phi-4-multimodal-instruct")
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

gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

class RaceTimingStats:
    """Track inference timing statistics for RACE tasks, use CUDA Event for precise measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration):
        """Add a timing record"""
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

def calculate_race_metrics(predictions, ground_truths):
    """Calculate RACE metrics: Accuracy, Precision, Recall, and F1 score"""

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
        'total_samples': len(predictions),
        'label_mapping': label_map
    }

def clean_text_response(response):
    """Clean model's response for RACE, only keep the first character for option label"""
    if not response:
        return ""
    resp = response.strip().upper()

    for ch in resp:
        if ch in ['A', 'B', 'C', 'D']:
            return ch
    return resp.split()[0] if resp.split() else ""

def load_audio_for_race(audio_path, audio_cache=None):
    """
    Load audio file, return format consistent with processor
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

def create_race_prompt(question, options):
    """Create prompt for RACE task"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        formatted_options += f"{chr(65+i)}) {opt}\n"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def main():

    args = parse_arguments()
    
    data_path_root = '/data/to/your/race_audio/path'
    
    results_dir_name = os.environ.get("RESULTS_DIR", "Race_Results")

    if not os.path.isabs(results_dir_name):
        result_dir = os.path.abspath(results_dir_name)
    else:
        result_dir = results_dir_name
    
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    print(f"\n=== RACE DART Evaluation Config ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = os.path.join(result_dir, f'race_results_dart{sparse_suffix}.json')
    timing_output_file = os.path.join(result_dir, f'race_timing_stats_dart{sparse_suffix}.json')
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

    timing_stats = RaceTimingStats()

    model_path = args.model_path
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
        
    generation_config = GenerationConfig.from_pretrained(model_path)

    configure_DART(model, args)
    print("Model loaded successfully")

    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: Benchmark file not found {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f"Sample number limited to: {len(benchmark)}")

    audio_cache = {}
    results = []

    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0
    
    all_predictions = []
    all_ground_truths = []

    print(f"Begin evaluation for {len(benchmark)} samples...")
    
    is_screen_env = not os.sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': os.sys.stdout
    }

    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc="RACE Evaluation", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:

            audio_rel = sample["audio_path"]
            audio_full = os.path.join(data_path_root, audio_rel)
            
            if not os.path.exists(audio_full):

                if idx < 10:
                    print(f"Warning: Audio file does not exist: {audio_full}")
                continue
                
            audio_raw, sr = load_audio_for_race(audio_full, audio_cache)
            
            if audio_raw is None:

                if idx < 5:
                    print(f"Skip sample {idx}: audio loading failed")
                continue
                
            audio = prepare_audio_for_processor(audio_raw[0])
            
            prompt = create_race_prompt(sample["question"], sample["options"])
            
            inputs = processor(
                text=prompt,
                audios=audio,
                return_tensors="pt"
            ).to("cuda")
            
            inputs['input_mode'] = torch.tensor([2])
            
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start_index = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end_index = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end_index - audio_token_start_index + 1
                

                if args.sparse:
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
                correct_choice = sample["answer"]
                is_correct = pred_choice == correct_choice
                

                if pred_choice in ['A', 'B', 'C', 'D'] and correct_choice in ['A', 'B', 'C', 'D']:
                    all_predictions.append(pred_choice)
                    all_ground_truths.append(correct_choice)
                
                if is_correct:
                    correct_count += 1
                

                if "high" in audio_rel:
                    total_high += 1
                    if is_correct:
                        correct_high += 1
                    difficulty = "high"
                elif "middle" in audio_rel:
                    total_middle += 1
                    if is_correct:
                        correct_middle += 1
                    difficulty = "middle"
                else:
                    difficulty = "unknown"
                

                result = {
                    "id": sample.get("id", f"sample_{idx}"),
                    "question": sample["question"],
                    "options": sample["options"],
                    "correct_answer": correct_choice,
                    "predicted_answer": pred_choice,
                    "correct": is_correct,
                    "difficulty": difficulty,
                    "audio_path": sample["audio_path"],
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
                

                current_acc = correct_count / (idx + 1) if idx >= 0 else 0
                progress_bar.set_postfix({
                    'Acc': f"{current_acc:.3f}",
                    'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                })

            del inputs, outputs, generate_ids, audio
            torch.cuda.empty_cache()
            
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:

            if idx % 100 == 0:
                print(f"Error processing sample {idx}: {e}")
            
            for var_name in ['inputs', 'outputs', 'generate_ids', 'audio']:
                try:
                    if var_name in locals():
                        del locals()[var_name]
                except:
                    pass
            torch.cuda.empty_cache()
            continue

    print("Processing timing statistics...")
    for result in results:
        if 'timing' in result and result['audio_path']:
            audio_rel = result['audio_path']
            audio_full = os.path.join(data_path_root, audio_rel)
            if audio_full in audio_cache:
                audio_np, sr = audio_cache[audio_full]
                audio_duration = len(audio_np) / sr
                timing = result['timing']
                timing_stats.add_record(
                    timing['prefill_time'],
                    timing['decode_time'],
                    timing['output_tokens'],
                    timing['input_tokens'],
                    audio_duration=audio_duration
                )
    
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    overall_metrics = calculate_race_metrics(all_predictions, all_ground_truths)

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
        "overall_metrics": overall_metrics,
        "config": {
            "gpu_id": gpu_id,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    timing_stats.export_to_json(timing_output_file)
    
    print(f"\n=== RACE DART Evaluation Results ===")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"F1 Score: {overall_metrics['f1_score']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"Valid samples: {overall_metrics['valid_samples']}/{overall_metrics['total_samples']}")
    print(f"High difficulty Accuracy: {correct_high / total_high * 100 if total_high > 0 else 0:.2f}%")
    print(f"Middle difficulty Accuracy: {correct_middle / total_middle * 100 if total_middle > 0 else 0:.2f}%")

if __name__ == "__main__":
    main()