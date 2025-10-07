#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLUE NER DART Evaluation Script
Integrates DART sparse attention mechanism, supports audio pruning.
"""

import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
import re
import gc
import librosa
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_slue_metrics(all_sample_results):
    """Calculate SLUE NER evaluation metrics"""
    predictions = [result.get('predicted_choice', '') for result in all_sample_results]
    labels = [result.get('ground_truth_choice', '') for result in all_sample_results]
    

    valid_pairs = [(p, t) for p, t in zip(predictions, labels) 
                   if p in ['A', 'B', 'C', 'D'] and t in ['A', 'B', 'C', 'D']]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'weighted_precision': 0.0,
            'weighted_recall': 0.0,
            'weighted_f1': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'valid_samples': 0,
            'total_samples': len(predictions)
        }
    
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    

    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    

    accuracy = accuracy_score(y_true, y_pred)
    

    try:
        weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    

    try:
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except:
        macro_precision = macro_recall = macro_f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions)
    }


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

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

class SLUETimingStats:
    """Track SLUE task inference timing statistics using CUDA Event for precise measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
        """Add a timing record"""
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

def load_slue_dataset(json_file, audio_base_dir):
    """
    Load SLUE task data from JSON file
    
    Args:
        json_file: Path to SLUE format JSON task file
        audio_base_dir: Base directory for audio files
    
    Returns:
        dataset: List containing task data
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
        print(f"Error: Invalid JSON file format, expected a list")
        return []
    
    print(f"Loaded {len(data)} tasks from JSON")
    

    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(data):

        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task missing 'path' key, skipped: {task}")
            continue
        

        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue
        

        task_name = task.get("task_name", "unknown")
        dataset_name = task.get("dataset_name", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        

        choice_a = task.get("choice_a", "")
        choice_b = task.get("choice_b", "")
        choice_c = task.get("choice_c", "")
        choice_d = task.get("choice_d", "")
        

        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"Warning: Unable to get audio info: {full_audio_path}, error: {e}")
            duration = 0
            sample_rate = 16000
        

        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_name": task_name,
            "dataset_name": dataset_name,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "entity_count": task.get("entity_count", 0),
            "entity_types": task.get("entity_types", []),
            "source_count": task.get("source_count", 0),
            "audio_duration_info": task.get("audio_duration_info", ""),
            "source_folder": task.get("source_folder", ""),
            "source_file": task.get("source_file", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1
    
    if missing_files > 5:
        print(f"Warning: {missing_files} audio files do not exist")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")
    return dataset


def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Properly process the audio file as in reference code"""
    try:

        audio_data, orig_sr = sf.read(audio_path)
        

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        

        if orig_sr != target_sr:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        

        audio_data = audio_data.astype(np.float32)
        
        return [(audio_data, target_sr)]
        
    except Exception as e:
        print(f"Audio processing error {audio_path}: {e}")
        return None

def create_slue_prompt(doc):
    """Generate prompt for SLUE format task"""
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
    

    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    

    return f"{user_prompt}<|audio_1|>{prompt_text}{prompt_suffix}{assistant_prompt}"

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

def calculate_slue_metrics(predictions, ground_truths):
    """Calculate SLUE NER metrics: Accuracy, Precision, Recall, and F1 score"""

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

def evaluate_slue_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate SLUE task accuracy"""
    try:
        return predicted_choice.upper() == ground_truth_choice.upper()
    except Exception as e:
        print(f"Evaluation error: {e}")
        return False

def main():

    args = parse_arguments()


    slue_json_file = "/data/to/your/slue/json/merged_audio_data.json"
    audio_base_dir = "/data/to/your/slue/audio"
    
    print(f"SLUE JSON file: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    

    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/slue/results')
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n=== SLUE DART NER Task Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"SLUE JSON file: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)


    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = f'{result_dir}/slue_results_dart{sparse_suffix}.json'
    timing_output_file = f'{result_dir}/slue_timing_stats_dart{sparse_suffix}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")


    timing_stats = SLUETimingStats()


    print("Loading Phi-4-multimodal-instruct model...")
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        revision="33e62acdd07cd7d6635badd529aa0a3467bb9c6a",
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_path)

    generation_config.return_legacy_cache = True
    model.eval()


    configure_DART(model, args)
    print("Model loaded successfully")
    device = "cuda" 
    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Dataset size: {len(samples)} samples")
    

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")


    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    for sample in samples:
        task_type_stats[sample["task_name"]] += 1
        dataset_stats[sample["dataset_name"]] += 1

    print(f"Task type distribution: {dict(task_type_stats)}")
    print(f"Dataset distribution: {dict(dataset_stats)}")


    results = []
    correct_count = 0
    total_count = 0


    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0


    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }


    with tqdm(enumerate(samples), total=len(samples), desc="SLUE Evaluation", **tqdm_kwargs) as pbar:
        for idx, sample in pbar:
            try:

                audio = prepare_audio_for_processor(sample["audio"]["path"])
                if audio is None:
                    continue
                

                prompt = create_slue_prompt(sample)
                

                inputs = processor(
                    text=prompt,
                    audios=audio,
                    return_tensors="pt"
                ).to(device)
                

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
                            max_new_tokens=10, # To do remeasure
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
                    

                    predicted_choice = extract_answer_choice(response)
                    ground_truth_choice = sample["answer_gt"]
                    

                    is_correct = evaluate_slue_accuracy(predicted_choice, ground_truth_choice)
                    
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    

                    result = {
                        "id": sample["id"],
                        "audio_path": sample["path"],
                        "question": sample["question"],
                        "predicted_choice": predicted_choice,
                        "ground_truth_choice": ground_truth_choice,
                        "correct": is_correct,
                        "task_type": sample["task_name"],
                        "dataset": sample["dataset_name"],
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
                        task_type=sample["task_name"]
                    )
                    

                    current_acc = correct_count / total_count if total_count > 0 else 0
                    pbar.set_postfix({
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
                print(f"Error processing sample {idx}: {e}")
                

                for var_name in ['inputs', 'outputs', 'generate_ids', 'audio']:
                    try:
                        if var_name in locals():
                            del locals()[var_name]
                    except:
                        pass
                torch.cuda.empty_cache()
                continue


    final_accuracy = correct_count / total_count if total_count > 0 else 0
    

    slue_metrics = calculate_slue_metrics(results)
    
    timing_summary = timing_stats.get_summary()


    summary = {
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": slue_metrics['accuracy'],
        "slue_metrics": slue_metrics,
        "task_type_stats": dict(task_type_stats),
        "dataset_stats": dict(dataset_stats),
        "config": {
            "gpu_id": gpu_id,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit
        },
        "timing": timing_summary
    }


    final_results = {
        "summary": summary,
        "samples": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")


    timing_stats.export_to_json(timing_output_file)
    print(f"Timing statistics saved to: {timing_output_file}")

    print(f"\n=== SLUE DART Evaluation Results ===")
    print(f"Overall Accuracy: {slue_metrics['accuracy']:.4f}")
    print(f"Weighted F1 Score: {slue_metrics['weighted_f1']:.4f}")
    print(f"Macro F1 Score: {slue_metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {slue_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {slue_metrics['weighted_recall']:.4f}")
    print(f"Macro Precision: {slue_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {slue_metrics['macro_recall']:.4f}")
    print(f"Valid samples: {slue_metrics['valid_samples']}/{slue_metrics['total_samples']}")
    print(f"Processed samples: {total_count}")

if __name__ == "__main__":
    main()