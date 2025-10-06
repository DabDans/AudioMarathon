#!/usr/bin/env python3

import os
import sys
import torch
import json
import argparse
import logging
import time
import warnings
import random
import glob
import re
import pandas as pd
import soundfile as sf
import numpy as np
import librosa
import traceback
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni_dart import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648
_AUDIO_SPECIAL_TOKEN_ID = 200011

random.seed(42)

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
    parser.add_argument('--reduction_ratio', type=float, default=0.3, help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    return parser.parse_args()

def configure_DART(model, args):
    if not hasattr(model.config, 'DART_config'):
        model.config.DART_config = {}
    
    if args.sparse:
        DART_config = {
            "K": args.pruned_layer,
            "sparse": True,
            "enable_dart": True,
            
            "image_token_start_index": args.image_token_start_index, 
            "image_token_length": args.image_token_length,
            
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            
            "reduction_ratio": args.reduction_ratio,
            
            "pivot_image_token": getattr(args, 'pivot_image_token', args.pivot_audio_token),
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            
            "text_length": 1,
            
            "qwen_dart_enabled": True,
            "multimodal_pruning": True,
        }
        
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            model.thinker.model.config.DART_config = DART_config
            print("DART configuration set to thinker.model.config")
        elif hasattr(model, 'model'):
            model.model.config.DART_config = DART_config
            print("DART configuration set to model.config")
        else:
            model.config.DART_config = DART_config
            print("DART configuration set to root config")
        
    else:
        model.config.DART_config = None
    
    print(f"Qwen2.5-Omni DART config: sparse={args.sparse}, "
          f"reduction_ratio={args.reduction_ratio}, "
          f"pruned_layer={args.pruned_layer}")

def calculate_acoustic_metrics(predictions, ground_truths, scene_labels):
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in scene_labels and t in scene_labels]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'valid_samples': 0,
            'total_samples': len(predictions),
            'label_mapping': {}
        }
    
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    
    label_map = {label: idx for idx, label in enumerate(sorted(scene_labels))}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")
print(f"CUDA_VISIBLE_DEVICES: {gpu_temp}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

data_path_root = '/data/to/your/dataset/path/TAU'
audio_dir = os.path.join(data_path_root, 'concatenated_resampled')
result_dir = './TAU_Results'
os.makedirs(result_dir, exist_ok=True)

class TAUTimingStats:
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, scene_label=None):
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "scene_label": scene_label
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        total_samples = len(self.timing_records)
        
        total_times = [record["total_time"] for record in self.timing_records]
        prefill_times = [record["prefill_time"] for record in self.timing_records]
        decode_times = [record["decode_time"] for record in self.timing_records]
        decode_tokens_per_sec = [record["decode_tokens_per_sec"] for record in self.timing_records]
        
        summary = {
            "total_samples": total_samples,
            "avg_total_time": sum(total_times) / total_samples,
            "avg_prefill_time": sum(prefill_times) / total_samples,
            "avg_decode_time": sum(decode_times) / total_samples,
            "avg_decode_tokens_per_sec": sum(decode_tokens_per_sec) / total_samples,
            "prefill_percentage": (sum(prefill_times) / sum(total_times)) * 100 if sum(total_times) > 0 else 0,
            "decode_percentage": (sum(decode_times) / sum(total_times)) * 100 if sum(total_times) > 0 else 0
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
    
    def print_summary(self):
        summary = self.get_summary()
        if "error" not in summary:
            print(f"Average total time: {summary['avg_total_time']:.3f}s")
            print(f"Average prefill time: {summary['avg_prefill_time']:.3f}s")
            print(f"Average decode time: {summary['avg_decode_time']:.3f}s")
            print(f"Average tokens/sec: {summary['avg_decode_tokens_per_sec']:.2f}")

def prepare_audio_for_processor(audio_path, target_sr=16000):
    try:
        audio_data, orig_sr = sf.read(audio_path)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        if orig_sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        
        audio_data = audio_data.astype(np.float32)
        
        return [(audio_data, target_sr)]
        
    except Exception as e:
        print(f"Audio processing error {audio_path}: {e}")
        return None

def load_tau_acoustic_scene_dataset(root_dir):
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    scene_counts = {}
    
    for item in metadata:
        audio_path = os.path.join(root_dir, item["path"])
        
        if os.path.exists(audio_path):
            scene_label = item["scene_label"]
            scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
            
            choices = [
                item["choice_a"],
                item["choice_b"],
                item["choice_c"],
                item["choice_d"]
            ]
            
            sample = {
                "audio_path": audio_path,
                "scene_label": scene_label,
                "choices": choices,
                "correct_answer": item["answer_gt"],
                "id": item.get("uniq_id", f"tau_{len(all_samples)}")
            }
            all_samples.append(sample)
        else:
            print(f"Audio file does not exist: {audio_path}")
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    
    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count}")
    
    if sample_limit > 0 and sample_limit < len(all_samples):
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Sample count limited to: {len(all_samples)}")
        
    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    text_lower = text.lower().strip()
    
    options = ['a', 'b', 'c', 'd']
    
    if text_lower in options:
        return text_lower.upper()
    
    for opt in options:
        if text_lower.startswith(f"{opt}.") or text_lower.startswith(f"{opt})") or text_lower.startswith(f"{opt}:"):
            return opt.upper()
    
    for opt in options:
        patterns = [f"option {opt}", f"choice {opt}", f"answer {opt}", f"({opt})"]
        if any(pattern in text_lower for pattern in patterns):
            return opt.upper()
    
    if choices:
        for i, choice_text in enumerate(choices):
            if choice_text.lower() in text_lower:
                return chr(65 + i)
    
    return ""

def create_tau_prompt(sample):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option."
    
    formatted_options = "Respond with only the letter of your answer (A, B, C, or D).\n"
    for i, choice in enumerate(sample["choices"]):
        formatted_options += f"{chr(65+i)}) {choice}\n"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nOptions:\n{formatted_options.strip()}\n\n{prompt_suffix}{assistant_prompt}"
    
    return prompt
    
    samples = []
    data_path = Path(data_path)
    
    possible_paths = [
        data_path / "TAU-urban-acoustic-scenes-2022-mobile-development",
        data_path / "audio",
        data_path / "evaluation_setup",
        data_path
    ]
    
    audio_path = None
    for path in possible_paths:
        if path.exists() and (path / "audio").exists():
            audio_path = path / "audio"
            break
        elif path.exists() and any(path.glob("*.wav")):
            audio_path = path
            break
    
    if audio_path is None:
        print(f"Warning: Could not find TAU audio directory in {data_path}")
        return create_dummy_tau_samples(sample_limit if sample_limit > 0 else 50)
    
    print(f"Loading TAU data from: {audio_path}")
    
    eval_setup = load_tau_evaluation_setup(data_path)
    
    count = 0
    for audio_file in audio_path.rglob('*.wav'):
        filename = audio_file.name
        
        scene = parse_tau_scene_from_filename(filename)
        
        if filename in eval_setup:
            scene = eval_setup[filename].get('scene_label', scene)
        
        samples.append({
            'audio_path': str(audio_file),
            'filename': filename,
            'scene': scene,
            'task': 'acoustic_scene_classification'
        })
        count += 1
        
        if sample_limit > 0 and count >= sample_limit:
            break
    
    if not samples:
        return create_dummy_tau_samples(sample_limit if sample_limit > 0 else 50)
    
    print(f"Loaded {len(samples)} TAU samples")
    return samples

def load_tau_evaluation_setup(data_path: Path) -> Dict:
    
    eval_data = {}
    
    setup_files = [
        data_path / "evaluation_setup" / "fold1_evaluate.csv",
        data_path / "meta.csv",
        data_path / "evaluation_setup.csv"
    ]
    
    for setup_file in setup_files:
        if setup_file.exists():
            try:
                with open(setup_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    if lines and ('filename' in lines[0].lower() or 'scene' in lines[0].lower()):
                        lines = lines[1:]
                    
                    for line in lines:
                        if '\t' in line:
                            parts = line.strip().split('\t')
                        else:
                            parts = line.strip().split(',')
                        
                        if len(parts) >= 2:
                            filename = parts[0]
                            scene_label = parts[1]
                            
                            eval_data[filename] = {
                                'scene_label': scene_label.strip()
                            }
                break
            except Exception as e:
                print(f"Error reading {setup_file}: {e}")
                continue
    
    print(f"Loaded evaluation setup for {len(eval_data)} TAU samples")
    return eval_data

def parse_tau_scene_from_filename(filename: str) -> str:
    
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) >= 1:
        scene = parts[0]
        
        scene_mapping = {
            'airport': 'airport',
            'bus': 'bus',
            'metro': 'metro_station',
            'metro_station': 'metro_station',
            'park': 'park',
            'public_square': 'public_square',
            'shopping_mall': 'shopping_mall',
            'street_pedestrian': 'street_pedestrian',
            'street_traffic': 'street_traffic',
            'tram': 'tram'
        }
        
        return scene_mapping.get(scene.lower(), scene)
    
    return 'unknown'

def create_dummy_tau_samples(count: int) -> List[Dict]:
    
    samples = []
    scenes = [
        'airport', 'bus', 'metro_station', 'park', 'public_square',
        'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'
    ]
    
    for i in range(count):
        scene = scenes[i % len(scenes)]
        filename = f"{scene}-location{i % 3}-{i:03d}.wav"
        
        samples.append({
            'audio_path': f"/dummy/tau/{filename}",
            'filename': filename,
            'scene': scene,
            'task': 'acoustic_scene_classification'
        })
    
    print(f"Created {len(samples)} dummy TAU samples")
    return samples

def process_tau_sample(sample: Dict, processor, model, timing_stats: TAUTimingStats, device) -> Dict:
    
    try:
        audio_path = sample['audio_path']
        if not os.path.exists(audio_path):
            audio = np.random.randn(16000 * 10)
            sr = 16000
        else:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert in acoustic scene classification and environmental audio analysis. Listen to the audio carefully and identify the acoustic scene or environment. Common urban scenes include: airport, bus, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram."
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "What acoustic scene or environment does this audio represent? Provide only the scene name."}
                ]
            }
        ]
        
        start_event, end_event = timing_stats.start_timing()
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        prefill_duration = timing_stats.end_timing(start_event, end_event)
        timing_stats.record_prefill(prefill_duration)
        
        start_event, end_event = timing_stats.start_timing()
        
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        
        decode_duration = timing_stats.end_timing(start_event, end_event)
        timing_stats.record_decode(decode_duration)
        
        total_duration = prefill_duration + decode_duration
        timing_stats.record_total(total_duration)
        
        response = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].size(1):], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        predicted_scene = parse_scene_response(response)
        
        result = {
            'filename': sample['filename'],
            'audio_path': sample['audio_path'],
            'true_scene': sample['scene'],
            'predicted_scene': predicted_scene,
            'raw_response': response,
            'correct': predicted_scene.lower() == sample['scene'].lower(),
            'processing_time': total_duration
        }
        
        del inputs, generate_ids
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        print(f"Error processing sample {sample.get('filename', 'unknown')}: {e}")
        return {
            'filename': sample.get('filename', 'unknown'),
            'audio_path': sample.get('audio_path', ''),
            'true_scene': sample.get('scene', ''),
            'predicted_scene': 'error',
            'raw_response': f"Error: {e}",
            'correct': False,
            'processing_time': 0.0
        }

def parse_scene_response(response: str) -> str:
    
    response = response.lower().strip()
    
    tau_scenes = [
        'airport', 'bus', 'metro_station', 'park', 'public_square',
        'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'
    ]
    
    scene_aliases = {
        'metro': 'metro_station',
        'subway': 'metro_station',
        'train': 'tram',
        'mall': 'shopping_mall',
        'shop': 'shopping_mall',
        'street': 'street_traffic',
        'road': 'street_traffic',
        'traffic': 'street_traffic',
        'pedestrian': 'street_pedestrian',
        'walking': 'street_pedestrian',
        'square': 'public_square',
        'plaza': 'public_square'
    }
    
    for scene in tau_scenes:
        if scene in response:
            return scene
    
    for alias, scene in scene_aliases.items():
        if alias in response:
            return scene
    
    words = response.split()
    if words:
        first_word = words[0].lower()
        if first_word in tau_scenes:
            return first_word
        elif first_word in scene_aliases:
            return scene_aliases[first_word]
    
    return 'unknown'

def evaluate_tau_results(results: List[Dict]) -> Dict:
    
    valid_results = [r for r in results if r['predicted_scene'] != 'error']
    
    if not valid_results:
        return {
            'accuracy': 0.0,
            'total_samples': len(results),
            'valid_samples': 0,
            'error_rate': 1.0
        }
    
    correct_predictions = sum(1 for r in valid_results if r['correct'])
    accuracy = correct_predictions / len(valid_results)
    
    scene_stats = {}
    all_scenes = set(r['true_scene'] for r in valid_results)
    
    for scene in all_scenes:
        scene_results = [r for r in valid_results if r['true_scene'] == scene]
        scene_correct = sum(1 for r in scene_results if r['correct'])
        scene_stats[scene] = {
            'total': len(scene_results),
            'correct': scene_correct,
            'accuracy': scene_correct / len(scene_results) if scene_results else 0.0
        }
    
    processing_times = [r['processing_time'] for r in valid_results if r['processing_time'] > 0]
    
    evaluation = {
        'accuracy': accuracy,
        'total_samples': len(results),
        'valid_samples': len(valid_results),
        'correct_predictions': correct_predictions,
        'error_rate': (len(results) - len(valid_results)) / len(results) if results else 0,
        'scene_statistics': scene_stats,
        'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
        'unique_scenes': len(all_scenes)
    }
    
    return evaluation

def save_tau_results(results: List[Dict], evaluation: Dict, timing_stats: Dict, 
                     dart_config: Dict, output_path: str):
    
    output_data = {
        'task': 'acoustic_scene_classification',
        'dataset': 'TAU_Urban_Acoustic_Scenes',
        'model': 'Qwen2.5-Omni-3B',
        'dart_config': dart_config,
        'summary': evaluation,
        'timing_stats': timing_stats,
        'samples': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def calculate_acoustic_metrics(predictions, ground_truths, labels):
    
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "unknown"]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    
    if not valid_predictions:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "valid_samples": 0,
            "total_samples": len(predictions)
        }
    
    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_ground_truths, valid_predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "valid_samples": len(valid_predictions),
        "total_samples": len(predictions),
        "classification_report": classification_report(
            valid_ground_truths, valid_predictions, labels=labels, zero_division=0
        )
    }

def main():
    args = parse_arguments()
    
    print(f"\n=== TAU DART Acoustic Scene Classification Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"Data directory: {audio_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)

    if args.sparse:
        method_name = "sparse"
        ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    else:
        method_name = "base"
        ratio_str = "ratio_1.000"
    
    output_file = f'{result_dir}/TAU_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/TAU_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

    print("Loading Qwen2.5-Omni model...")
    model_path = "/data/to/your/Qwen_2.5Omni-3B/Model/folder"
    device_map = {"": 0}
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.disable_talker()
    
    configure_DART(model, args)
    print("Model loaded successfully")
    
    timing_stats = TAUTimingStats()
    
    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    
    print(f"Total processing {len(samples)} samples")
    
    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    
    scene_stats = {scene: {"total": 0, "correct": 0} for scene in scene_counts}
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    print(f"Starting to process {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing TAU acoustic scene samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            
            instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.\n"
            instruction += f"A: {sample['choices'][0]}\nB: {sample['choices'][1]}\nC: {sample['choices'][2]}\nD: {sample['choices'][3]}\n"
            instruction += "Respond with only the letter of your answer (A, B, C, or D)."

            qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            task_prompt = "You are a helpful assistant that analyzes urban soundscape audio to identify acoustic scenes. Please listen to the audio carefully and classify the scene type."
            sys_prompt = f"{qwen_intro} {task_prompt}"

            audio_path_for_inference = sample["audio_path"]

            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path_for_inference},
                    {"type": "text", "text": instruction},
                ]}
            ]

            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if isinstance(text, list):
                text = text[0]

            inputs = processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=True
            )
            inputs = inputs.to(model.device).to(model.dtype)

            audio_token_length = 0
            audio_token_start = 0
            input_token_length = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0

            audio_detected = False

            if hasattr(inputs, 'input_ids'):
                token_ids = inputs.input_ids[0].tolist()
                if _AUDIO_BOS_TOKEN_ID in token_ids and _AUDIO_EOS_TOKEN_ID in token_ids:
                    audio_start = token_ids.index(_AUDIO_BOS_TOKEN_ID)
                    audio_end = token_ids.index(_AUDIO_EOS_TOKEN_ID)
                    audio_token_start = audio_start
                    audio_token_length = audio_end - audio_start + 1
                    audio_detected = True

            if not audio_detected:
                audio_token_start = args.audio_token_start_index
                audio_token_length = args.audio_token_length

            if args.sparse:
                args.audio_token_start_index = audio_token_start
                args.audio_token_length = audio_token_length
                configure_DART(model, args)

            full_start_event = torch.cuda.Event(enable_timing=True)
            full_end_event = torch.cuda.Event(enable_timing=True)
            
            first_token_start_event = torch.cuda.Event(enable_timing=True)
            first_token_end_event = torch.cuda.Event(enable_timing=True)

            full_start_event.record()
            
            first_token_start_event.record()
            with torch.no_grad():
                first_token_output = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
            first_token_end_event.record()
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
            full_end_event.record()

            torch.cuda.synchronize()
            first_token_time = first_token_start_event.elapsed_time(first_token_end_event) / 1000.0
            total_time = full_start_event.elapsed_time(full_end_event) / 1000.0
            
            prefill_time = first_token_time
            decode_time = max(0.0, total_time - prefill_time)

            output_text = processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            if "assistant\n" in output_text:
                output_text = output_text.split("assistant\n")[-1].strip()

            if hasattr(output, 'shape') and len(output.shape) > 1:
                output_tokens = output.shape[1] - inputs.input_ids.shape[1]
            else:
                output_tokens = len(output_text.split())

            output_text = output_text.strip()

            predicted_answer = extract_acoustic_scene_answer(output_text, sample['choices'])
            ground_truth = sample["correct_answer"].upper()

            is_correct = (predicted_answer == ground_truth)

            all_predictions.append(predicted_answer if predicted_answer else "ERROR")
            all_ground_truths.append(ground_truth)

            scene_label = sample["scene_label"]
            scene_stats[scene_label]["total"] += 1
            if is_correct:
                scene_stats[scene_label]["correct"] += 1

            timing_stats.add_record(prefill_time, decode_time, output_tokens, input_token_length, scene_label)

            sample_result = {
                "audio_file": os.path.basename(sample["audio_path"]),
                "scene_label": scene_label,
                "ground_truth": ground_truth,
                "model_output": output_text,
                "extracted_answer": predicted_answer,
                "is_correct": is_correct,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "total_time": total_time
            }

            all_sample_results.append(sample_result)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            pbar.update(1)

    all_scene_labels = list(set(all_ground_truths))
    acoustic_metrics = calculate_acoustic_metrics(all_predictions, all_ground_truths, all_scene_labels)
    final_stats = timing_stats.get_summary()
    
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    
    scene_accuracies = {}
    for scene, stats in scene_stats.items():
        if stats["total"] > 0:
            scene_accuracies[scene] = stats["correct"] / stats["total"]
        else:
            scene_accuracies[scene] = 0.0

    final_results = {
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "overall_accuracy": correct_samples / total_samples if total_samples > 0 else 0,
            "scene_accuracies": scene_accuracies,
            "scene_stats": scene_stats,
            "acoustic_metrics": acoustic_metrics,
            "config": {
                "gpu_id": gpu_id,
                "sparse": args.sparse,
                "pruned_layer": args.pruned_layer,
                "reduction_ratio": args.reduction_ratio,
                "sample_limit": sample_limit
            },
            "timing": final_stats
        },
        "samples": all_sample_results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

    timing_stats.export_to_json(timing_output_file)
    print(f"Timing stats saved to: {timing_output_file}")

    print(f"\n=== TAU DART Evaluation Results ===")
    print(f"Overall accuracy: {acoustic_metrics['accuracy']:.4f}")
    print(f"F1 Score: {acoustic_metrics['f1_score']:.4f}")
    print(f"Precision: {acoustic_metrics['precision']:.4f}")
    print(f"Recall: {acoustic_metrics['recall']:.4f}")
    print(f"Valid samples: {acoustic_metrics['valid_samples']}/{acoustic_metrics['total_samples']}")

if __name__ == "__main__":
    main()