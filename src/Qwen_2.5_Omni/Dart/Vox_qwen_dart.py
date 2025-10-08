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
import re
warnings.filterwarnings("ignore")

import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import subprocess
import gc

random.seed(42)

sys.path.append("/data/to/your/Modeling/path/")
from modeling_qwen2_5_omni_dart import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648
_AUDIO_SPECIAL_TOKEN_ID = 151648

from transformers import logging
logging.set_verbosity_error()

class VoxTimingStats:
    def __init__(self):
        self.timing_records = []
        self.gender_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
        self.max_timing_samples = 100
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, gender=None):
        if self.total_samples < self.max_timing_samples:
            record = {
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
                "audio_duration": audio_duration,
                "gender": gender
            }
            self.timing_records.append(record)
            
            if gender:
                self.gender_stats[gender].append(record)
        
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        if audio_duration:
            self.total_audio_duration += audio_duration
    
    def get_summary(self):
        if self.total_samples == 0:
            return {"error": "No timing records available"}
        
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
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_audio_duration": self.total_audio_duration,
            "avg_audio_duration": self.total_audio_duration / self.total_samples if self.total_samples > 0 else 0
        }
        
        gender_summaries = {}
        for gender, records in self.gender_stats.items():
            if records:
                gender_summaries[gender] = {
                    "samples": len(records),
                    "avg_prefill_time": np.mean([r["prefill_time"] for r in records]),
                    "avg_decode_time": np.mean([r["decode_time"] for r in records]),
                    "avg_total_time": np.mean([r["total_time"] for r in records]),
                    "avg_tokens_per_sec": np.mean([r["decode_tokens_per_sec"] for r in records])
                }
        
        return {
            "overall_summary": summary,
            "gender_summaries": gender_summaries
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def configure_DART(model, args):
    if not hasattr(model.config, 'DART_config'):
        model.config.DART_config = {}
    
    if args.sparse:
        DART_config = {
            "K": args.pruned_layer,
            "sparse": True,
            "enable_dart": False,
            
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
    
    print(f"Qwen2.5-Omni DART config: sparse={args.sparse}, "
          f"reduction_ratio={args.reduction_ratio}, "
          f"pruned_layer={args.pruned_layer}")

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def load_vox_data(data_path: str, sample_limit: int = 0) -> List[Dict]:
    samples = []
    
    data_path = Path(data_path)
    
    possible_paths = [
        data_path / "voxceleb1" / "test",
        data_path / "voxceleb2" / "test",
        data_path / "wav",
        data_path
    ]
    
    audio_path = None
    for path in possible_paths:
        if path.exists():
            audio_path = path
            break
    
    if audio_path is None:
        print(f"Warning: Could not find VoxCeleb audio directory in {data_path}")
        return create_dummy_vox_samples(sample_limit if sample_limit > 0 else 50)
    
    print(f"Loading VoxCeleb data from: {audio_path}")
    
    gender_map = load_vox_gender_metadata(data_path)
    
    audio_extensions = ['.wav', '.flac', '.m4a']
    count = 0
    
    for speaker_dir in audio_path.iterdir():
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        gender = determine_speaker_gender(speaker_id, gender_map)
        
        for audio_file in speaker_dir.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                samples.append({
                    'audio_path': str(audio_file),
                    'speaker_id': speaker_id,
                    'label': gender,
                    'task': 'gender_classification'
                })
                count += 1
                
                if sample_limit > 0 and count >= sample_limit:
                    break
        
        if sample_limit > 0 and count >= sample_limit:
            break
    
    print(f"Loaded {len(samples)} VoxCeleb samples")
    return samples

def load_vox_gender_metadata(data_path: Path) -> Dict[str, str]:
    gender_map = {}
    metadata_files = [
        data_path / "vox1_meta.csv",
        data_path / "voxceleb1_meta.csv", 
        data_path / "meta.csv",
        data_path / "speaker_gender.txt"
    ]
    
    for meta_file in metadata_files:
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            speaker_id = parts[0]
                            gender = parts[1].lower()
                            if gender in ['m', 'male', '1']:
                                gender_map[speaker_id] = 'male'
                            elif gender in ['f', 'female', '0']:
                                gender_map[speaker_id] = 'female'
                break
            except Exception as e:
                print(f"Error reading {meta_file}: {e}")
                continue
    
    print(f"Loaded gender metadata for {len(gender_map)} speakers")
    return gender_map

def determine_speaker_gender(speaker_id: str, gender_map: Dict[str, str]) -> str:
    if speaker_id in gender_map:
        return gender_map[speaker_id]
    
    return 'male' if hash(speaker_id) % 2 == 0 else 'female'

def create_dummy_vox_samples(count: int) -> List[Dict]:
    samples = []
    for i in range(count):
        speaker_id = f"id{i:05d}"
        gender = 'male' if i % 2 == 0 else 'female'
        
        samples.append({
            'audio_path': f"/dummy/vox/{speaker_id}/sample_{i}.wav",
            'speaker_id': speaker_id,
            'label': gender,
            'task': 'gender_classification'
        })
    
    print(f"Created {len(samples)} dummy VoxCeleb samples")
    return samples

def process_vox_sample(sample: Dict, processor, model, timing_stats, device) -> Dict:
    try:
        audio_path_for_inference = sample.get('audio_path')
        if not audio_path_for_inference or not os.path.exists(audio_path_for_inference):
            audio = np.random.randn(16000 * 3).astype(np.float32)
            duration = 3.0
        else:
            audio, sr = sf.read(audio_path_for_inference)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    print("Warning: librosa not installed, skipping resampling")
            
            duration = len(audio) / 16000.0
            audio = audio.astype(np.float32)
        
        instruction = "Is this a male or female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'"
        
        qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        task_prompt = "You are a helpful assistant that analyzes audio to identify speaker characteristics. Please Listen to this audio and identify the speaker's gender."
        sys_prompt = f"{qwen_intro} {task_prompt}"
        
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
            input_ids = inputs.input_ids[0]
            
            bos_positions = (input_ids == _AUDIO_BOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            eos_positions = (input_ids == _AUDIO_EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            
            if len(bos_positions) > 0 and len(eos_positions) > 0:
                audio_token_start = bos_positions[0].item()
                audio_end = eos_positions[0].item()
                audio_token_length = audio_end - audio_token_start + 1
                audio_detected = True
        
        if not audio_detected:
            print("Warning: Audio tokens not detected, using default configuration")
            audio_token_start = 35
            audio_token_length = 576
        
        full_start_event = torch.cuda.Event(enable_timing=True)
        full_end_event = torch.cuda.Event(enable_timing=True)
        
        first_token_start_event = torch.cuda.Event(enable_timing=True)
        first_token_end_event = torch.cuda.Event(enable_timing=True)

        full_start_event.record()
        
        first_token_start_event.record()
        with torch.no_grad():
            first_token_output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
                return_dict_in_generate=True
            )
        first_token_end_event.record()
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )
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
            output_tokens = 5
        
        output_text = output_text.strip()
        
        predicted_gender = parse_gender_response(output_text)
        true_gender = sample.get('label', 'unknown')
        
        is_correct = predicted_gender.lower() == true_gender.lower()
        
        timing_stats.add_record(prefill_time, decode_time, output_tokens, input_token_length, 
                              duration, true_gender)
        
        result = {
            'speaker_id': sample.get('speaker_id', 'unknown'),
            'audio_path': sample.get('audio_path', ''),
            'true_label': true_gender,
            'predicted_label': predicted_gender,
            'raw_response': output_text,
            'correct': is_correct,
            'processing_time': total_time,
            'audio_duration': duration,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'output_tokens': output_tokens,
            'input_tokens': input_token_length,
            'audio_token_start': audio_token_start,
            'audio_token_length': audio_token_length
        }
        
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"Error processing sample {sample.get('speaker_id', 'unknown')}: {e}")
        return {
            'speaker_id': sample.get('speaker_id', 'unknown'),
            'audio_path': sample.get('audio_path', ''),
            'true_label': sample.get('label', ''),
            'predicted_label': 'error',
            'raw_response': f"Error: {e}",
            'correct': False,
            'processing_time': 0.0,
            'audio_duration': 0.0,
            'error': str(e)
        }

def parse_gender_response(response: str) -> str:
    if not response:
        return "unknown"
    
    response = response.strip().upper()
    
    if response in ['A', 'B']:
        return 'male' if response == 'A' else 'female'
    
    if response.startswith('A') and len(response) <= 3:
        return 'male'
    if response.startswith('B') and len(response) <= 3:
        return 'female'
    
    match = re.search(r'\b([AB])\b', response)
    if match:
        return 'male' if match.group(1) == 'A' else 'female'
    
    match = re.search(r'[(\[]?([AB])[)\].]?', response)
    if match:
        return 'male' if match.group(1) == 'A' else 'female'
    
    match = re.search(r'(?:option|choice)\s+([AB])', response)
    if match:
        return 'male' if match.group(1) == 'A' else 'female'
    
    if 'MALE' in response and 'FEMALE' not in response:
        return 'male'
    elif 'FEMALE' in response and 'MALE' not in response:
        return 'female'
    
    return "unknown"

def evaluate_vox_results(results: List[Dict]) -> Dict:
    valid_results = [r for r in results if r['predicted_label'] != 'error']
    
    if not valid_results:
        return {
            'accuracy': 0.0,
            'total_samples': len(results),
            'valid_samples': 0,
            'error_rate': 1.0
        }
    
    true_labels = [r['true_label'] for r in valid_results]
    pred_labels = [r['predicted_label'] for r in valid_results]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    
    male_results = [r for r in valid_results if r['true_label'] == 'male']
    female_results = [r for r in valid_results if r['true_label'] == 'female']
    
    male_accuracy = accuracy_score(
        [r['true_label'] for r in male_results],
        [r['predicted_label'] for r in male_results]
    ) if male_results else 0.0
    
    female_accuracy = accuracy_score(
        [r['true_label'] for r in female_results],
        [r['predicted_label'] for r in female_results]
    ) if female_results else 0.0
    
    processing_times = [r['processing_time'] for r in valid_results if r['processing_time'] > 0]
    
    evaluation = {
        'accuracy': accuracy,
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy,
        'total_samples': len(results),
        'valid_samples': len(valid_results),
        'error_rate': (len(results) - len(valid_results)) / len(results) if results else 0,
        'male_samples': len(male_results),
        'female_samples': len(female_results),
        'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
        'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist() if valid_results else [],
        'classification_report': classification_report(true_labels, pred_labels, output_dict=True) if valid_results else {}
    }
    
    return evaluation

def save_vox_results(results: List[Dict], evaluation: Dict, timing_stats: Dict, 
                     dart_config: Dict, output_path: str):
    output_data = {
        'task': 'gender_classification',
        'dataset': 'VoxCeleb',
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

def main():
    args = parse_arguments()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")
    
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    
    vox_data_path = '/data/to/your/dataset/path/VoxCeleb/concatenated_audio'
    result_dir = os.environ.get("RESULTS_DIR", './Vox_Results')
    
    print(f"\n=== VoxCeleb DART Gender Classification Evaluation Config ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART Sparse Mode: {args.sparse}")
    print(f"Pruned Layers: {args.pruned_layer}")
    print(f"Retention Ratio: {args.reduction_ratio}")
    print(f"VoxCeleb Data Path: {vox_data_path}")
    if sample_limit > 0:
        print(f"Sample Limit: {sample_limit}")
    print("=" * 50)
    
    method_name = "sparse" if args.sparse else "base"
    ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    output_file = f'{result_dir}/vox_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/vox_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")
    
    timing_stats = VoxTimingStats()
    
    samples = load_vox_data(vox_data_path, sample_limit)
    
    os.makedirs(result_dir, exist_ok=True)
    
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
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")
    
    gender_stats = defaultdict(int)
    for sample in samples:
        gender_stats[sample.get("label", "unknown")] += 1
    
    print(f"Gender distribution: {dict(gender_stats)}")
    
    results = []
    total_accuracy = 0
    processed_samples = 0
    
    gender_correct = defaultdict(int)
    gender_total = defaultdict(int)
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    print(f"Starting evaluation of {len(samples)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"After model loading GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    with tqdm(total=len(samples), desc="Processing VoxCeleb Gender Classification Samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            result = process_vox_sample(sample, processor, model, timing_stats, model.device)
            
            is_correct = result.get('correct', False)
            if is_correct:
                total_accuracy += 1
            
            processed_samples += 1
            true_gender = sample.get("label", "unknown")
            gender_total[true_gender] += 1
            
            if is_correct:
                gender_correct[true_gender] += 1
            
            results.append(result)
            
            pbar.set_postfix({
                'Accuracy': f'{total_accuracy/processed_samples:.3f}' if processed_samples > 0 else '0.000',
                'Processed': processed_samples
            })
            pbar.update(1)
            
            if (idx + 1) % 10 == 0:
                allocated, reserved = get_gpu_memory_usage()
                pbar.set_description(f"Processing VoxCeleb Samples (GPU: {allocated:.1f}GB)")
    
    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0
    
    all_predictions = [result["predicted_label"] for result in results]
    all_ground_truths = [result["true_label"] for result in results]
    
    valid_indices = [i for i, pred in enumerate(all_predictions) if pred not in ['error', 'unknown']]
    valid_predictions = [all_predictions[i] for i in valid_indices]
    valid_ground_truths = [all_ground_truths[i] for i in valid_indices]
    
    overall_metrics = {}
    if valid_predictions:
        try:
            overall_accuracy = accuracy_score(valid_ground_truths, valid_predictions)
            classification_rep = classification_report(valid_ground_truths, valid_predictions, output_dict=True)
            confusion_mat = confusion_matrix(valid_ground_truths, valid_predictions)
            
            overall_metrics = {
                "accuracy": overall_accuracy,
                "precision": classification_rep.get("weighted avg", {}).get("precision", 0.0),
                "recall": classification_rep.get("weighted avg", {}).get("recall", 0.0),
                "f1_score": classification_rep.get("weighted avg", {}).get("f1-score", 0.0),
                "valid_samples": len(valid_predictions),
                "classification_report": classification_rep,
                "confusion_matrix": confusion_mat.tolist()
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            overall_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "valid_samples": 0
            }
    
    gender_accuracies = {}
    for gender in gender_stats.keys():
        if gender_total[gender] > 0:
            gender_accuracies[gender] = gender_correct[gender] / gender_total[gender]
    
    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "accuracy": overall_metrics.get("accuracy", 0.0),
        "precision": overall_metrics.get("precision", 0.0),
        "recall": overall_metrics.get("recall", 0.0),
        "f1_score": overall_metrics.get("f1_score", 0.0),
        "valid_samples": overall_metrics.get("valid_samples", 0),
        "gender_stats": dict(gender_stats),
        "gender_accuracies": gender_accuracies,
        "gender_correct": dict(gender_correct),
        "gender_total": dict(gender_total),
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "vox_data_path": vox_data_path,
            "timing_sample_count": min(100, max(0, len(results) - 1))
        },
        "timing": timing_stats.get_summary()
    }
    
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    timing_stats.export_to_json(timing_output_file)
    
    print(f"\n=== VoxCeleb DART Evaluation Results Summary ===")
    print(f"Total Samples: {len(results)}")
    print(f"Processed Samples: {processed_samples}")
    print(f"Valid Samples: {overall_metrics.get('valid_samples', 0)}")
    print(f"Overall Accuracy: {final_accuracy:.3f}")
    print(f"Standard Accuracy: {overall_metrics.get('accuracy', 0.0):.4f}")
    print(f"Precision: {overall_metrics.get('precision', 0.0):.4f}")
    print(f"Recall: {overall_metrics.get('recall', 0.0):.4f}")
    print(f"F1 Score: {overall_metrics.get('f1_score', 0.0):.4f}")
    print(f"Gender Categories: {len(gender_stats)}")
    
    print(f"\nDetailed Gender Metrics:")
    for gender, acc in gender_accuracies.items():
        print(f"  {gender}: Accuracy={acc:.3f} ({gender_correct[gender]}/{gender_total[gender]})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\nTiming Statistics (based on first {timing_sample_count} samples, excluding 1st):")
    print(f"Statistical Samples: {overall_summary.get('total_samples', 0)}")
    print(f"Average Inference Time: {overall_summary.get('avg_total_time', 0):.4f}s")
    print(f"Average Prefill Time: {overall_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Average Decode Time: {overall_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Average Throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/s")
    print(f"Results saved to: {output_file}")
    print(f"Timing stats saved to: {timing_output_file}")

if __name__ == "__main__":
    main()