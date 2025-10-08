import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
from scipy.io import wavfile
from scipy import signal
import librosa
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import random


random.seed(42)


sys.path.append("/data/to/your/Modeling/path/")
from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)


from qwen_omni_utils import process_mm_info


_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      




os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"


logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

class RaceTimingStats:
    """Track inference timing statistics for RACE tasks"""
    def __init__(self):
        self.timing_records = []
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
            "audio_duration": audio_duration,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get summary statistics (excluding the first sample)"""
        if self.total_samples == 0:
            return {}
        
        return {
            "total_samples": self.total_samples,
            "avg_prefill_time": self.total_prefill_time / self.total_samples,
            "avg_decode_time": self.total_decode_time / self.total_samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        }
    
    def export_to_json(self, output_file):
        """Export statistics data to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def clean_text_response(response):
    """Clean model response for RACE task, keep only the first character as option label"""
    if not response:
        return ""
    
    
    resp = response.strip().upper()
    
    
    for ch in resp:
        if ch in ["A", "B", "C", "D"]:
            return ch
    
    
    words = resp.split()
    for word in words:
        for ch in word:
            if ch in ["A", "B", "C", "D"]:
                return ch
    
    
    return words[0] if words else ""

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file according to Qwen2.5-Omni requirements"""
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio.shape}, sample rate={sr}Hz")
        except Exception as e:
            print(f"Librosa loading failed: {e}")
            
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"Soundfile processing successful: shape={audio.shape}, sample rate={sr}Hz")
                
            except Exception as e:
                print(f"Soundfile loading also failed: {e}")
                
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generating silent replacement audio")
        
        
        if len(audio) == 0:
            print("Warning: Audio is empty, creating 3 seconds of silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            
        
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def load_audio_for_race(audio_path, audio_cache=None):
    """
    Load audio file, return format consistent with Qwen2.5-Omni
    Returns: audio_array, sampling_rate
    """
    if audio_cache is not None and audio_path in audio_cache:
        
        audio_np, sr = audio_cache[audio_path]
    else:
        
        audio_np = prepare_audio_for_qwen_omni(audio_path)
        sr = 16000  
        
        
        if audio_cache is not None:
            audio_cache[audio_path] = (audio_np, sr)
    
    return audio_np, sr

def create_race_prompt(question, options):
    """Create RACE task prompt (adapted to Qwen2.5 format but keeping content consistent)"""
    instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  
        formatted_options += f"{letter}. {opt}\n"
    
    
    prompt = f"{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}"
    
    return prompt

def calculate_race_metrics(y_true, y_pred, subset_labels=None):
    """
    Calculate detailed evaluation metrics for RACE reading comprehension task
    
    Args:
        y_true: True label list (A/B/C/D format)
        y_pred: Predicted label list (A/B/C/D format)
        subset_labels: Subset label list for grouped analysis
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    clean_subset_labels = [] if subset_labels is not None else None
    
    
    valid_labels = ['A', 'B', 'C', 'D']
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in valid_labels and pred_label in valid_labels:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
            if subset_labels is not None:
                clean_subset_labels.append(subset_labels[i])
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'per_class_metrics': {},
            'subset_metrics': {},
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true),
            'class_labels': valid_labels
        }
    
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=valid_labels, average=None, zero_division=0
    )
    
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=valid_labels, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=valid_labels, average='weighted', zero_division=0
    )
    
    
    per_class_metrics = {}
    for i, label in enumerate(valid_labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    
    subset_metrics = {}
    if clean_subset_labels is not None:
        unique_subsets = list(set(clean_subset_labels))
        for subset in unique_subsets:
            subset_indices = [i for i, s in enumerate(clean_subset_labels) if s == subset]
            if len(subset_indices) > 0:
                subset_y_true = [clean_y_true[i] for i in subset_indices]
                subset_y_pred = [clean_y_pred[i] for i in subset_indices]
                
                subset_accuracy = accuracy_score(subset_y_true, subset_y_pred)
                subset_precision, subset_recall, subset_f1, _ = precision_recall_fscore_support(
                    subset_y_true, subset_y_pred, average='macro', zero_division=0
                )
                
                subset_metrics[subset] = {
                    'accuracy': float(subset_accuracy),
                    'precision': float(subset_precision),
                    'recall': float(subset_recall),
                    'f1_score': float(subset_f1),
                    'samples': len(subset_indices)
                }
    
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=valid_labels,
        target_names=[f"Choice {label}" for label in valid_labels],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': per_class_metrics,
        'subset_metrics': subset_metrics,
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true),
        'class_labels': valid_labels
    }

def main():
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 1))
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
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    
    data_path_root = '/path/to/your/subsetrace_audio'  
    
    
    result_dir = './Race_Results'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    
    output_file = f'{result_dir}/Race_results_qwen25_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
    timing_output_file = f'{result_dir}/Race_timing_stats_qwen25_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    
    timing_stats = RaceTimingStats()

    print(f"\n=== RACE Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    
    print("Loading Qwen2.5-Omni model...")
    model_path = "/path/to/your/model"  
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
        trust_remote_code=True,
        output_attentions=False,
    )
    model.disable_talker()
    
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        
        if not hasattr(model.thinker.model.config, 'image_layer_idx'):
            model.thinker.model.config.image_layer_idx = False
        if not hasattr(model.thinker.model.config, 'audio_layer_idx'):
            model.thinker.model.config.audio_layer_idx = None
        if not hasattr(model.thinker.model.config, 'audio_token_num'):
            model.thinker.model.config.audio_token_num = None
        if not hasattr(model.thinker.model.config, 'audio_token_start'):
            model.thinker.model.config.audio_token_start = None
        if not hasattr(model.thinker.model.config, 'audio_prune_ratio'):
            model.thinker.model.config.audio_prune_ratio = 0
        if not hasattr(model.thinker.model.config, 'random'):
            model.thinker.model.config.random = False
        if not hasattr(model.thinker.model.config, 'frame'):
            model.thinker.model.config.frame = False
        print(f"Initializing thinker.model.config pruning configuration parameters")
    
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    
    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: Benchmark file not found: {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    
    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f"Sample count limited to: {sample_limit}")

    audio_cache = {}
    results = []

    
    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0

    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    
    tqdm_kwargs = {
        'ascii': True,        
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(benchmark)} samples...")
    
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc="RACE evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            audio_rel = sample["audio_path"]
            audio_full = os.path.join(data_path_root, audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"Warning: Audio file does not exist: {audio_full}")
                continue
                
            
            audio_np, sr = load_audio_for_race(audio_full, audio_cache)
            
            
            prompt_text = create_race_prompt(sample['question'], sample['options'])

            
            if "high" in audio_rel:
                total_high += 1
            elif "middle" in audio_rel:
                total_middle += 1

            
            task_instruction = "You are a helpful assistant that analyzes reading comprehension passages with audio narration. Please listen to the passage and answer the multiple-choice question based on what you heard."
            full_user_prompt = f"{task_instruction}\n\n{prompt_text}"

            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_full},  
                        {"type": "text", "text": full_user_prompt}
                    ]
                }
            ]
            
            
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            
            
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
                
                
                bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                
                if bos_positions and eos_positions:
                    
                    audio_token_start = bos_positions[0]
                    audio_token_end = eos_positions[0]
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    audio_detected = True
                    
                    
                    model.thinker.model.config.image_layer_idx = False  
                    model.thinker.model.config.audio_layer_idx = prune_layer_idx
                    model.thinker.model.config.audio_token_num = audio_token_length
                    model.thinker.model.config.audio_token_start = audio_token_start
                    model.thinker.model.config.audio_prune_ratio = prune_ratio
                    model.thinker.model.config.random = use_random
                    model.thinker.model.config.frame = use_frame 
                    
            if not audio_detected:
                model.thinker.model.config.audio_layer_idx = None
                model.thinker.model.config.audio_prune_ratio = 0

            
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            
            
            audio_tokens = 0
            if hasattr(processor.tokenizer, 'audio_bos_token_id') and hasattr(processor.tokenizer, 'audio_eos_token_id'):
                input_ids = inputs['input_ids'][0]
                audio_bos_positions = (input_ids == processor.tokenizer.audio_bos_token_id).nonzero(as_tuple=True)[0]
                audio_eos_positions = (input_ids == processor.tokenizer.audio_eos_token_id).nonzero(as_tuple=True)[0]
                
                if len(audio_bos_positions) > 0 and len(audio_eos_positions) > 0:
                    for bos_pos in audio_bos_positions:
                        eos_candidates = audio_eos_positions[audio_eos_positions > bos_pos]
                        if len(eos_candidates) > 0:
                            eos_pos = eos_candidates[0]
                            audio_tokens += eos_pos - bos_pos - 1
                            
                
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
                    if hasattr(model.thinker.model.config, 'sparse_attention_config'):
                        model.thinker.model.config.sparse_attention_config['audio_tokens'] = audio_tokens.item() if hasattr(audio_tokens, 'item') else audio_tokens
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prefill_end_event.record()

            
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)

            decode_start_event.record()
            out_ids = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=5,
                thinker_do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            decode_end_event.record()
            
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0  
            
            
            
            generated_tokens = out_ids[:, inputs["input_ids"].shape[-1]:]
            response = processor.batch_decode(
                generated_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            

            
            output_tokens = len(out_ids[0]) - len(inputs["input_ids"][0])
            input_tokens = len(inputs["input_ids"][0])
            
            pred = clean_text_response(response)
            
            
            if idx < 5:
                print(f"\n=== Sample {idx} debug information ===")
                print(f"Model raw output: '{response}'")
                print(f"Extracted answer: '{pred}'")
                print(f"Correct answer: '{sample['answer']}'")
                print(f"Is correct: {pred == sample['answer']}")
                print(f"Output token count: {output_tokens}")
                print("=" * 30)

            
            correct = int(pred == sample["answer"])
            if correct:
                correct_count += 1
                if "high" in audio_rel:
                    correct_high += 1
                elif "middle" in audio_rel:
                    correct_middle += 1
            
            current_acc = (correct_count / (idx + 1)) * 100
            
            
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(benchmark):
                progress_bar.set_postfix({
                    'acc': f'{current_acc:.2f}%', 
                    'ans': f'{pred}/{sample["answer"]}',
                    'audio_len': f'{len(audio_np)/sr:.1f}s'
                })
                
                if is_screen_env:
                    
                    print(f"  Progress: {sample_count}/{len(benchmark)} ({sample_count/len(benchmark)*100:.1f}%), "
                          f"Accuracy: {current_acc:.2f}%")
            else:
                progress_bar.set_postfix({
                    'acc': f'{current_acc:.2f}%', 
                    'ans': f'{pred}/{sample["answer"]}',
                    'audio_len': f'{len(audio_np)/sr:.1f}s'
                })

            
            results.append({
                "idx": idx,
                "article_id": sample.get("article_id", ""),
                "question_idx": sample.get("question_idx", idx),
                "pred": pred, 
                "gt": sample["answer"],
                "correct": correct,
                "raw_response": response,  
                "audio_path": audio_rel,
                "subset": "high" if "high" in audio_rel else "middle",
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "output_tokens": output_tokens,
                "input_tokens": input_tokens
            })

            
            if idx > 0 and idx <= 100:
                timing_stats.add_record(
                    prefill_time, decode_time, 
                    output_tokens,
                    input_tokens,
                    len(audio_np) / sr
                )

            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            continue

    
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    
    y_true = [r["gt"] for r in results]
    y_pred = [r["pred"] for r in results]
    subset_labels = [r["subset"] for r in results]
    
    
    detailed_metrics = calculate_race_metrics(y_true, y_pred, subset_labels)

    
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
        "sklearn_metrics": detailed_metrics,  
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "timing_sample_count": min(100, max(0, len(results) - 1))
        },
        "timing": timing_stats.get_summary()
    }

    
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    
    print(f"Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    
    json_output_file = f'{result_dir}/Race_results_qwen25_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    print(f"Saving complete results to: {json_output_file}")
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    
    print(f"Saving timing statistics to: {timing_output_file}")
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== RACE Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    
    
    metrics = detailed_metrics
    print(f"\n=== Detailed Evaluation Metrics (sklearn) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro Average): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted Average): {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro Average): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro Average): {metrics['recall_macro']:.4f}")
    
    print(f"\n=== Per-Choice Evaluation Metrics ===")
    for choice, per_class_metrics in metrics['per_class_metrics'].items():
        print(f"Choice {choice}:")
        print(f"  Precision: {per_class_metrics['precision']:.4f}")
        print(f"  Recall: {per_class_metrics['recall']:.4f}")
        print(f"  F1 Score: {per_class_metrics['f1_score']:.4f}")
        print(f"  Sample Count: {per_class_metrics['support']}")
    
    print(f"\n=== Subset Evaluation Metrics ===")
    for subset, subset_metrics in metrics['subset_metrics'].items():
        print(f"{subset.upper()} set:")
        print(f"  Accuracy: {subset_metrics['accuracy']:.4f}")
        print(f"  Precision: {subset_metrics['precision']:.4f}")
        print(f"  Recall: {subset_metrics['recall']:.4f}")
        print(f"  F1 Score: {subset_metrics['f1_score']:.4f}")
        print(f"  Sample Count: {subset_metrics['samples']}")
    
    print(f"\n=== Traditional Accuracy Statistics ===")
    if total_high > 0:
        print(f"HIGH set accuracy: {correct_high/total_high*100:.2f}% ({correct_high}/{total_high})")
    if total_middle > 0:
        print(f"MIDDLE set accuracy: {correct_middle/total_middle*100:.2f}% ({correct_middle}/{total_middle})")
    
    timing_summary = timing_stats.get_summary()
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\n=== Inference Time Statistics ===")
    print(f"Average inference time: {timing_summary.get('avg_total_time', 0):.4f} seconds (first {timing_sample_count} samples, excluding the first one)")
    print(f"Average Prefill time: {timing_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average Decode time: {timing_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    
    print(f"\n=== Classification Detailed Report ===")
    print(metrics['classification_report'])
    
    print(f"\nResults saved to: {output_file}")
    print(f"Complete results saved to: {json_output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()
