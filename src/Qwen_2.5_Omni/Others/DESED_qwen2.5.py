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
from modeling_qwen2_5_omni_dart import (
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
    print(f"limit:{sample_limit}")

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

class GlobalTimingStats:
    
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_time = 0.0
        self.total_input_tokens = 0
        self.total_audio_tokens = 0
        self.timing_records = []
        self.max_samples = 100  
    
    def add_record(self, prefill_time, total_time, output_tokens, input_tokens, audio_tokens, sample_index):
       
        if sample_index == 0:
            return
        
        if self.samples >= self.max_samples:
            return
            
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_time += total_time
        self.total_input_tokens += input_tokens
        self.total_audio_tokens += audio_tokens
        
        
        self.timing_records.append({
            "sample_index": sample_index,
            "prefill_time": prefill_time,
            "total_time": total_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "audio_tokens": audio_tokens
        })
    
    def get_summary(self):
        """Get statistics summary"""
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_total_time": 0,
                "avg_input_tokens": 0,
                "avg_audio_tokens": 0
            }
        
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_total_time": self.total_time / self.samples,
            "avg_input_tokens": self.total_input_tokens / self.samples,
            "avg_audio_tokens": self.total_audio_tokens / self.samples
        }
    
    def export_to_json(self, output_file):
        """Export statistics data to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    Load data from new DESED task JSON file
    
    Args:
        json_file: DESED task JSON file path
        audio_base_dir: Audio file base directory
    
    Returns:
        dataset: List containing task data
    """
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []
    
    print(f"Loading DESED task JSON: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    
    
    if not isinstance(data, dict) or 'tasks' not in data:
        print(f"Error: JSON file format is incorrect, expected dictionary format containing 'tasks' field")
        return []
    
    tasks = data['tasks']
    print(f"Loaded {len(tasks)} tasks from JSON")
    
    
    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task missing audio path: {task}")
            continue
        
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:  
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue
        
        
        task_type = task.get("task_type", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        
        
        choices = task.get("choices", {})
        choice_a = choices.get("A", "")
        choice_b = choices.get("B", "")
        choice_c = choices.get("C", "")
        choice_d = choices.get("D", "")
        
        
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"Unable to read audio file information {full_audio_path}: {e}")
            continue
        
        
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_type": task_type,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "original_events": task.get("all_events", []),
            "all_events": task.get("all_events", []),
            "primary_event": task.get("primary_event", ""),
            "correct_event": task.get("correct_event", ""),
            "path_extracted_event": task.get("path_extracted_event", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"qa_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[task_type] += 1
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    return dataset

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file according to Qwen2.5-Omni requirements"""
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio.shape}, sample_rate={sr}Hz")
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
                print(f"Soundfile processing successful: shape={audio.shape}, sample_rate={sr}Hz")
                
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

def create_qa_prompt(doc):
    """Generate prompt for sound event detection task"""
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
    
    return prompt_text

def extract_answer_choice(response):
    """Extract answer choice (A, B, C, D) from model response"""
    if not response:
        return ""
    
    
    if "assistant\n" in response:
        
        assistant_start = response.rfind("assistant\n") + len("assistant\n")
        response = response[assistant_start:].strip()
    
    
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

def calculate_desed_metrics(y_true, y_pred):
    """
    Calculate detailed evaluation metrics for DESED sound event detection
    
    Args:
        y_true: True label list (A/B/C/D)
        y_pred: Predicted label list (A/B/C/D) 
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in ['A', 'B', 'C', 'D'] and pred_label in ['A', 'B', 'C', 'D']:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true)
        }
    
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    
    labels = ['A', 'B', 'C', 'D']
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=labels, average=None, zero_division=0
    )
    
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='weighted', zero_division=0
    )
    
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=labels,
        target_names=['Choice A', 'Choice B', 'Choice C', 'Choice D'],
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
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true)
    }

def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate sound event detection task accuracy"""
    try:
        
        pred = predicted_choice.strip().upper() if predicted_choice else ""
        gt = ground_truth_choice.strip().upper() if ground_truth_choice else ""
        
        
        accuracy = 1.0 if pred == gt else 0.0
        
        return {
            "accuracy": accuracy,
            "predicted_choice": pred,
            "ground_truth_choice": gt,
            "is_correct": pred == gt
        }
    except Exception as e:
        print(f"Error evaluating sound event detection accuracy: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def main():
    
    qa_json_file = "/path/to/your/subsetDESED/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json"
    audio_base_dir = "/path/to/your/subsetDESED/DESED_dataset/concatenated_audio"
    
    print(f"QA JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    
    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)
    
    result_dir = os.environ.get("RESULTS_DIR", './DESED_Results')
    os.makedirs(result_dir, exist_ok=True)

    
    output_file = f'{result_dir}/DESED_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
    timing_output_file = f'{result_dir}/DESED_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    
    timing_stats = GlobalTimingStats()

    print(f"\n=== DESED Sound Event Detection Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Task JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    
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
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
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
        print(f"Initialized thinker.model.config pruning configuration parameters")
    
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Using dataset with {len(samples)} samples")
    
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, processing {len(samples)} samples")

    
    task_type_stats = defaultdict(int)
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_stats[task_type] += 1
    
    print(f"Task type statistics: {dict(task_type_stats)}")

    
    results = []
    correct_count = 0
    total_count = 0

    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    
    tqdm_kwargs = {
        'ascii': True,      
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(samples)} samples...")
    
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="DESED Sound Event Detection Evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            
            audio_path = sample["audio"]["path"]
            audio = prepare_audio_for_qwen_omni(audio_path)
            
            
            ground_truth_choice = sample.get("answer_gt", "")
            task_type = sample.get("task_type", "unknown")
            
            
            prompt_text = create_qa_prompt(sample)

            
            sys_prompt = "You are a helpful assistant that analyzes audio to detect and classify sound events. Please listen carefully and select the most appropriate answer from the given choices."

            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},  
                        {"type": "text", "text": prompt_text}
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
                    print(f"Warning: No audio tokens detected for sample {idx}")

            
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            with torch.no_grad():
                prefill_output = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prefill_end_event.record()
            
            
            total_start_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)

            total_start_event.record()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=5,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            total_end_event.record()
            
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
            total_time = total_start_event.elapsed_time(total_end_event) / 1000.0  
            
            
            output_text = processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            
            if "assistant\n" in output_text:
                
                assistant_start = output_text.rfind("assistant\n") + len("assistant\n")
                output_text = output_text[assistant_start:].strip()
            
            
            if hasattr(output, 'shape') and len(output.shape) > 1:
                output_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            else:
                output_tokens = 0  
            
            
            output_text = output_text.strip()
            
            
            predicted_choice = extract_answer_choice(output_text)
            
            
            is_correct = predicted_choice == ground_truth_choice
            
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            
            timing_stats.add_record(prefill_time, total_time, output_tokens, input_token_length, audio_token_length, idx)
            
        except Exception as e:
            print(f"Inference error: {e}")
            traceback.print_exc()
            output_text = "ERROR"
            predicted_choice = ""
            is_correct = False
            prefill_time = 0
            total_time = 0
            output_tokens = 0
            input_token_length = 0
            audio_token_length = 0
            ground_truth_choice = sample.get("answer_gt", "")
            
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  
            
            total_count += 1  

        
        sample_result = {
            "idx": idx,
            "id": sample.get("id", f"sample_{idx}"),
            "filename": sample.get("filename", ""),
            "task_type": sample.get("task_type", "unknown"),
            "path": sample.get("path", ""),
            "duration": sample.get("duration", 0),
            "question": sample.get("question", ""),
            "choice_a": sample.get("choice_a", ""),
            "choice_b": sample.get("choice_b", ""),
            "choice_c": sample.get("choice_c", ""),
            "choice_d": sample.get("choice_d", ""),
            "ground_truth_choice": ground_truth_choice,
            "predicted_choice": predicted_choice,
            "is_correct": is_correct,
            "model_output": output_text,
            "input_tokens": input_token_length,
            "audio_tokens": audio_token_length,
            "output_tokens": output_tokens,
            "prefill_time": prefill_time,
            "total_time": total_time
        }
        
        results.append(sample_result)
        
        
        current_acc = correct_count / total_count if total_count > 0 else 0
        progress_bar.set_postfix({
            'Acc': f"{current_acc:.3f}",
            'Total': total_count
        })
        
        
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    
    final_accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    
    y_true = [sample["ground_truth_choice"] for sample in results]
    y_pred = [sample["predicted_choice"] for sample in results]
    
    
    detailed_metrics = calculate_desed_metrics(y_true, y_pred)
    
    
    timing_summary = timing_stats.get_summary()

    
    final_results = {
        "summary": {
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": final_accuracy,
            "f1_score": detailed_metrics['f1_macro'],
            "precision": detailed_metrics['precision_macro'],
            "recall": detailed_metrics['recall_macro'],
            "weighted_f1": detailed_metrics['f1_weighted'],
            "detailed_metrics": detailed_metrics,
            "config": {
                "gpu_id": gpu_id,
                "prune_layer_idx": prune_layer_idx,
                "prune_ratio": prune_ratio,
                "prune_method": method_is,
                "sample_limit": sample_limit,
                "task_json_file": qa_json_file,
                "audio_base_dir": audio_base_dir,
                "model_path": model_path
            },
            "timing": timing_summary
        },
        "samples": results
    }

    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== DESED Sound Event Detection Evaluation Results (Qwen2.5-Omni) ===")
    print(f"Total samples: {total_count}")
    print(f"Correct samples: {correct_count}")
    print(f"Accuracy: {final_accuracy:.2%}")
    print(f"F1 Score (Macro): {detailed_metrics['f1_macro']:.2%}")
    print(f"F1 Score (Weighted): {detailed_metrics['f1_weighted']:.2%}")
    print(f"Precision: {detailed_metrics['precision_macro']:.2%}")
    print(f"Recall: {detailed_metrics['recall_macro']:.2%}")
    
    print(f"\n=== Inference Time Statistics ===")
    print(f"Sample count: {timing_summary['samples']}")
    print(f"Average Prefill time: {timing_summary['avg_prefill_time']:.4f} seconds")
    print(f"Average total time: {timing_summary['avg_total_time']:.4f} seconds")
    print(f"Average input tokens: {timing_summary['avg_input_tokens']:.1f}")
    print(f"Average audio tokens: {timing_summary['avg_audio_tokens']:.1f}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()