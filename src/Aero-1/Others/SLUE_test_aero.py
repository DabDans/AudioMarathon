import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
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

# Disable transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("Successfully imported required modules")

# Set output buffering - ensure real-time display
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

# Get GPU ID
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

# Audio pruning config
prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

# Set flags according to pruning method
use_random = (prune_method == "random")
use_frame = (prune_method == "frame")
if use_random is False and use_frame is False:
    prune_method = "fast_v"
if prune_ratio == 0:
    method_is = "base"
else:
    method_is = prune_method

# Sample limit (if provided)
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

# Debug mode toggle
debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
if debug_mode:
    print("Debug mode enabled - detailed output will be shown")

# Data path config
slue_json_file = "/data/to/your/eval/SLUE/merged_audio_data.json"
audio_base_dir = "/data/to/your/eval/SLUE"
result_dir = '/data/to/your/SLUE_Results'
os.makedirs(result_dir, exist_ok=True)

# Modify output file paths and naming
output_file = f'{result_dir}/SLUE_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/SLUE_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
cuda_event_output_file = f'{result_dir}/SLUE_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing stats will be saved to: {timing_output_file}")
print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class FolderTimingStats:
    """Track inference timing stats for each folder"""
    def __init__(self):
        self.folder_stats = {}
        self.current_folder = None
    
    def set_current_folder(self, folder_name):
        self.current_folder = folder_name
        if folder_name not in self.folder_stats:
            self.folder_stats[folder_name] = {
                "samples": 0,
                "total_prefill_time": 0.0,
                "total_decode_time": 0.0,
                "total_tokens": 0,
                "timing_records": []
            }
    
    def add_record(self, prefill_time, decode_time, output_tokens):
        if self.current_folder is None:
            return
        
        folder_data = self.folder_stats[self.current_folder]
        folder_data["samples"] += 1
        folder_data["total_prefill_time"] += prefill_time
        folder_data["total_decode_time"] += decode_time
        folder_data["total_tokens"] += output_tokens
        
        folder_data["timing_records"].append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        })
    
    def export_to_json(self, output_file):
        """Export stats to JSON file"""
        result = {
            "folder_summaries": {
                folder: {
                    "folder": folder,
                    "samples": stats["samples"],
                    "avg_prefill_time": stats["total_prefill_time"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_decode_time": stats["total_decode_time"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_total_time": (stats["total_prefill_time"] + stats["total_decode_time"]) / stats["samples"] if stats["samples"] > 0 else 0,
                    "total_tokens": stats["total_tokens"],
                    "avg_tokens": stats["total_tokens"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_tokens_per_sec": stats["total_tokens"] / stats["total_decode_time"] if stats["total_decode_time"] > 0 else 0
                }
                for folder, stats in self.folder_stats.items() if stats["samples"] > 0
            },
            "detailed_records": self.folder_stats
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

class CudaEventTimingStats:
    """
    CUDA Event batch timing stats class
    
    Follows NVIDIA CUDA best practice guidelines for GPU timing:
    - Use event synchronization (cudaEventSynchronize) rather than device synchronization for efficiency
    - Correctly understand that event recording is a task posted to the GPU, actual time is recorded when GPU executes
    - Must wait for event actual completion before calculating time delta
    """
    
    def __init__(self):
        self.timing_records = []
        self.prefill_times = []
        self.decode_times = []
        self.total_times = []
    
    def add_timing_record(self, prefill_time, decode_time, total_time):
        """Add a timing record"""
        self.prefill_times.append(prefill_time)
        self.decode_times.append(decode_time)
        self.total_times.append(total_time)
        
        self.timing_records.append({
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time
        })
    
    def get_time_statistics(self, times_list, name=""):
        """Calculate timing statistics (mean only)"""
        if not times_list:
            return {}
        
        stats = {
            f"{name}_avg": sum(times_list) / len(times_list),
            f"{name}_count": len(times_list)
        }
        return stats
    
    def get_full_statistics(self):
        """Get full timing statistics"""
        stats = {}
        stats.update(self.get_time_statistics(self.prefill_times, "prefill"))
        stats.update(self.get_time_statistics(self.decode_times, "decode"))
        stats.update(self.get_time_statistics(self.total_times, "total"))
        return stats
    
    def print_statistics(self):
        """Print timing statistics (mean only)"""
        if not self.timing_records:
            print("No timing stats data")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill timing stats:")
        print(f"  Mean: {prefill_stats['prefill_avg']:.6f}s")
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode timing stats:")
        print(f"  Mean: {decode_stats['decode_avg']:.6f}s")
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total timing stats:")
        print(f"  Mean: {total_stats['total_avg']:.6f}s")
        print(f"  Sample count: {total_stats['total_count']}")

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30-second chunks (480,000 samples @16kHz)"""
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    audio_splits = []
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio with librosa and split, compatible with Aero-1 official example"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:  # 30 seconds @ 16kHz
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30 seconds, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_slue_dataset(json_file, audio_base_dir):
    """
    Load SLUE task data from JSON file
    
    Args:
        json_file: SLUE format JSON task file path
        audio_base_dir: audio file base directory
    
    Returns:
        dataset: list containing task data
    """
    dataset = []
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
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
        print(f"Error: JSON file format incorrect, expected list")
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
            print(f"Warning: Task missing audio path: {task}")
            continue
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file not found: {full_audio_path}")
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
            print(f"Cannot read audio file info {full_audio_path}: {e}")
            continue
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
        print(f"Warning: Total {missing_files} audio files not found")
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type stats: {dict(task_type_stats)}")
    print(f"Dataset stats: {dict(dataset_stats)}")
    return dataset

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

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference function using CUDA Event API for precise GPU timing
    
    Follows NVIDIA CUDA best practice for correct timing logic:
    1. Post start timing task to GPU stream
    2. Post computation task to GPU stream  
    3. Post end timing task to GPU stream
    4. Synchronize event before reading time delta
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
    print("Loading Aero-1 model...")
    sys.stdout.flush()
    model_name = "lmms-lab/Aero-1-Audio-1.5B"
    print(f"Using Aero-1 model: {model_name}")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(
        model_name,
        revision="main",
        trust_remote_code=True
    )
    print("Successfully loaded Aero processor")
    sys.stdout.flush()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation="sdpa",  # if no flash_attention_2, use sdpa
        trust_remote_code=True
    )
    model.eval()
    print("Successfully loaded Aero-1 model")
    sys.stdout.flush()
    if prune_ratio > 0:
        print(f"Note: Official Aero-1 model, audio pruning may not be available")
        print(f"Pruning config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    timing_stats = FolderTimingStats()
    cuda_event_stats = CudaEventTimingStats()
    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, will process {len(samples)} samples")
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    for sample in samples:
        task_type_stats[sample.get("task_name", "unknown")] += 1
        dataset_stats[sample.get("dataset_name", "unknown")] += 1
    print(f"Task type stats: {dict(task_type_stats)}")
    print(f"Dataset stats: {dict(dataset_stats)}")
    results = {
        "model_name": "Aero-1-Audio-1.5B",
        "pruning_config": {
            "note": "Official model - pruning may not be available",
            "layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "method": prune_method
        },
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "task_type_correct": defaultdict(int),
            "task_type_total": defaultdict(int),
            "dataset_correct": defaultdict(int),
            "dataset_total": defaultdict(int),
            "timing": {
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_prefill_time": 0,
                "total_decode_time": 0,
                "total_total_time": 0,
            }
        }
    }
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout,
        'mininterval': 0.1,
        'maxinterval': 1.0,
        'disable': False,
        'leave': True,
        'position': 0
    }
    if is_screen_env:
        tqdm_kwargs['mininterval'] = 0.05
        tqdm_kwargs['maxinterval'] = 0.5
    allocated, reserved = get_gpu_memory_usage()
    print(f"After model loaded, GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    with tqdm(total=len(samples), desc="Processing SLUE Tasks (Aero-1)", **tqdm_kwargs) as pbar:
        timing_stats.set_current_folder("SLUE_Tasks_Aero1")
        for i, item in enumerate(samples):
            audio_path = item['path']
            task_name = item.get('task_name', 'unknown')
            dataset_name = item.get('dataset_name', 'unknown')
            ground_truth_choice = item.get('answer_gt', '')
            output = ""
            predicted_choice = ""
            is_correct = False
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            try:
                audio_chunks, sample_rate = prepare_audio_for_processor(audio_path)
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",
                    })
                question = item.get("question", "")
                choice_a = item.get("choice_a", "")
                choice_b = item.get("choice_b", "")
                choice_c = item.get("choice_c", "")
                choice_d = item.get("choice_d", "")
                prompt_text = f"""{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."""
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
                ).to("cuda")
                audio_token_length = 0
                _AUDIO_SPECIAL_TOKEN_ID=151667
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    print(audio_token_length)
                    model.config.image_layer_idx = None
                    model.config.audio_layer_idx = prune_layer_idx
                    model.config.audio_token_num = audio_token_length
                    model.config.audio_token_start = audio_token_start
                    model.config.audio_prune_ratio = prune_ratio
                    model.config.random = use_random
                    model.config.frame = use_frame
                if debug_mode:
                    print(f"Processing audio: {os.path.basename(audio_path)}")
                    print(f"Task type: {task_name}")
                    print(f"Dataset: {dataset_name}")
                    print(f"Number of audio chunks: {len(audio_chunks)}")
                    print(f"Total audio length: {sum(len(chunk) for chunk in audio_chunks)}, sample rate: {sample_rate}")
                    print(f"Generated prompt: {prompt}")
                    print(f"Input IDs shape: {inputs.input_ids.shape}")
                    if hasattr(inputs, 'audio_features'):
                        print(f"Audio features shape: {inputs.audio_features.shape}")
                    print(f"Estimated audio token length: {audio_token_length}")
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
                predicted_choice = extract_answer_choice(output)
                is_correct = predicted_choice.upper() == ground_truth_choice.upper()
                if debug_mode:
                    print(f"Model output: '{output}'")
                    print(f"Inference time: total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                    print(f"Output token count: {output_tokens}")
                    print(f"Extracted answer: '{predicted_choice}'")
                    print(f"Ground truth: '{ground_truth_choice}'")
                    print(f"Answer correct: {is_correct}")
                    print("=" * 50)
                    sys.stdout.flush()
                results["summary"]["total_samples"] += 1
                results["summary"]["task_type_total"][task_name] += 1
                results["summary"]["dataset_total"][dataset_name] += 1
                if is_correct:
                    results["summary"]["correct_samples"] += 1
                    results["summary"]["task_type_correct"][task_name] += 1
                    results["summary"]["dataset_correct"][dataset_name] += 1
                results["summary"]["timing"]["total_prefill_time"] += prefill_time
                results["summary"]["timing"]["total_decode_time"] += decode_time
                results["summary"]["timing"]["total_total_time"] += total_time
                timing_stats.add_record(prefill_time, decode_time, output_tokens)
                cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                output = "ERROR"
                predicted_choice = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                total_time = 0
                output_tokens = 0
                audio_token_length = 0
            sample_result = {
                "idx": i,
                "id": item.get("id", f"sample_{i}"),
                "filename": item.get("filename", ""),
                "task_name": task_name,
                "dataset_name": dataset_name,
                "audio_path": audio_path,
                "duration": item.get("duration", 0),
                "question": item.get("question", ""),
                "choice_a": item.get("choice_a", ""),
                "choice_b": item.get("choice_b", ""),
                "choice_c": item.get("choice_c", ""),
                "choice_d": item.get("choice_d", ""),
                "ground_truth_choice": ground_truth_choice,
                "predicted_choice": predicted_choice,
                "is_correct": is_correct,
                "model_output": output,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                "entity_count": item.get("entity_count", 0),
                "entity_types": item.get("entity_types", []),
                "source_count": item.get("source_count", 0)
            }
            results["samples"].append(sample_result)
            if 'inputs' in locals():
                del inputs
            if 'audio_chunks' in locals():
                del audio_chunks
            if 'result' in locals():
                del result
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if (i + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {i+1}] GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
            update_interval = 50 if is_screen_env else 20
            sample_count = i + 1
            if sample_count % update_interval == 0 or sample_count == len(samples):
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                pbar.set_postfix_str(
                    f"Accuracy:{current_accuracy:.1%}, Task:{task_name[:8]}, Pred:{predicted_choice}, GT:{ground_truth_choice}"
                )
                if is_screen_env:
                    print(f"Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.1%}")
                    sys.stdout.flush()
            pbar.update()
            if is_screen_env and sample_count % 25 == 0:
                sys.stdout.flush()
    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / total_samples
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / total_samples
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / total_samples
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    task_type_accuracies = {}
    for task_name, total in results["summary"]["task_type_total"].items():
        if total > 0:
            correct = results["summary"]["task_type_correct"][task_name]
            task_type_accuracies[task_name] = correct / total
    dataset_accuracies = {}
    for dataset_name, total in results["summary"]["dataset_total"].items():
        if total > 0:
            correct = results["summary"]["dataset_correct"][dataset_name]
            dataset_accuracies[dataset_name] = correct / total
    results["summary"]["task_type_accuracies"] = task_type_accuracies
    results["summary"]["dataset_accuracies"] = dataset_accuracies
    json_output_file = f'{result_dir}/SLUE_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    timing_stats.export_to_json(timing_output_file)
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)
    print(f"CUDA Event stats saved to: {cuda_event_output_file}")
    print("\n=== SLUE Task Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"\nTask type accuracies:")
    for task_name, acc in task_type_accuracies.items():
        correct = results["summary"]["task_type_correct"][task_name]
        total = results["summary"]["task_type_total"][task_name]
        print(f"  {task_name}: {acc:.2%} ({correct}/{total})")
    print(f"\nDataset accuracies:")
    for dataset_name, acc in dataset_accuracies.items():
        correct = results["summary"]["dataset_correct"][dataset_name]
        total = results["summary"]["dataset_total"][dataset_name]
        print(f"  {dataset_name}: {acc:.2%} ({correct}/{total})")
    print(f"Average inference time: {results['summary']['timing']['avg_total_time']:.4f} seconds")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} seconds")
    cuda_event_stats.print_statistics()
    print(f"Results saved to: {json_output_file}")
    print(f"Timing stats saved to: {timing_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()