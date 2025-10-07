import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import numpy as np
import soundfile as sf
import librosa
import warnings
import traceback
import time
import glob
import random
import sys
import io
import gc

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Disable transformers warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("Successfully imported required modules")

# Get GPU ID
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

# Audio pruning configuration
prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

# Set flags according to method name
use_random = (prune_method == "random")
use_frame = (prune_method == "frame")
if use_random == False and use_frame == False:
    prune_method = "fast_v"

# Decide method name
if prune_ratio == 0:
    method_is = "base"
else:
    method_is = prune_method

# Sample limit and debug mode
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]

if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")
if debug_mode:
    print("Debug mode enabled - Detailed output will be shown")

# Data path configuration - VoxCeleb age classification dataset path
data_path_root = os.environ.get("VOXCELEB_AGE_DATA_PATH", 
    '/data/to/your/VoxCeleb/concatenated_audio')
result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/Vox_Results')
os.makedirs(result_dir, exist_ok=True)

# Output file paths and naming
output_file = f'{result_dir}/VoxCeleb_age_Aero1_results_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
timing_output_file = f'{result_dir}/VoxCeleb_age_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
cuda_event_output_file = f'{result_dir}/VoxCeleb_age_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'

print(f"Results will be saved to: {output_file}")
print(f"Timing stats will be saved to: {timing_output_file}")
print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")

class FolderTimingStats:
    """Track inference timing statistics per folder"""
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
        
        # Add detailed record
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
    """CUDA Event batch timing statistics class"""
    
    def __init__(self):
        self.timing_records = []
        self.prefill_times = []
        self.decode_times = []
        self.total_times = []
    
    def add_timing_record(self, prefill_time, decode_time, total_time):
        """Add a timing measurement record"""
        self.prefill_times.append(prefill_time)
        self.decode_times.append(decode_time)
        self.total_times.append(total_time)
        
        self.timing_records.append({
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time
        })
    
    def get_time_statistics(self, times_list, name=""):
        """Compute time statistics (average only)"""
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
        """Print timing statistics (average only)"""
        if not self.timing_records:
            print("No timing statistics data")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        
        # Prefill stats
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill time stats:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        
        # Decode stats
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode time stats:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        
        # Total stats
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total time stats:")
        print(f"  Average: {total_stats['total_avg']:.6f}s")
        print(f"  Samples: {total_stats['total_count']}")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    
    # Use librosa for resampling
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
    """Load audio with librosa and split, compatible with Aero-1 official examples"""
    
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure data type is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate does not match
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split if audio longer than 30s
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30s limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # If not longer than 30s, return as single chunk list
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        if debug_mode:
            traceback.print_exc()
        # Return silence chunk list
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    """Load dataset from concatenated_audio directory, based on age_classification_task_meta.json"""
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(wav_path):
            if debug_mode:
                print(f"Warning: File does not exist {wav_path}")
            continue
        
        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()
        speaker_age = item.get("speaker_age", 0)
        
        all_samples.append({
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            #"choice_e": item["choice_e"],
            "answer_gt": age_group,
            "task": "Speaker_Age_Classification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    
    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Applying sample limit: Randomly selecting {sample_limit} of {len(all_samples)} samples")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Limited sample count: {len(all_samples)}")
    
    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    random.shuffle(all_samples)
    
    return all_samples

def extract_age_answer(text, choices):
    """Extract age group answer from model output text"""
    text_lower = text.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return choices["choice_d"]
    #if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)'):
    #    return choices["choice_e"]
        
    for option, choice_text in choices.items():
        option_letter = option[-1].lower()
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"{option_letter})" in text_lower:
            return choice_text
    
    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    
    if len(choice_matches) == 1:
        return choice_matches[0]
    
    return ""

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference function with precise GPU timing using CUDA Event API
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
    print(f"\n=== VoxCeleb Age Classification Evaluation Config (Aero-1) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Data path: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)
    
    # Step1: Load Aero-1 model
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
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    model.eval()
    print("Successfully loaded Aero-1 model")
    sys.stdout.flush()

    if prune_ratio > 0:
        print(f"Note: Using official Aero-1 model, audio pruning may not be available")
        print(f"Pruning config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    
    timing_stats = FolderTimingStats()
    cuda_event_stats = CudaEventTimingStats()
    
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    
    if not samples:
        print("Error: Failed to load any samples")
        return
    
    grouped_samples = {}
    for sample in samples:
        speaker_id = sample["speaker_id"]
        if speaker_id not in grouped_samples:
            grouped_samples[speaker_id] = []
        grouped_samples[speaker_id].append(sample)
    
    age_group_counts = {}
    for s in samples:
        group = s["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age group stats:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"After model load, GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "age_group_stats": {},
            "timing": {
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_prefill_time": 0,
                "total_decode_time": 0,
                "total_total_time": 0,
            },
            "config": {
                "model_name": "Aero-1-Audio-1.5B",
                "gpu_id": gpu_id,
                "prune_layer_idx": prune_layer_idx,
                "prune_ratio": prune_ratio,
                "prune_method": method_is,
                "sample_limit": sample_limit,
                "data_path": data_path_root
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
        'disable': False
    }
    
    if is_screen_env:
        tqdm_kwargs['mininterval'] = 0.05
        tqdm_kwargs['maxinterval'] = 0.5
    
    with tqdm(total=len(grouped_samples), desc="Processing speaker groups", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        for speaker_id, items in grouped_samples.items():
            folder_count += 1
            pbar_folders.set_description(f"Processing speaker [{folder_count}/{total_folders}]: {speaker_id}")
            
            timing_stats.set_current_folder(speaker_id)
            sample_count = 0
            total_samples = len(items)            
            for i, item in enumerate(items):
                sample_count = i + 1
                wav_path = item['wav_path']
                task = item.get('task', 'Speaker_Age_Classification')
                
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                audio_token_length = 0
                predicted_age_group = ""
                is_correct = False
                ground_truth = item["age_group"].strip()
                
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": []
                        }
                    ]
                    
                    audio_chunks, sample_rate = prepare_audio_for_processor(wav_path)
                    
                    for chunk in audio_chunks:
                        messages[0]["content"].append({
                            "type": "audio",
                            "audio": "placeholder",
                        })
                    
                    instruction = "Listen to this audio and identify the speaker's age group. Choose the most appropriate option: (a) Young Adult (18-30), (b) Early Career (31-40), (c) Mid Career (41-50), (d) Senior (51-70), (e) Elderly (71+). Answer with only the letter (a, b, c, d, or e)."
                    
                    messages[0]["content"].append({
                        "type": "text",
                        "text": instruction
                    })
                    
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    
                    inputs = processor(
                        text=prompt,
                        audios=audio_chunks,
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    audio_token_length = 0
                    _AUDIO_SPECIAL_TOKEN_ID = 151667
                    if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                        token_ids = inputs.input_ids[0].tolist()
                        audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                        audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                        audio_token_length = audio_token_end - audio_token_start + 1
                        
                        model.config.image_layer_idx = None
                        model.config.audio_layer_idx = prune_layer_idx
                        model.config.audio_token_num = audio_token_length
                        model.config.audio_token_start = audio_token_start
                        model.config.audio_prune_ratio = prune_ratio
                        model.config.random = use_random
                        model.config.frame = use_frame
                    
                    if debug_mode:
                        print(f"Processing audio: {os.path.basename(wav_path)}")
                        print(f"Speaker ID: {speaker_id}")
                        print(f"Audio chunks: {len(audio_chunks)}")
                        print(f"Age group: {ground_truth}")
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
                    
                    choices = {
                        "choice_a": item["choice_a"],
                        "choice_b": item["choice_b"],
                        "choice_c": item["choice_c"],
                        "choice_d": item["choice_d"]
                        #"choice_e": item["choice_e"]
                    }
                    predicted_age_group = extract_age_answer(output, choices)
                    
                    is_correct = predicted_age_group == ground_truth
                    
                    if debug_mode:
                        print(f"Model output: '{output}'")
                        print(f"Extracted answer: '{predicted_age_group}'")
                        print(f"Ground truth: '{ground_truth}'")
                        print(f"Answer correct: {is_correct}")
                        print(f"Inference time: Total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                        print(f"Output tokens: {output_tokens}")
                        print("=" * 50)
                        sys.stdout.flush()
                    
                    results["summary"]["total_samples"] += 1
                    if ground_truth not in results["summary"]["age_group_stats"]:
                        results["summary"]["age_group_stats"][ground_truth] = {
                            "total": 0,
                            "correct": 0
                        }
                    
                    results["summary"]["age_group_stats"][ground_truth]["total"] += 1
                    if is_correct:
                        results["summary"]["age_group_stats"][ground_truth]["correct"] += 1
                        results["summary"]["correct_samples"] += 1
                    
                    if i > 0:
                        results["summary"]["timing"]["total_prefill_time"] += prefill_time
                        results["summary"]["timing"]["total_decode_time"] += decode_time
                        results["summary"]["timing"]["total_total_time"] += total_time
                        timing_stats.add_record(prefill_time, decode_time, output_tokens)
                        
                        cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    if debug_mode:
                        traceback.print_exc()
                    output = "ERROR"
                    predicted_age_group = "error"
                    is_correct = False
                    prefill_time = 0
                    decode_time = 0
                    output_tokens = 0
                    audio_token_length = 0
                
                sample_result = {
                    "audio_file": os.path.basename(wav_path),
                    "speaker_id": item["speaker_id"],
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "extracted_answer": predicted_age_group,
                    "is_correct": is_correct,
                    "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                    "audio_tokens": audio_token_length,
                    "output_tokens": output_tokens,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
                results["samples"].append(sample_result)
                
                if 'inputs' in locals():
                    del inputs
                if 'audio_chunks' in locals():
                    del audio_chunks
                if 'result' in locals():
                    del result
                
                torch.cuda.empty_cache()
                
                if sample_count % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                update_interval = 10 if is_screen_env else 1
                
                if sample_count % update_interval == 0 or sample_count == total_samples:
                    current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                    
                    pbar_folders.set_postfix_str(
                        f"Samples:{sample_count}/{total_samples}, Accuracy:{current_accuracy:.2%}"
                    )
                    
                    if is_screen_env:
                        print(f"  Progress: {sample_count}/{total_samples} ({sample_count/total_samples*100:.1f}%), "
                              f"Accuracy: {current_accuracy:.2%}")
                        sys.stdout.flush()
            
            pbar_folders.update()
            
            if folder_count % 10 == 0:
                allocated, reserved = get_gpu_memory_usage()
                if debug_mode:
                    print(f"  [Speaker {folder_count}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    total_samples = results["summary"]["total_samples"]
    timing_sample_count = max(0, total_samples - len(grouped_samples))
    if timing_sample_count > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / timing_sample_count
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / timing_sample_count
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / timing_sample_count
    else:
        results["summary"]["timing"]["avg_prefill_time"] = 0
        results["summary"]["timing"]["avg_decode_time"] = 0
        results["summary"]["timing"]["avg_total_time"] = 0
    
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    
    for age_group, stats in results["summary"]["age_group_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)
    
    print("\n=== VoxCeleb Age Classification Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={method_is}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    
    print("\nAccuracy by age group:")
    for age_group, stats in results["summary"]["age_group_stats"].items():
        print(f"  {age_group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"\nAverage inference time: {results['summary']['timing']['avg_total_time']:.4f}s (excluding warm-up samples)")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f}s (excluding warm-up samples)")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f}s (excluding warm-up samples)")
    
    cuda_event_stats.print_statistics()
    
    print(f"Results saved to: {output_file}")
    print(f"Timing stats saved to: {timing_output_file}")
    print(f"CUDA Event stats saved to: {cuda_event_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()