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
    print("Debug mode enabled - detailed output will be shown")

# Data path configuration
data_path_root = os.environ.get("VOXCELEB_DATA_PATH", 
    '/data/to/your/concatenated_audio/path')
result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/Vox_Results/path')
os.makedirs(result_dir, exist_ok=True)

# Output file path and naming
output_file = f'{result_dir}/VoxCeleb_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
timing_output_file = f'{result_dir}/VoxCeleb_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
cuda_event_output_file = f'{result_dir}/VoxCeleb_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'

print(f"Results will be saved to: {output_file}")
print(f"Timing stats will be saved to: {timing_output_file}")
print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")

# Timing stats class
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
    """CUDA Event batch timing stats class"""
    
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
        """Calculate timing statistics (average only)"""
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
            print("No timing data available")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        
        # Prefill stats
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill timing stats:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        
        # Decode stats
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode timing stats:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        
        # Total stats
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total timing stats:")
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
    """Split audio into 30-second chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio by librosa and split into chunks, compatible with Aero-1 official example"""
    
    try:
        # Use librosa to load audio (official recommended way)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate mismatch
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split audio if longer than 30 seconds
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30 seconds limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # If audio length ≤ 30s, return single chunk list
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        if debug_mode:
            traceback.print_exc()
        # Return silent chunk list
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_concatenated_audio_dataset(root_dir):
    """Load dataset from concatenated_audio dir, balance male/female samples using gender_id_task_meta.json"""
    # Load metadata JSON file
    meta_file = os.path.join(root_dir, "gender_id_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    # Iterate through metadata entries
    for item in metadata:
        # Get audio file path
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        # Check file existence
        if not os.path.exists(wav_path):
            if debug_mode:
                print(f"Warning: file not found {wav_path}")
            continue
        
        # Extract speaker ID and gender info
        speaker_id = item["speaker_id_original"]
        gender = item["answer_gt"].lower().strip()
        
        # Construct sample info
        all_samples.append({
            "speaker_id": speaker_id,
            "gender": gender,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "answer_gt": gender,
            "task": "Speaker_Gender_Identification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    
    # Group samples by gender
    male_samples = [sample for sample in all_samples if sample["gender"].lower() == "male"]
    female_samples = [sample for sample in all_samples if sample["gender"].lower() == "female"]
    print(f"Raw sample counts: male={len(male_samples)}, female={len(female_samples)}")
    
    # Find the minimum sample count per gender
    min_samples_per_gender = min(len(male_samples), len(female_samples))
    
    # Randomly sample equal number from majority gender
    if len(male_samples) > min_samples_per_gender:
        male_samples = random.sample(male_samples, min_samples_per_gender)
    
    if len(female_samples) > min_samples_per_gender:
        female_samples = random.sample(female_samples, min_samples_per_gender)
    
    # Merge balanced samples
    balanced_samples = male_samples + female_samples
    
    # Shuffle dataset randomly
    random.shuffle(balanced_samples)
    
    print(f"Balanced sample counts: male={len(male_samples)}, female={len(female_samples)}, total={len(balanced_samples)}")
    
    return balanced_samples

def extract_gender_answer(text, choice_a="male", choice_b="female"):
    """Extract gender answer from model output text, handle direct a/b replies"""
    text_lower = text.lower().strip()
    
    # Normalize choice values
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    
    # Directly detect a/b answers
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a_lower
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b_lower
        
    # Check for explicit a/b option indication
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a_lower
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b_lower
    
    # Check for direct inclusion of choice text
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a_lower
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b_lower
    
    # Try more precise pattern matching if still undecided
    import re
    if choice_a_lower == "male" and choice_b_lower == "female":
        # Use word boundaries for exact match
        male_match = re.search(r'\bmale\b', text_lower) is not None
        female_match = re.search(r'\bfemale\b', text_lower) is not None
        
        if male_match and not female_match:
            return "male"
        if female_match and not male_match:
            return "female"
    
    # If still undecided, return empty string
    return ""

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference function with precise GPU timing using CUDA Event API
    """
    
    # Ensure GPU idle for precise timing
    torch.cuda.synchronize()
    
    # Create CUDA Events
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        # === Stage 1: Prefill timing ===
        event_start.record()
        
        # Prefill compute
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, 
                           output_hidden_states=False, return_dict=True)
        
        event_prefill_end.record()
        
        # === Stage 2: Full Generation timing ===
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
        
        # === Event synchronize ===
        event_start.synchronize()
        event_prefill_end.synchronize()
        event_total_end.synchronize()
        
        # === Calculate precise time difference ===
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
    print(f"\n=== VoxCeleb Gender Identification Evaluation Config (Aero-1) ===")
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
    
    # Model path config - use official model name
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

    # Load model
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

    # Set audio pruning config
    if prune_ratio > 0:
        print(f"Note: Using official Aero-1 model, audio pruning may not be available")
        print(f"Pruning config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    
    # Create timing stat trackers
    timing_stats = FolderTimingStats()
    cuda_event_stats = CudaEventTimingStats()
    
    # Scan VoxCeleb dataset
    samples = load_concatenated_audio_dataset(data_path_root)
    
    if not samples:
        print("Error: No samples loaded")
        return
    
    # Apply sample limit
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample limit applied, processing {len(samples)} samples")
    
    # Group by speaker ID instead of gender
    grouped_samples = {}
    for sample in samples:
        speaker_id = sample["speaker_id"]
        if speaker_id not in grouped_samples:
            grouped_samples[speaker_id] = []
        grouped_samples[speaker_id].append(sample)
    
    # Count male/female
    male_count = sum(1 for s in samples if s["gender"].lower() == "male")
    female_count = sum(1 for s in samples if s["gender"].lower() == "female")
    print(f"Gender stats: male samples={male_count}, female samples={female_count}")
    
    # Print initial memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    
    # Create result data structure
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "male_total": 0,
            "male_correct": 0,
            "female_total": 0,
            "female_correct": 0,
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
    
    # Detect if running in screen or non-interactive env
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive env, using simplified progress display")
        sys.stdout.flush()
    
    # Set tqdm arguments
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
    
    # Create overall progress bar for outer loop
    with tqdm(total=len(grouped_samples), desc="Processing speaker groups", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        
        # Loop over speaker groups
        for speaker_id, items in grouped_samples.items():
            folder_count += 1
            # Update progress bar description
            pbar_folders.set_description(f"Processing speaker [{folder_count}/{total_folders}]: {speaker_id}")
            
            # Set current folder for timing stats
            timing_stats.set_current_folder(speaker_id)
            
            # Remove inner tqdm, use simple counter instead
            sample_count = 0
            total_samples = len(items)

            # Iterate over all samples for current speaker
            for i, item in enumerate(items):
                sample_count = i + 1
                wav_path = item['wav_path']
                task = item.get('task', 'Speaker_Gender_Identification')
                
                # Predefine variables to avoid UnboundLocalError
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                audio_token_length = 0
                predicted_gender = ""
                is_correct = False
                ground_truth = item["gender"].lower().strip()
                
                try:
                    # Use official message format - support multi-audio chunks
                    messages = [
                        {
                            "role": "user",
                            "content": []
                        }
                    ]
                    
                    # Prepare audio input - returns audio chunks list
                    audio_chunks, sample_rate = prepare_audio_for_processor(wav_path)
                    
                    # Add audio content for each chunk to message
                    for chunk in audio_chunks:
                        messages[0]["content"].append({
                            "type": "audio",
                            "audio": "placeholder",  # will be replaced by actual audio
                        })
                    
                    # Use dedicated gender identification prompt
                    instruction = "Listen to this audio and identify the speaker's gender. Is this a male or female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'."
                    
                    # Add text content
                    messages[0]["content"].append({
                        "type": "text",
                        "text": instruction
                    })
                    
                    # Process message by chat template
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    
                    # Use processor for text and audio chunks
                    inputs = processor(
                        text=prompt,
                        audios=audio_chunks,
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    # Calculate audio token length (for stats)
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
                    
                    # Only show detailed info in debug mode
                    if debug_mode:
                        print(f"Processing audio: {os.path.basename(wav_path)}")
                        print(f"Speaker ID: {speaker_id}")
                        print(f"Audio chunks: {len(audio_chunks)}")
                        print(f"Gender label: {ground_truth}")
                        print(f"Estimated audio token length: {audio_token_length}")
                        sys.stdout.flush()
                    
                    # Inference with CUDA Event precise timing
                    result = cuda_timing_inference(
                        model=model,
                        processor=processor,
                        inputs=inputs,
                        max_new_tokens=10
                    )
                    
                    # Get result
                    output = result['response_text']
                    prefill_time = result['prefill_time']
                    decode_time = result['decode_time']
                    total_time = result['total_time']
                    output_tokens = result['output_tokens']
                    
                    # Extract answer - more robust way
                    predicted_gender = extract_gender_answer(output)
                    
                    # Check if answer is correct
                    is_correct = predicted_gender == ground_truth
                    
                    # Only show detailed result in debug mode
                    if debug_mode:
                        print(f"Model output: '{output}'")
                        print(f"Extracted answer: '{predicted_gender}'")
                        print(f"Ground truth: '{ground_truth}'")
                        print(f"Correct: {is_correct}")
                        print(f"Inference time: total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                        print(f"Output tokens: {output_tokens}")
                        print("=" * 50)
                        sys.stdout.flush()
                    
                    # Update stats
                    results["summary"]["total_samples"] += 1
                    if ground_truth in ["male", "female"]:
                        results["summary"][f"{ground_truth}_total"] += 1
                        if is_correct:
                            results["summary"][f"{ground_truth}_correct"] += 1
                            results["summary"]["correct_samples"] += 1
                    
                    # Update timing stats
                    results["summary"]["timing"]["total_prefill_time"] += prefill_time
                    results["summary"]["timing"]["total_decode_time"] += decode_time
                    results["summary"]["timing"]["total_total_time"] += total_time
                    
                    # Record timing stats
                    timing_stats.add_record(prefill_time, decode_time, output_tokens)
                    
                    # Collect CUDA Event-specific timing stats
                    cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    if debug_mode:
                        traceback.print_exc()
                    output = "ERROR"
                    predicted_gender = "error"
                    is_correct = False
                    prefill_time = 0
                    decode_time = 0
                    output_tokens = 0
                    audio_token_length = 0
                
                # Save sample result
                sample_result = {
                    "audio_file": os.path.basename(wav_path),
                    "speaker_id": item["speaker_id"],
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "extracted_answer": predicted_gender,
                    "is_correct": is_correct,
                    "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                    "audio_tokens": audio_token_length,
                    "output_tokens": output_tokens,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
                
                # Add to result list
                results["samples"].append(sample_result)
                
                # Memory cleanup
                if 'inputs' in locals():
                    del inputs
                if 'audio_chunks' in locals():
                    del audio_chunks
                if 'result' in locals():
                    del result
                
                torch.cuda.empty_cache()
                
                # Deep cleanup every 10 samples
                if sample_count % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Update progress every 10 samples in screen env, every sample in normal env
                update_interval = 10 if is_screen_env else 1
                
                if sample_count % update_interval == 0 or sample_count == total_samples:
                    # Calculate accuracy
                    current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                    
                    # Update progress bar postfix
                    pbar_folders.set_postfix_str(
                        f"Samples:{sample_count}/{total_samples}, Accuracy:{current_accuracy:.2%}"
                    )
                    
                    if is_screen_env:
                        # Print extra progress line in screen env
                        print(f"  Progress: {sample_count}/{total_samples} ({sample_count/total_samples*100:.1f}%), "
                              f"Accuracy: {current_accuracy:.2%}")
                        sys.stdout.flush()
            
            # Folder done, update outer progress bar
            pbar_folders.update()
            
            # Deep cleanup every 10 speakers
            if folder_count % 10 == 0:
                allocated, reserved = get_gpu_memory_usage()
                if debug_mode:
                    print(f"  [Speaker {folder_count}] GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    
    # Calculate average timings
    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / total_samples
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / total_samples
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / total_samples
    
    # Calculate accuracy
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    results["summary"]["male_accuracy"] = results["summary"]["male_correct"] / results["summary"]["male_total"] if results["summary"]["male_total"] > 0 else 0
    results["summary"]["female_accuracy"] = results["summary"]["female_correct"] / results["summary"]["female_total"] if results["summary"]["female_total"] > 0 else 0
    
    # Calculate precision, recall, F1 (female as positive)
    tp = results["summary"]["female_correct"]
    fp = results["summary"]["male_total"] - results["summary"]["male_correct"]
    fn = results["summary"]["female_total"] - results["summary"]["female_correct"]
    tn = results["summary"]["male_correct"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results["summary"]["precision"] = precision
    results["summary"]["recall"] = recall
    results["summary"]["f1_score"] = f1_score
    
    # Save results to single JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save timing stats
    timing_stats.export_to_json(timing_output_file)
    
    # Save CUDA Event stats to separate file
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)
    
    # Print result summary
    print("\n=== VoxCeleb Gender Identification Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={method_is}")
    print(f"Total samples: {total_samples}")
    print(f"Total accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Male accuracy: {results['summary']['male_accuracy']:.2%} ({results['summary']['male_correct']}/{results['summary']['male_total']})")
    print(f"Female accuracy: {results['summary']['female_accuracy']:.2%} ({results['summary']['female_correct']}/{results['summary']['female_total']})")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 score: {f1_score:.2%}")
    print(f"Average inference time: {results['summary']['timing']['avg_total_time']:.4f} s")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} s")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} s")
    
    # Show CUDA Event detailed stats
    cuda_event_stats.print_statistics()
    
    print(f"Results saved to: {output_file}")
    print(f"Timing stats saved to: {timing_output_file}")
    print(f"CUDA Event stats saved to: {cuda_event_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()