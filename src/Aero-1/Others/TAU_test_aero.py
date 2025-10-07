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

# Data path configuration - change to TAU dataset path
data_path_root = os.environ.get("TAU_DATA_PATH", '/data/to/your/tau/root')
audio_dir = os.path.join(data_path_root, 'concatenated_resampled')  # Resampled audio directory
result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/tau/results')
os.makedirs(result_dir, exist_ok=True)

# Change output file paths and naming
output_file = f'{result_dir}/TAU_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
timing_output_file = f'{result_dir}/TAU_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
cuda_event_output_file = f'{result_dir}/TAU_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'

print(f"Results will be saved to: {output_file}")
print(f"Timing statistics will be saved to: {timing_output_file}")
print(f"CUDA Event statistics will be saved to: {cuda_event_output_file}")

# Timing statistics class
class FolderTimingStats:
    """Track per-folder inference timing statistics"""
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
        """Export statistics to JSON file"""
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
        """Calculate time statistics (average only)"""
        if not times_list:
            return {}
        
        stats = {
            f"{name}_avg": sum(times_list) / len(times_list),
            f"{name}_count": len(times_list)
        }
        return stats
    
    def get_full_statistics(self):
        """Get full time statistics"""
        stats = {}
        stats.update(self.get_time_statistics(self.prefill_times, "prefill"))
        stats.update(self.get_time_statistics(self.decode_times, "decode"))
        stats.update(self.get_time_statistics(self.total_times, "total"))
        return stats
    
    def print_statistics(self):
        """Print time statistics (average only)"""
        if not self.timing_records:
            print("No timing statistics data")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        
        # Prefill statistics
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill time statistics:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        
        # Decode statistics
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode time statistics:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        
        # Total statistics
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total time statistics:")
        print(f"  Average: {total_stats['total_avg']:.6f}s")
        print(f"  Sample count: {total_stats['total_count']}")

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
    """Load audio with librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Load audio with librosa (official recommended way)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate does not match
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split audio if longer than 30s
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

def load_tau_acoustic_scene_dataset(root_dir):
    """Load acoustic scene classification task from TAU dataset"""
    # Load metadata JSON file
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found: {meta_file}")
        return [], {}
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    # Scene class count
    scene_counts = {}
    
    # Iterate through each entry in metadata
    for item in metadata:
        # Get audio file path
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)
        
        # Check if file exists
        if not os.path.exists(wav_path):
            if debug_mode:
                print(f"Warning: File not found {wav_path}")
            continue
        
        # Extract scene label and correct option
        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"] # A, B, C, D
        
        # Update scene class count
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        
        # Build sample info
        all_samples.append({
            "scene_label": scene_label,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification"
        })
    
    print(f"Total {len(all_samples)} valid audio samples loaded")
    
    # Show scene distribution
    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count} samples ({count/len(all_samples)*100:.1f}%)")
    
    # Sample limit handling
    if sample_limit > 0 and sample_limit < len(all_samples):
        print(f"Randomly selecting {sample_limit} samples for evaluation due to sample limit setting")
        all_samples = random.sample(all_samples, sample_limit)
        
    # Shuffle samples
    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """Extract acoustic scene answer option (A/B/C/D) from model output text"""
    text_lower = text.lower().strip()
    
    # Directly detect a/b/c/d answer
    options = ['a', 'b', 'c', 'd']
    
    # Exact match: if output is just one of a/b/c/d
    if text_lower in options:
        return text_lower.upper()
    
    # Check start match: "a.", "a)", "a:" etc.
    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    
    # Check if contains explicit option indicator
    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    
    # If choices exist, check if choice text is in answer
    if choices:
        best_match = None
        max_overlap = 0
        
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()
            # Prefer full choice name in text
            if choice_lower in text_lower:
                return chr(65 + i)  # A, B, C, D
            
            # Check important keywords in text
            keywords = choice_lower.split(' - ')[0].split()  # Use first part as keywords
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        
        if best_match and max_overlap > 1:  # At least 2 keywords match
            return best_match
    
    # If unable to determine, return empty string
    return ""

def group_samples_by_scene(samples):
    """Group samples by scene"""
    grouped = {}
    for sample in samples:
        scene = sample["scene_label"]
        if scene not in grouped:
            grouped[scene] = []
        grouped[scene].append(sample)
    return grouped

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference function using CUDA Event API for precise GPU timing measurement
    """
    
    # Ensure GPU is idle for precise timing
    torch.cuda.synchronize()
    
    # Create CUDA events
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        # === Stage 1: Prefill timing ===
        event_start.record()
        
        # Prefill computation
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
        
        # === Event synchronization ===
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
    print(f"\n=== TAU Acoustic Scene Classification Evaluation Config (Aero-1) ===")
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
        print(f"Note: Official Aero-1 model may not support audio pruning")
        print(f"Pruning config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    
    # Create timing stats
    timing_stats = FolderTimingStats()
    cuda_event_stats = CudaEventTimingStats()
    
    # Scan TAU dataset
    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    
    if not samples:
        print("Error: Failed to load any samples")
        return
    
    # Group by scene
    grouped_samples = group_samples_by_scene(samples)
    
    # Print initial memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"After model load, GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    # Create result data structure
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "scene_stats": {scene: {"total": 0, "correct": 0} for scene in scene_counts},
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
    
    # Detect if running in screen or non-interactive environment
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
    
    # Setup tqdm parameters
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
    
    # Create main progress bar for outer loop
    with tqdm(total=len(grouped_samples), desc="Processing scene groups", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        
        # Process each scene group
        for scene_label, items in grouped_samples.items():
            folder_count += 1
            # Update progress bar description
            pbar_folders.set_description(f"Processing scene [{folder_count}/{total_folders}]: {scene_label}")
            
            # Set current processing folder
            timing_stats.set_current_folder(scene_label)
            
            # Remove inner tqdm progress bar, use simple counter instead
            sample_count = 0
            total_samples = len(items)

            # Iterate through all samples in current scene
            for i, item in enumerate(items):
                sample_count = i + 1
                wav_path = item['wav_path']
                
                # Predefine variables to avoid UnboundLocalError
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                audio_token_length = 0
                predicted_answer = ""
                is_correct = False
                ground_truth = item["answer_gt"].upper()
                
                try:
                    # Use official message format - supports multiple audio chunks
                    messages = [
                        {
                            "role": "user",
                            "content": []
                        }
                    ]
                    
                    # Prepare audio input - now returns list of audio chunks
                    audio_chunks, sample_rate = prepare_audio_for_processor(wav_path)
                    
                    # For each audio chunk, add audio content to message
                    for chunk in audio_chunks:
                        messages[0]["content"].append({
                            "type": "audio",
                            "audio": "placeholder",  # This will be replaced with actual audio
                        })
                    
                    # Build acoustic scene classification prompt
                    instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option."
                    option_text = f"A: {item['choice_a']}\nB: {item['choice_b']}\nC: {item['choice_c']}\nD: {item['choice_d']}"
                    format_instruction = "Respond with only the letter of your answer (A, B, C, or D)."
                    
                    # Add text content
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"{instruction}\n\n{option_text}\n\n{format_instruction}"
                    })
                    
                    # Use chat template to process messages
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    
                    # Use processor to process text and audio chunks
                    inputs = processor(
                        text=prompt,
                        audios=audio_chunks,  # Pass list of audio chunks
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    # Compute audio token length (for statistics)
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
                    
                    # Show detailed info only in debug mode
                    if debug_mode:
                        print(f"Processing audio: {os.path.basename(wav_path)}")
                        print(f"Scene label: {scene_label}")
                        print(f"Number of audio chunks: {len(audio_chunks)}")
                        print(f"Options: A:{item['choice_a']}, B:{item['choice_b']}, C:{item['choice_c']}, D:{item['choice_d']}")
                        print(f"Ground truth: {ground_truth}")
                        print(f"Estimated audio token length: {audio_token_length}")
                        sys.stdout.flush()
                    
                    # Inference with CUDA Event precise timing
                    result = cuda_timing_inference(
                        model=model,
                        processor=processor,
                        inputs=inputs,
                        max_new_tokens=10
                    )
                    
                    # Get results
                    output = result['response_text']
                    prefill_time = result['prefill_time']
                    decode_time = result['decode_time']
                    total_time = result['total_time']
                    output_tokens = result['output_tokens']
                    
                    # Extract answer - more robust way
                    choices = [item['choice_a'], item['choice_b'], item['choice_c'], item['choice_d']]
                    predicted_answer = extract_acoustic_scene_answer(output, choices)
                    
                    # Check if answer is correct
                    is_correct = predicted_answer == ground_truth
                    
                    # Show detailed results only in debug mode
                    if debug_mode:
                        print(f"Model output: '{output}'")
                        print(f"Extracted answer: '{predicted_answer}'")
                        print(f"Ground truth: '{ground_truth}'")
                        print(f"Correct: {is_correct}")
                        print(f"Inference time: total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                        print(f"Output tokens: {output_tokens}")
                        print("=" * 50)
                        sys.stdout.flush()
                    
                    # Update statistics
                    results["summary"]["total_samples"] += 1
                    results["summary"]["scene_stats"][scene_label]["total"] += 1
                    
                    if is_correct:
                        results["summary"]["correct_samples"] += 1
                        results["summary"]["scene_stats"][scene_label]["correct"] += 1
                    
                    # Update timing statistics
                    results["summary"]["timing"]["total_prefill_time"] += prefill_time
                    results["summary"]["timing"]["total_decode_time"] += decode_time
                    results["summary"]["timing"]["total_total_time"] += total_time
                    
                    # Add timing record
                    timing_stats.add_record(prefill_time, decode_time, output_tokens)
                    
                    # Collect CUDA Event specific timing statistics
                    cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    if debug_mode:
                        traceback.print_exc()
                    output = "ERROR"
                    predicted_answer = "ERROR"
                    is_correct = False
                    prefill_time = 0
                    decode_time = 0
                    output_tokens = 0
                    audio_token_length = 0
                
                # Save sample result
                sample_result = {
                    "audio_file": os.path.basename(wav_path),
                    "scene_label": scene_label,
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "extracted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                    "audio_tokens": audio_token_length,
                    "output_tokens": output_tokens,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
                
                # Add to results list
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
                
                # In screen environment, update every 10 samples; in standard, update every sample
                update_interval = 10 if is_screen_env else 1
                
                if sample_count % update_interval == 0 or sample_count == total_samples:
                    # Calculate accuracy
                    current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                    
                    # Update outer progress bar suffix
                    pbar_folders.set_postfix_str(
                        f"Sample:{sample_count}/{total_samples}, Accuracy:{current_accuracy:.2%}"
                    )
                    
                    if is_screen_env:
                        # Extra progress print in screen environment
                        print(f"  Progress: {sample_count}/{total_samples} ({sample_count/total_samples*100:.1f}%), "
                              f"Accuracy: {current_accuracy:.2%}")
                        sys.stdout.flush()
            
            # After processing a scene, update outer progress bar
            pbar_folders.update()
            
            # Memory cleanup every 3 scenes
            if folder_count % 3 == 0:
                allocated, reserved = get_gpu_memory_usage()
                if debug_mode:
                    print(f"  [Scene {folder_count}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    # Calculate average times
    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / total_samples
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / total_samples
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / total_samples
    
    # Calculate overall and per-scene accuracy
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    
    for scene in results["summary"]["scene_stats"]:
        stats = results["summary"]["scene_stats"][scene]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0
    
    # Save results as single JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save timing statistics
    timing_stats.export_to_json(timing_output_file)
    
    # Save CUDA Event statistics to separate file
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)
    
    # Print result summary
    print("\n=== TAU Acoustic Scene Classification Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={method_is}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    
    # Print scene-level results sorted by accuracy
    sorted_scenes = sorted(
        [(scene, stats["accuracy"], stats["correct"], stats["total"]) 
         for scene, stats in results["summary"]["scene_stats"].items()],
        key=lambda x: x[1], reverse=True
    )
    
    print("\nScene accuracy:")
    for scene, acc, correct, total in sorted_scenes:
        print(f"  {scene}: {acc:.2%} ({correct}/{total})")
    
    print(f"\nAverage inference time: {results['summary']['timing']['avg_total_time']:.4f} seconds")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} seconds")
    
    # Show CUDA Event detailed statistics
    cuda_event_stats.print_statistics()
    
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    print(f"CUDA Event statistics saved to: {cuda_event_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()