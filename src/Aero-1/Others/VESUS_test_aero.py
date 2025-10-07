import os
import sys
import json
import time
import random
import warnings
import gc
import traceback
import librosa
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any, Tuple

try:
    import torch
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from transformers import logging
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please run: pip install torch transformers soundfile numpy librosa")
    sys.exit(1)

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Disable transformers warnings
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

# Set flags based on method name
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
data_path = os.environ.get("VESUS_DATA_PATH", 
    "/data/to/your/vesus_dataset/path")
emotion_json_file = os.path.join(data_path, "audio_emotion_dataset.json")
result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/results/path')
os.makedirs(result_dir, exist_ok=True)

# Output file paths
output_file = f'{result_dir}/VESUS_Aero1_results_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
timing_output_file = f'{result_dir}/VESUS_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
cuda_event_output_file = f'{result_dir}/VESUS_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'

print(f"Results will be saved to: {output_file}")
print(f"Timing stats will be saved to: {timing_output_file}")
print(f"CUDA Event stats will be saved to: {cuda_event_output_file}")

class VESUSTimingStats:
    """Track inference timing statistics for VESUS emotion recognition task"""
    def __init__(self):
        self.timing_records = []
        self.emotion_stats = defaultdict(list)
        self.person_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   emotion_label=None, person_id=None):
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
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "emotion_label": emotion_label,
            "person_id": person_id
        }
        
        self.timing_records.append(record)
        
        if emotion_label:
            self.emotion_stats[emotion_label].append(record)
        
        if person_id:
            self.person_stats[person_id].append(record)
    
    def get_summary(self):
        """Get overall statistics summary"""
        if self.total_samples == 0:
            return {"error": "No samples processed"}
        
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
            "avg_tokens_per_sec": avg_tokens_per_sec
        }
        
        # Add emotion statistics
        emotion_summaries = {}
        for emotion, records in self.emotion_stats.items():
            if len(records) > 0:
                emotion_summaries[emotion] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "emotion_summaries": emotion_summaries
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
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
            print("No timing statistics data")
            return
        
        print("\n=== CUDA Event Timing Statistics ===")
        
        # Prefill statistics
        prefill_stats = self.get_time_statistics(self.prefill_times, "prefill")
        print(f"Prefill timing statistics:")
        print(f"  Average: {prefill_stats['prefill_avg']:.6f}s")
        
        # Decode statistics
        decode_stats = self.get_time_statistics(self.decode_times, "decode")
        print(f"Decode timing statistics:")
        print(f"  Average: {decode_stats['decode_avg']:.6f}s")
        
        # Total statistics
        total_stats = self.get_time_statistics(self.total_times, "total")
        print(f"Total timing statistics:")
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
    """Downsample audio to target sampling rate"""
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
    """Load audio with librosa and split, compatible with Aero-1 official sample"""
    
    try:
        # Construct full audio file path
        full_audio_path = os.path.join(data_path, audio_path)
        
        if not os.path.exists(full_audio_path):
            if debug_mode:
                print(f"Audio file does not exist: {full_audio_path}")
            return None
        
        # Use librosa to load audio (recommended by official)
        audio, sample_rate = librosa.load(full_audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # If audio sampling rate does not match, downsample
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # If audio length exceeds 30 seconds, split
        if len(audio) > 480000:  # 30 seconds @ 16kHz
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30s limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # If audio length does not exceed 30s, return a single chunk in a list
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        if debug_mode:
            traceback.print_exc()
        return None

def load_vesus_dataset(json_file_path):
    """Load VESUS emotion recognition dataset"""
    if not os.path.exists(json_file_path):
        print(f"Error: Dataset file does not exist: {json_file_path}")
        return []
    
    print(f"Loading VESUS emotion dataset: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter valid items
        valid_samples = []
        for item in data:
            if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
                valid_samples.append(item)
        
        print(f"Loaded {len(valid_samples)} valid samples")
        
        # Count emotion distribution
        emotion_counts = defaultdict(int)
        for sample in valid_samples:
            emotion = sample.get('emotion_label', 'unknown')
            emotion_counts[emotion] += 1
        
        print(f"Emotion distribution: {dict(emotion_counts)}")
        
        return valid_samples
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []

def extract_emotion_answer(text, choices):
    """Extract emotion answer from model output text"""
    text_lower = text.lower().strip()
    
    # Directly detect a/b/c/d answers
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return "A"
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return "B"
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return "C"
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return "D"
    
    # Check for explicit option indicators
    option_patterns = {
        'A': ["option a", "choice a", "a)", "(a)"],
        'B': ["option b", "choice b", "b)", "(b)"],
        'C': ["option c", "choice c", "c)", "(c)"],
        'D': ["option d", "choice d", "d)", "(d)"]
    }
    
    for option, patterns in option_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return option
    
    # Try matching by emotion keywords
    emotion_keywords = {
        'angry': ['anger', 'frustrated', 'mad', 'furious'],
        'happy': ['joy', 'cheerful', 'pleased', 'delighted'],
        'sad': ['sadness', 'melancholy', 'depressed', 'sorrow'],
        'fearful': ['fear', 'anxiety', 'scared', 'afraid'],
        'monotone': ['flat', 'emotionless', 'neutral', 'bland']
    }
    
    # Check options for emotion keywords
    for choice_key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
        if choice_key in choices:
            choice_text = choices[choice_key].lower()
            for emotion, keywords in emotion_keywords.items():
                if emotion in choice_text or any(keyword in choice_text for keyword in keywords):
                    if any(keyword in text_lower for keyword in keywords) or emotion in text_lower:
                        return choice_key[-1].upper()  # Return A/B/C/D
    
    return ""

def cuda_timing_inference(model, processor, inputs, max_new_tokens=64):
    """
    Inference function using CUDA Event API for precise GPU timing
    """
    
    # Ensure GPU idle before precise timing
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
        
        # === Calculate precise time differences ===
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
    print(f"\n=== VESUS Emotion Recognition Evaluation Config (Aero-1) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Data path: {data_path}")
    print(f"JSON file: {emotion_json_file}")
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

    # Load VESUS dataset
    samples = load_vesus_dataset(emotion_json_file)
    
    if not samples:
        print("Error: No data samples found")
        return
    
    # Apply sample limit
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, processing {len(samples)} samples")

    # Print initial memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"After model loading, GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # Create timing stats trackers
    timing_stats = VESUSTimingStats()
    cuda_event_stats = CudaEventTimingStats()

    results = []
    total_correct = 0
    emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    person_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    print(f"Begin evaluating {len(samples)} samples...")
    
    # Detect if running in screen or non-interactive environment
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
    
    # Set tqdm parameters
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

    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="VESUS Evaluation (Aero-1)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        # Predefine variables to avoid UnboundLocalError
        prefill_time = 0
        decode_time = 0
        output_tokens = 0
        audio_token_length = 0
        predicted_answer = ""
        is_correct = False
        resp = ""
        
        try:
            # Load audio data
            audio_path = sample.get("path", "")
            audio_result = prepare_audio_for_processor(audio_path)
            
            if audio_result is None:
                continue
            
            audio_chunks, sample_rate = audio_result
            
            # Get sample info
            emotion_label = sample.get("emotion_label", "unknown")
            person_id = sample.get("person_id", "unknown")
            answer_gt = sample.get("answer_gt", "").upper()
            
            # Use official message format - support multiple audio chunks
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Add audio content to message for each chunk
            for chunk in audio_chunks:
                messages[0]["content"].append({
                    "type": "audio",
                    "audio": "placeholder",  # Will be replaced with actual audio
                })
            
            # Build prompt for emotion recognition
            question = sample.get("question", "What emotion is expressed in this audio segment?")
            choice_a = sample.get("choice_a", "")
            choice_b = sample.get("choice_b", "")
            choice_c = sample.get("choice_c", "")
            choice_d = sample.get("choice_d", "")
            
            prompt_text = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D)."""
            
            # Add text content
            messages[0]["content"].append({
                "type": "text",
                "text": prompt_text
            })
            
            # Apply chat template to messages
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process text and audio chunks with processor
            inputs = processor(
                text=prompt,
                audios=audio_chunks,  # Pass audio chunks list
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to("cuda")
            
            # Compute audio token length (for stats)
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
            
            # Show detailed info in debug mode only
            if debug_mode:
                print(f"Processing audio: {os.path.basename(audio_path)}")
                print(f"Emotion label: {emotion_label}")
                print(f"Person ID: {person_id}")
                print(f"Audio chunks: {len(audio_chunks)}")
                print(f"Question: {question}")
                print(f"Choices: A:{choice_a}, B:{choice_b}, C:{choice_c}, D:{choice_d}")
                print(f"Ground truth answer: {answer_gt}")
                print(f"Estimated audio token length: {audio_token_length}")
                sys.stdout.flush()
            
            # Inference with CUDA Event precise timing
            result = cuda_timing_inference(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=64
            )
            
            # Get results
            resp = result['response_text']
            prefill_time = result['prefill_time']
            decode_time = result['decode_time']
            total_time = result['total_time']
            output_tokens = result['output_tokens']

            # Extract answer
            predicted_answer = extract_emotion_answer(resp, sample)
            is_correct = (predicted_answer == answer_gt)

            if is_correct:
                total_correct += 1
            
            # Update stats
            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1
            
            if is_correct:
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1

            # Show detailed results in debug mode only
            if debug_mode:
                print(f"Model output: '{resp}'")
                print(f"Extracted answer: '{predicted_answer}'")
                print(f"Ground truth answer: '{answer_gt}'")
                print(f"Answer correct: {is_correct}")
                print(f"Inference time: Total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                print(f"Output tokens: {output_tokens}")
                print("=" * 50)
                sys.stdout.flush()

            # Save detailed results
            results.append({
                "idx": idx,
                "path": audio_path,
                "emotion_label": emotion_label,
                "person_id": person_id,
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "answer_gt": answer_gt,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response_text": resp,
                "audio_chunks": len(audio_chunks),
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time
            })

            # Collect timing info
            timing_stats.add_record(
                prefill_time, decode_time, 
                output_tokens,
                inputs["input_ids"].shape[1],
                emotion_label, person_id
            )
            
            # Collect CUDA Event timing stats
            cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)

            # Memory cleanup
            if 'inputs' in locals():
                del inputs
            if 'audio_chunks' in locals():
                del audio_chunks
            if 'result' in locals():
                del result
            
            torch.cuda.empty_cache()
            
            # Deep cleanup every 10 samples
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Print memory usage every 100 samples
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            if debug_mode:
                traceback.print_exc()
            
            # Set error status variable values
            predicted_answer = "ERROR"
            is_correct = False
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            resp = "ERROR"
            
            # Save error results
            results.append({
                "idx": idx,
                "path": sample.get("path", ""),
                "emotion_label": sample.get("emotion_label", "unknown"),
                "person_id": sample.get("person_id", "unknown"),
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "answer_gt": sample.get("answer_gt", "").upper(),
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response_text": resp,
                "audio_chunks": 1,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            })
            
            continue
        
        # Update progress bar
        update_interval = 50 if is_screen_env else 20
        sample_count = idx + 1
        
        if sample_count % update_interval == 0 or sample_count == len(samples):
            current_accuracy = total_correct / sample_count
            progress_bar.set_postfix_str(
                f"Accuracy:{current_accuracy:.3f}, Emotion:{emotion_label[:8]}, Person:{person_id}"
            )
            
            if is_screen_env:
                print(f"Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                      f"Accuracy: {current_accuracy:.3f}")
                sys.stdout.flush()
        
        progress_bar.update()
        
        if is_screen_env and sample_count % 25 == 0:
            sys.stdout.flush()

    # Calculate final statistics
    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate accuracy by emotion
    emotion_accuracies = {}
    for emotion, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracies[emotion] = stats["correct"] / stats["total"]

    # Calculate accuracy by person
    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]

    # Create result summary
    summary = {
        "total_samples": total_samples,
        "correct_samples": total_correct,
        "overall_accuracy": overall_accuracy,
        "emotion_stats": dict(emotion_stats),
        "emotion_accuracies": emotion_accuracies,
        "person_stats": dict(person_stats),
        "person_accuracies": person_accuracies,
        "config": {
            "model_name": "Aero-1-Audio-1.5B",
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file
        },
        "timing": timing_stats.get_summary()
    }

    # Save results
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Save timing stats
    timing_stats.export_to_json(timing_output_file)
    
    # Save CUDA Event stats to separate file
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)

    # Output result summary
    print(f"\n=== VESUS Emotion Recognition Evaluation Result Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={method_is}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"Emotion accuracies:")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\nAverage inference time: {overall_summary.get('avg_total_time', 0):.4f}s")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/sec")
    
    # Show CUDA Event detailed stats
    cuda_event_stats.print_statistics()
    
    print(f"Results saved to: {output_file}")
    print(f"Timing stats saved to: {timing_output_file}")
    print(f"CUDA Event stats saved to: {cuda_event_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()