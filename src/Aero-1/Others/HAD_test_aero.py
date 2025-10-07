import os
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import numpy as np
import soundfile as sf
import warnings
import traceback
import time
import glob
import random
import sys
import io

# Import Aero model
import sys
import os

# Add required libraries
import librosa

print("Successfully imported required modules")

# Set output buffering - ensure real-time display
import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Get GPU ID
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")

# Audio pruning configuration
prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

# Set flag by method name
use_random = (prune_method == "random")
use_frame = (prune_method == "frame")
if use_random==False and use_frame==False:
    prune_method = "fast_v"
# Decide method name
if prune_ratio == 0:
    method_is = "base"
else:
    method_is = prune_method

# Sample limit (if provided)
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

# Debug mode switch
debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
if debug_mode:
    print("Debug mode enabled - verbose output will be shown")

# Data path configuration
data_path_root = '/data/to/your/HAD/concatenated_audio'  # Directory containing 'real' and 'fake' folders
result_dir = '/data/to/your/HAD_Results'
os.makedirs(result_dir, exist_ok=True)

# Change output file path and naming
output_file = f'{result_dir}/HAD_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/HAD_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing statistics will be saved to: {timing_output_file}")

# Timing statistics class
class FolderTimingStats:
    """Track inference timing statistics for each folder"""
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

# Audio processing functions
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
    SAMPLE_RATE = 16000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio using librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Load audio using librosa (official recommended method)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate mismatch
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split if audio longer than 30 seconds
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30 seconds limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # If audio length <= 30s, return single chunk list
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_had_dataset(root_dir):
    """Load HAD dataset, balance real and fake sample counts"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    
    all_samples = []
    
    # Load real audio samples
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "real",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "real",
                "task": "Audio_Authenticity_Detection"
            })
    
    # Load fake audio samples
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for wav_path in fake_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "fake",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "fake",
                "task": "Audio_Authenticity_Detection"
            })
    
    print(f"Total loaded audio samples: {len(all_samples)}")
    sys.stdout.flush()
    
    # Group samples by label
    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"Original sample counts: real={len(real_samples)}, fake={len(fake_samples)}")
    sys.stdout.flush()
    
    # Get minimum count per category
    min_samples_per_category = min(len(real_samples), len(fake_samples))
    
    # Randomly reduce larger group to match smaller
    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)
    
    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)
    
    # Merge balanced real and fake
    balanced_samples = real_samples + fake_samples
    
    # Shuffle dataset
    random.shuffle(balanced_samples)
    
    print(f"Balanced sample counts: real={len(real_samples)}, fake={len(fake_samples)}, total={len(balanced_samples)}")
    sys.stdout.flush()
    
    return balanced_samples

def extract_authenticity_answer(text):
    """Extract audio authenticity answer from model output text"""
    text_lower = text.lower().strip()
    
    # Direct detection of real/fake answer
    if 'real' in text_lower and 'fake' not in text_lower:
        return 'real'
    elif 'fake' in text_lower and 'real' not in text_lower:
        return 'fake'
    
    # Detect authenticity/artificial synonyms
    if any(word in text_lower for word in ['authentic', 'genuine', 'natural']) and \
       not any(word in text_lower for word in ['fake', 'artificial', 'synthetic']):
        return 'real'
    elif any(word in text_lower for word in ['artificial', 'synthetic', 'generated']) and \
         not any(word in text_lower for word in ['real', 'authentic', 'genuine']):
        return 'fake'
    
    # If unable to determine, return empty string
    return ""

def main():
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
        attn_implementation="sdpa",  # Use sdpa if flash_attention_2 is unavailable
        trust_remote_code=True
    )
    model.eval()
    print("Successfully loaded Aero-1 model")
    sys.stdout.flush()
    
    # Set audio pruning config - may need adjustment for official model
    if prune_ratio > 0:
        print(f"Note: Official Aero-1 model, audio pruning may not be available")
        print(f"Pruning config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    
    # Create timing stats tracker
    timing_stats = FolderTimingStats()
    
    # Scan HAD dataset
    samples = load_had_dataset(data_path_root)
    
    # Apply sample limit if set
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample limit applied, processing {len(samples)} samples")
    
    # Group samples by category for stats
    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)
    
    # Count real and fake samples
    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"Category stats: real samples={real_count}, fake samples={fake_count}")
    
    # Create results structure
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
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
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
    
    # Detect if running in 'screen' or non-interactive environment
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected 'screen' or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
    
    # Set tqdm parameters
    tqdm_kwargs = {
        'ascii': True,        # Use ASCII instead of Unicode
        'dynamic_ncols': True, # Auto terminal width
        'file': sys.stdout,    # Ensure direct output to stdout
        'mininterval': 0.1,   # Minimum update interval
        'maxinterval': 1.0,   # Maximum update interval
        'disable': False,     # Don't disable progress bar
        'leave': True,        # Keep progress bar
        'position': 0         # Progress bar position
    }
    
    # In 'screen', use more frequent updates
    if is_screen_env:
        tqdm_kwargs['mininterval'] = 0.05
        tqdm_kwargs['maxinterval'] = 0.5
    
    # Progress bar for samples
    with tqdm(total=len(samples), desc="Processing HAD audio samples (Aero-1)", **tqdm_kwargs) as pbar:
        
        # Set current folder for timing stats
        timing_stats.set_current_folder("HAD_Audio_Detection_Aero1")
        
        # Loop over samples
        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            label = item['label']
            task = item.get('task', 'Audio_Authenticity_Detection')
            
            # Predefine variables to avoid UnboundLocalError
            ground_truth = item["answer_gt"].lower().strip()
            output = ""
            predicted_label = ""
            is_correct = False
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            
            # Use official message format - support multiple audio chunks
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            try:
                # Prepare audio input - returns audio chunks list
                audio_chunks, sample_rate = prepare_audio_for_processor(audio_path)
                
                # Add audio chunk content to message
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",  # Will be replaced by actual audio
                    })
                
                # Add text content
                messages[0]["content"].append({
                    "type": "text",
                    "text": "Is this audio real or fake? Answer with 'real' or 'fake' only."
                })
                
                # Use chat template for messages
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                
                # Process text and audio chunks
                inputs = processor(
                    text=prompt,
                    audios=audio_chunks,  # Pass list of audio chunks
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to("cuda")
                
                # Compute audio token length (for statistics)
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
                
                # Verbose info in debug mode
                if debug_mode:
                    print(f"Processing audio: {os.path.basename(audio_path)}")
                    print(f"Audio chunk count: {len(audio_chunks)}")
                    print(f"Total audio length: {sum(len(chunk) for chunk in audio_chunks)}, sample rate: {sample_rate}")
                    print(f"Generated prompt: {prompt}")
                    print(f"Input IDs shape: {inputs.input_ids.shape}")
                    if hasattr(inputs, 'audio_features'):
                        print(f"Audio features shape: {inputs.audio_features.shape}")
                    print(f"Estimated audio token length: {audio_token_length}")
                    sys.stdout.flush()
                
                # Inference - separate prefill and decode
                start_time = time.time()
                
                with torch.no_grad():
                    # Prefill stage: forward pass
                    prefill_start = time.time()
                    
                    model_outputs = model(
                        **inputs,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    prefill_time = time.time() - prefill_start
                    
                    # Decode stage: actual generation
                    decode_start = time.time()
                    
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        use_cache=True
                    )
                    
                    decode_time = time.time() - decode_start
                
                total_time = time.time() - start_time
                
                # Decode output - only new generated part
                input_length = inputs['input_ids'].shape[1]
                new_tokens = generate_ids[:, input_length:]
                output_tokens = new_tokens.shape[1]
                
                # Decode new tokens
                output = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                
                # Extract answer
                predicted_label = extract_authenticity_answer(output)
                
                # Check answer correctness
                is_correct = predicted_label == ground_truth
                
                # Verbose info in debug mode
                if debug_mode:
                    print(f"Model output: '{output}'")
                    print(f"Inference time: total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                    print(f"Output tokens: {output_tokens}")
                    print(f"Extracted answer: '{predicted_label}'")
                    print(f"Ground truth: '{ground_truth}'")
                    print(f"Answer correct: {is_correct}")
                    print("=" * 50)
                    sys.stdout.flush()
                
                # Update stats
                results["summary"]["total_samples"] += 1
                if ground_truth in ["real", "fake"]:
                    results["summary"][f"{ground_truth}_total"] += 1
                    if is_correct:
                        results["summary"][f"{ground_truth}_correct"] += 1
                        results["summary"]["correct_samples"] += 1
                
                # Update timing
                results["summary"]["timing"]["total_prefill_time"] += prefill_time
                results["summary"]["timing"]["total_decode_time"] += decode_time
                results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                
                # Record timing stats
                timing_stats.add_record(prefill_time, decode_time, output_tokens)
                
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                output = "ERROR"
                predicted_label = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                audio_token_length = 0
            
            # Save sample result
            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "audio_label": label,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_label,
                "is_correct": is_correct,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1
            }
            
            # Add to results list
            results["samples"].append(sample_result)
            torch.cuda.empty_cache()
            
            # In screen env, update every 50 samples; standard env, every 20
            update_interval = 50 if is_screen_env else 20
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                # Calculate accuracy
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                
                # Update progress bar postfix
                pbar.set_postfix_str(
                    f"Accuracy:{current_accuracy:.1%}"
                )
                
                if is_screen_env:
                    # Extra progress print for screen (less frequent)
                    print(f"Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.1%}")
                    sys.stdout.flush()
            
            # Update progress bar
            pbar.update()
            
            # Less frequent forced flush
            if is_screen_env and sample_count % 25 == 0:
                sys.stdout.flush()
    
    # Calculate average times
    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / total_samples
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / total_samples
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / total_samples
    
    # Calculate accuracies
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    results["summary"]["real_accuracy"] = results["summary"]["real_correct"] / results["summary"]["real_total"] if results["summary"]["real_total"] > 0 else 0
    results["summary"]["fake_accuracy"] = results["summary"]["fake_correct"] / results["summary"]["fake_total"] if results["summary"]["fake_total"] > 0 else 0
    
    # Calculate precision, recall, F1 (fake as positive)
    tp = results["summary"]["fake_correct"]  # True positive: correctly detected fake
    fp = results["summary"]["real_total"] - results["summary"]["real_correct"]  # False positive: real detected as fake
    fn = results["summary"]["fake_total"] - results["summary"]["fake_correct"]  # False negative: fake detected as real
    tn = results["summary"]["real_correct"]  # True negative: correctly detected real
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results["summary"]["precision"] = precision
    results["summary"]["recall"] = recall
    results["summary"]["f1_score"] = f1_score
    
    # Save results as single JSON file
    json_output_file = f'{result_dir}/HAD_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save timing statistics
    timing_stats.export_to_json(timing_output_file)
    
    # Print result summary
    print("\n=== HAD Audio Authenticity Detection Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Real audio accuracy: {results['summary']['real_accuracy']:.2%} ({results['summary']['real_correct']}/{results['summary']['real_total']})")
    print(f"Fake audio accuracy: {results['summary']['fake_accuracy']:.2%} ({results['summary']['fake_correct']}/{results['summary']['fake_total']})")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 score: {f1_score:.2%}")
    print(f"Average inference time: {results['summary']['timing']['avg_total_time']:.4f} seconds")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} seconds")
    print(f"Results saved to: {json_output_file}")
    sys.stdout.flush()  # Ensure final result is flushed

if __name__ == "__main__":
    main()