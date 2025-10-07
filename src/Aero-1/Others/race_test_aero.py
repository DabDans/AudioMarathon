import os
import json
import glob
import time
import torch
import soundfile as sf
import numpy as np
import librosa
import warnings
import traceback
import gc
import sys
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
from transformers import logging

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Disable transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("Successfully imported required modules")

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
        """Get summary statistics"""
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
        """Export statistics to a JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

class CudaEventTimingStats:
    """CUDA Event batch timing statistics"""
    
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
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    
    # Use librosa for resampling
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio using librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Load audio using librosa (official recommended way)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate does not match
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split if audio length exceeds 30s
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            print(f"Audio length {len(audio)} exceeds 30s limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # If audio length is within 30s, return as a single chunk list
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        # Return silence chunk list
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def clean_text_response(response):
    """Clean model response for RACE task, keeping only the first character as option label"""
    if not response:
        return ""
    resp = response.strip().upper()
    # Only take the first non-empty character
    for ch in resp:
        if ch in ["A","B","C","D"]:
            return ch
    return resp.split()[0] if resp.split() else ""

def cuda_timing_inference(model, processor, inputs, max_new_tokens=3):
    """
    Inference function with precise GPU timing using CUDA Event API
    """
    
    # Ensure GPU idle for precise timing
    torch.cuda.synchronize()
    
    # Create CUDA events
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        # === Phase 1: Prefill timing ===
        event_start.record()
        
        # Prefill computation
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, 
                           output_hidden_states=False, return_dict=True)
        
        event_prefill_end.record()
        
        # === Phase 2: Full Generation timing ===
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
        
        # === Compute precise time difference ===
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
    # Get environment variable configuration
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    print(f"Using GPU ID: {gpu_id}")

    # Audio pruning configuration
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
    prune_method = os.environ.get("PRUNE_METHOD", "base")

    # Flag setting based on method name
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
        print("Debug mode enabled - verbose output will be shown")

    # Data path configuration
    data_path_root = os.environ.get("RACE_DATA_PATH", '/data/to/your/race_audio/path')
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/Race_Results/path')
    os.makedirs(result_dir, exist_ok=True)

    # Output file paths
    output_file = f'{result_dir}/RACE_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    timing_output_file = f'{result_dir}/RACE_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    cuda_event_output_file = f'{result_dir}/RACE_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    print(f"CUDA Event statistics will be saved to: {cuda_event_output_file}")

    # Create timing stats tracker
    timing_stats = RaceTimingStats()
    cuda_event_stats = CudaEventTimingStats()

    print(f"\n=== RACE Evaluation Config (Aero-1) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer idx: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Data path: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

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

    # Set audio prune config
    if prune_ratio > 0:
        print(f"Note: Using official Aero-1 model, audio pruning feature may not be available")
        print(f"Prune config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")

    # Load benchmark data
    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: benchmark file not found: {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Apply sample limit
    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f"Sample count limited to: {sample_limit}")

    # Print initial memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loaded - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    results = []

    # Stats variables
    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0

    print(f"Starting evaluation of {len(benchmark)} samples...")
    
    # Detect if running in screen or non-interactive environment
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
    
    # Tqdm config
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

    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc="RACE Evaluation (Aero-1)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        
        # Predefine variables to avoid UnboundLocalError
        prefill_time = 0
        decode_time = 0
        output_tokens = 0
        audio_token_length = 0
        pred = ""
        correct = 0
        
        try:
            audio_rel = sample["audio_path"]
            audio_full = os.path.join(data_path_root, audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"Warning: audio file not found: {audio_full}")
                continue
            
            # Determine high or middle subset
            if "high" in audio_rel:
                total_high += 1
            elif "middle" in audio_rel:
                total_middle += 1

            # Official message format - supports multiple audio chunks
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Prepare audio input - now returns audio chunks list
            audio_chunks, sample_rate = prepare_audio_for_processor(audio_full)
            
            # Add audio content for each chunk into message
            for chunk in audio_chunks:
                messages[0]["content"].append({
                    "type": "audio",
                    "audio": "placeholder",  # This will be replaced by actual audio
                })
            
            # Format options
            formatted_options = ""
            for i, opt in enumerate(sample['options']):
                letter = chr(65 + i)  # A, B, C, D...
                formatted_options += f"{letter}. {opt}\n"
            
            # Add RACE task text content
            instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
            format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
            
            messages[0]["content"].append({
                "type": "text",
                "text": f"{instruction}\n\nQuestion: {sample['question']}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}"
            })
            
            # Use chat template to process messages
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Use processor to handle text and audio chunks
            inputs = processor(
                text=prompt,
                audios=audio_chunks,  # Pass audio chunks list
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
            
            # Verbose info in debug mode
            if debug_mode:
                print(f"Processing audio: {os.path.basename(audio_full)}")
                print(f"Number of audio chunks: {len(audio_chunks)}")
                print(f"Question: {sample['question']}")
                print(f"Options: {sample['options']}")
                print(f"Correct answer: {sample['answer']}")
                print(f"Estimated audio token length: {audio_token_length}")
                sys.stdout.flush()
            
            # Inference with CUDA Event precise timing
            result = cuda_timing_inference(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=3
            )
            
            # Get results
            output = result['response_text']
            prefill_time = result['prefill_time']
            decode_time = result['decode_time']
            total_time = result['total_time']
            output_tokens = result['output_tokens']
            
            # Clean response and extract answer
            pred = clean_text_response(output)
            
            # Result stats
            correct = int(pred == sample["answer"])
            if correct:
                correct_count += 1
                if "high" in audio_rel:
                    correct_high += 1
                elif "middle" in audio_rel:
                    correct_middle += 1
            
            current_acc = (correct_count / (idx + 1)) * 100
            
            # Verbose output in debug mode
            if debug_mode:
                print(f"Model output: '{output}'")
                print(f"Extracted answer: '{pred}'")
                print(f"Correct answer: '{sample['answer']}'")
                print(f"Answer correct: {correct}")
                print(f"Inference time: Total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                print(f"Output tokens: {output_tokens}")
                print("=" * 50)
                sys.stdout.flush()

            # Record result
            results.append({
                "idx": idx,
                "article_id": sample.get("article_id", ""),
                "question_idx": sample.get("question_idx", idx),
                "pred": pred, 
                "gt": sample["answer"],
                "correct": correct,
                "audio_path": audio_rel,
                "subset": "high" if "high" in audio_rel else "middle",
                "audio_chunks": len(audio_chunks),
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time
            })

            # Add timing statistics
            timing_stats.add_record(
                prefill_time=prefill_time,
                decode_time=decode_time,
                output_tokens=output_tokens,
                input_tokens=inputs["input_ids"].shape[1],
                audio_duration=sum(len(chunk) for chunk in audio_chunks) / sample_rate
            )
            
            # Collect CUDA Event specialized timing stats
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
            
            # Set error state variable values
            pred = "ERROR"
            correct = 0
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            
            # Save error result
            results.append({
                "idx": idx,
                "article_id": sample.get("article_id", ""),
                "question_idx": sample.get("question_idx", idx),
                "pred": pred, 
                "gt": sample["answer"],
                "correct": correct,
                "audio_path": sample.get("audio_path", ""),
                "subset": "high" if "high" in sample.get("audio_path", "") else "middle",
                "audio_chunks": 1,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            })
            
            current_acc = (correct_count / (idx + 1)) * 100
            continue
        
        # Update progress bar
        update_interval = 50 if is_screen_env else 20
        sample_count = idx + 1
        
        if sample_count % update_interval == 0 or sample_count == len(benchmark):
            audio_duration = sum(len(chunk) for chunk in audio_chunks) / sample_rate if 'audio_chunks' in locals() else 0
            progress_bar.set_postfix_str(
                f"Accuracy:{current_acc:.1f}%, Answer:{pred}/{sample['answer']}, Duration:{audio_duration:.1f}s"
            )
            
            if is_screen_env:
                print(f"Progress: {sample_count}/{len(benchmark)} ({sample_count/len(benchmark)*100:.1f}%), "
                      f"Accuracy: {current_acc:.1f}%")
                sys.stdout.flush()
        
        progress_bar.update()
        
        if is_screen_env and sample_count % 25 == 0:
            sys.stdout.flush()

    # Calculate final accuracy
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    # Create result summary
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
        "config": {
            "model_name": "Aero-1-Audio-1.5B",
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "data_path": data_path_root
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

    # Save timing statistics
    timing_stats.export_to_json(timing_output_file)
    
    # Save CUDA Event statistics to separate file
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)

    # Print result summary
    print(f"\n=== RACE Evaluation Result Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Prune config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={method_is}")
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    if total_high > 0:
        print(f"HIGH set accuracy: {correct_high/total_high*100:.2f}% ({correct_high}/{total_high})")
    if total_middle > 0:
        print(f"MIDDLE set accuracy: {correct_middle/total_middle*100:.2f}% ({correct_middle}/{total_middle})")
    
    timing_summary = timing_stats.get_summary()
    print(f"Avg inference time: {timing_summary.get('avg_total_time', 0):.4f}s")
    print(f"Avg Prefill time: {timing_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Avg Decode time: {timing_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Avg throughput: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/sec")
    
    # Show CUDA Event detailed statistics
    cuda_event_stats.print_statistics()
    
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    print(f"CUDA Event statistics saved to: {cuda_event_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()