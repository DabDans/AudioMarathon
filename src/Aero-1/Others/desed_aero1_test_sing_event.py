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
import librosa

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class DESEDTimingStats:
    """Track inference timing statistics for DESED sound event detection"""
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
        """Add a timing record"""
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        if audio_duration:
            self.total_audio_duration += audio_duration
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "task_type": task_type
        }
        
        self.timing_records.append(record)
        
        if task_type:
            self.task_type_stats[task_type].append(record)
    
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
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_audio_duration": self.total_audio_duration,
            "avg_audio_duration": self.total_audio_duration / self.total_samples if self.total_samples > 0 else 0
        }
        
        task_summaries = {}
        for task_type, records in self.task_type_stats.items():
            if len(records) > 0:
                task_summaries[task_type] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "task_summaries": task_summaries
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

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    audio_splits = []
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio with librosa and split for Aero-1 model"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    Load data from new DESED task JSON file
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
        print(f"Error: JSON file format incorrect, expected dict with 'tasks' field")
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
        
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"Failed to read audio file info {full_audio_path}: {e}")
            continue
        
        choices = task.get("choices", {})
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_type": task.get("task_type", "unknown"),
            "question": task.get("question", ""),
            "choice_a": choices.get("A", ""),
            "choice_b": choices.get("B", ""),
            "choice_c": choices.get("C", ""),
            "choice_d": choices.get("D", ""),
            "answer_gt": task.get("answer_gt", ""),
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
        task_type_stats[item["task_type"]] += 1
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    return dataset

def create_aero_qa_prompt(doc):
    """Generate QA-style prompt for Aero-1 model"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    
    prompt_text = f"""Listen to the audio carefully and answer the following question.

{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please select the correct answer and respond with only the letter (A, B, C, or D)."""
    
    return prompt_text

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

def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate QA task accuracy"""
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
        print(f"Error evaluating QA accuracy: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    print(f"Using GPU ID: {gpu_id}")
    
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0.0))
    prune_method = os.environ.get("PRUNE_METHOD", "fast_v")
    
    use_random = (prune_method == "random")
    use_frame = (prune_method == "frame")
    if not use_random and not use_frame:
        prune_method = "fast_v"
    
    method_is = "base" if prune_ratio == 0 else prune_method
    
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    if debug_mode:
        print("Debug mode enabled - detailed output will be shown")
    
    qa_json_file = "/data/to/your/qa_json_file/path/desed_sound_event_detection_task.json"
    audio_base_dir = "/data/to/your/audio_base_dir/path"
    
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/results_dir/path')
    os.makedirs(result_dir, exist_ok=True)
    
    output_file = f'{result_dir}/DESED_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    timing_output_file = f'{result_dir}/DESED_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    
    _AUDIO_SPECIAL_TOKEN_ID = 151667  # Aero-1 model audio special token ID
    
    timing_stats = DESEDTimingStats()
    
    print(f"\n=== DESED Sound Event Detection Evaluation Config (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Task JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)
    
    print("Loading Aero-1 model...")
    sys.stdout.flush()
    
    model_name = "lmms-lab/Aero-1-Audio-1.5B"
    processor = AutoProcessor.from_pretrained(model_name, revision="main", trust_remote_code=True)
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
    
    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, will process {len(samples)} samples")
    
    task_type_stats = defaultdict(int)
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_stats[task_type] += 1
    print(f"Task type statistics: {dict(task_type_stats)}")
    
    results = {
        "model_name": "Aero-1-Audio-1.5B",
        "dataset": "DESED",
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
    print(f"GPU memory after model load - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    with tqdm(total=len(samples), desc="Processing DESED audio samples (Aero-1)", **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            audio_path = sample["audio"]["path"]
            task_type = sample.get("task_type", "unknown")
            ground_truth = sample.get("answer_gt", "").upper().strip()
            
            output = ""
            predicted_choice = ""
            is_correct = False
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            
            try:
                audio_chunks, sample_rate = prepare_audio_for_processor(audio_path)
                
                prompt_text = create_aero_qa_prompt(sample)
                
                messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",
                    })
                
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
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    if hasattr(model.config, 'audio_layer_idx'):
                        model.config.image_layer_idx = None
                        model.config.audio_layer_idx = prune_layer_idx
                        model.config.audio_token_num = audio_token_length
                        model.config.audio_token_start = audio_token_start
                        model.config.audio_prune_ratio = prune_ratio
                        model.config.random = use_random
                        model.config.frame = use_frame
                
                if debug_mode:
                    print(f"Processing audio: {os.path.basename(audio_path)}")
                    print(f"Task type: {task_type}")
                    print(f"Number of audio chunks: {len(audio_chunks)}")
                    print(f"Total audio length: {sum(len(chunk) for chunk in audio_chunks)}, Sample rate: {sample_rate}")
                    print(f"Question: {sample.get('question', '')}")
                    print(f"Ground truth: {ground_truth}")
                    print(f"Input IDs shape: {inputs.input_ids.shape}")
                    print(f"Estimated audio token length: {audio_token_length}")
                    sys.stdout.flush()
                
                start_time = time.time()
                
                with torch.no_grad():
                    prefill_start = time.time()
                    model_outputs = model(
                        **inputs,
                        use_cache=True,
                        return_dict=True
                    )
                    prefill_time = time.time() - prefill_start
                    
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
                
                input_length = inputs['input_ids'].shape[1]
                new_tokens = generate_ids[:, input_length:]
                output_tokens = new_tokens.shape[1]
                
                output = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                
                predicted_choice = extract_answer_choice(output)
                
                metrics = evaluate_qa_accuracy(predicted_choice, ground_truth)
                is_correct = metrics["is_correct"]
                
                if debug_mode:
                    print(f"Model output: '{output}'")
                    print(f"Inference time: Total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                    print(f"Output tokens: {output_tokens}")
                    print(f"Extracted answer: '{predicted_choice}'")
                    print(f"Ground truth: '{ground_truth}'")
                    print(f"Answer correct: {is_correct}")
                    print("=" * 50)
                    sys.stdout.flush()
                
                results["summary"]["total_samples"] += 1
                results["summary"]["task_type_total"][task_type] += 1
                if is_correct:
                    results["summary"]["correct_samples"] += 1
                    results["summary"]["task_type_correct"][task_type] += 1
                
                results["summary"]["timing"]["total_prefill_time"] += prefill_time
                results["summary"]["timing"]["total_decode_time"] += decode_time
                results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                
                if idx > 0:  # Skip timing stats for the first sample
                    timing_stats.add_record(
                        prefill_time, decode_time, output_tokens,
                        input_length, sample.get("duration", 0), task_type
                    )
                
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_choice = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                audio_token_length = 0
            
            sample_result = {
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "filename": sample.get("filename", ""),
                "task_type": task_type,
                "path": sample.get("path", ""),
                "duration": sample.get("duration", 0),
                "question": sample.get("question", ""),
                "choice_a": sample.get("choice_a", ""),
                "choice_b": sample.get("choice_b", ""),
                "choice_c": sample.get("choice_c", ""),
                "choice_d": sample.get("choice_d", ""),
                "ground_truth_choice": ground_truth,
                "predicted_choice": predicted_choice,
                "is_correct": is_correct,
                "response_text": output,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                "original_events": sample.get("original_events", []),
                "metrics_detail": metrics
            }
            
            results["samples"].append(sample_result)
            
            torch.cuda.empty_cache()
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"[Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            update_interval = 50 if is_screen_env else 20
            sample_count = idx + 1
            if sample_count % update_interval == 0 or sample_count == len(samples):
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                pbar.set_postfix_str(
                    f"Accuracy:{current_accuracy:.1%}, Task:{task_type[:10]}, Predicted:{predicted_choice}, GT:{ground_truth}"
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
    for task_type in results["summary"]["task_type_total"]:
        total_count = results["summary"]["task_type_total"][task_type]
        correct_count = results["summary"]["task_type_correct"][task_type]
        task_type_accuracies[task_type] = correct_count / total_count if total_count > 0 else 0.0
    
    results["summary"]["task_type_accuracies"] = task_type_accuracies
    results["summary"]["task_type_correct"] = dict(results["summary"]["task_type_correct"])
    results["summary"]["task_type_total"] = dict(results["summary"]["task_type_total"])
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n=== DESED Sound Event Detection Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Dataset: DESED")
    print(f"Pruning config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    
    print(f"\nTask type accuracies:")
    for task_type, accuracy in task_type_accuracies.items():
        correct_count = results["summary"]["task_type_correct"][task_type]
        total_count = results["summary"]["task_type_total"][task_type]
        print(f"  {task_type}: {accuracy:.2%} ({correct_count}/{total_count})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== Timing statistics ===")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/sec")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()