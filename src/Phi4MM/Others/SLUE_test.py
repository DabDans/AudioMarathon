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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import random
random.seed(42)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

use_random = (prune_method == "random")
use_frame = (prune_method == "frame")
if use_random==False and use_frame==False:
    prune_method = "fast_v"
if prune_ratio == 0:
    method_is = "base"
else:
    method_is = prune_method


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

class SLUETimingStats:
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
        self.max_timing_samples = 100
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
        if self.total_samples < self.max_timing_samples:
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
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def load_slue_dataset(json_file, audio_base_dir):
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []
    
    print(f"LoadSLUE JSONfile: {json_file}")
    print(f" audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    
    if not isinstance(data, list):
        print(f"Error: JSON file format incorrect, expected list format")
        return []
    
    print(f"FromJSONloaded {len(data)}  tasks")
    
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
                print(f"Warning: Audio file does not exist: {full_audio_path}")
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
        print(f"Warning: Total {missing_files}  audio files do not exist")
    
    print(f"loaded {len(dataset)}  valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")
    return dataset

def prepare_audio_for_processor(audio_path, target_sr=16000):
    
    try:
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"soundfile loading failed: {e}")
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav = temp_file.name
                
                print(f"Convert using ffmpeg: {audio_path} -> {temp_wav}")
                subprocess.run([
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ar', str(target_sr), '-ac', '1',
                    temp_wav
                ], stderr=subprocess.DEVNULL)
                
                sample_rate, audio = wavfile.read(temp_wav)
                audio = audio.astype(np.float32)
                if audio.max() > 1.0:
                    audio = audio / 32768.0
                os.remove(temp_wav)
                print(f"ffmpegConversion successful: shape={audio.shape}, sample rate={sample_rate}Hz")
                
            except Exception as e:
                print(f"ffmpeg conversion failed: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sample_rate = target_sr
                print("Generate silent replacement audio")
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.flatten()
        print(f"Convert to mono: shape={audio.shape}")
        
        if sample_rate != target_sr and sample_rate > 0:
            audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
            sample_rate = target_sr
            print(f"Resample to{target_sr}Hz:  new length={len(audio)}")
            
        if len(audio) == 0:
            print("Warning: Audio is empty，Create3 seconds silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
        
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        return [(audio, sample_rate)]
        
    except Exception as e:
        print(f"Audio processed error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [(silence, target_sr)]

def create_slue_prompt(doc):
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    
    
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    return f"{user_prompt}<|audio_1|>{prompt_text}{prompt_suffix}{assistant_prompt}"

def extract_answer_choice(response):
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

def evaluate_slue_accuracy(predicted_choice, ground_truth_choice):
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
        print(f"evaluationSLUEaccuracyerror occurred: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def calculate_slue_metrics(predictions, ground_truths):
    try:
        valid_pairs = []
        for pred, gt in zip(predictions, ground_truths):
            if pred and gt and pred.strip() and gt.strip():
                valid_pairs.append((pred.strip().upper(), gt.strip().upper()))
        
        if len(valid_pairs) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'macro_f1': 0.0,
                'valid_samples': 0
            }
        
        valid_preds, valid_gts = zip(*valid_pairs)
        
        accuracy = accuracy_score(valid_gts, valid_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_gts, valid_preds, average='macro', zero_division=0
        )
        
        _, _, macro_f1, _ = precision_recall_fscore_support(
            valid_gts, valid_preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'macro_f1': float(macro_f1),
            'valid_samples': len(valid_pairs)
        }
    except Exception as e:
        print(f"CalculateSLUEmetricserror occurred: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'macro_f1': 0.0,
            'valid_samples': 0
        }

def main():


    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    slue_json_file = "/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/SLUE/merged_audio_data.json"
    audio_base_dir = "/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/SLUE"
    
    print(f"SLUE JSONfile: {slue_json_file}")
    print(f" audio base directory: {audio_base_dir}")
    
    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    
    result_dir = os.environ.get("RESULTS_DIR", './SLUE_Results')
    os.makedirs(result_dir, exist_ok=True)

    output_file = f'{result_dir}/slue_results_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    timing_output_file = f'{result_dir}/slue_timing_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    _AUDIO_SPECIAL_TOKEN_ID = 200011

    timing_stats = SLUETimingStats()

    print(f"\n=== SLUE NERtask evaluation configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Originalmethod parameter: {os.environ.get('PRUNE_METHOD', 'N/A')}")
    print(f"use_random: {use_random}, use_frame: {use_frame}")
    print(f"SLUE JSONfile: {slue_json_file}")
    print(f" audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f" samplelimit: {sample_limit}")
    print("=" * 40)

    print("LoadPhi-4-multimodal-instructmodel...")
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="sdpa",
        revision="33e62acdd07cd7d6635badd529aa0a3467bb9c6a",
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Using dataset: {len(samples)}  samples")
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Apply sample limit，processed {len(samples)}  samples")

    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    for sample in samples:
        task_name = sample.get("task_name", "unknown")
        dataset_name = sample.get("dataset_name", "unknown")
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1
    
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0
    
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)
    dataset_correct = defaultdict(int)
    dataset_total = defaultdict(int)

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    print(f"Start evaluation {len(samples)}  samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"modelLoadafter completion GPU memory - allocated: {allocated:.2f}GB,  reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="SLUEevaluation", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            audio_path = sample["audio"]["path"]
            audio = prepare_audio_for_processor(audio_path)
            
            ground_truth_choice = sample.get("answer_gt", "")
            task_name = sample.get("task_name", "unknown")
            dataset_name = sample.get("dataset_name", "unknown")
            
            prompt = create_slue_prompt(sample)

            inputs = processor(
                text=prompt,
                audios=audio,
                return_tensors="pt",
            ).to(device)
            inputs['input_mode'] = torch.tensor([2])

            audio_token_length = 0
            pruning_applied = False
            
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end - audio_token_start + 1
                print(audio_token_length)
                print(use_random)
                print(prune_ratio)
                model.config.image_layer_idx = None
                model.config.audio_layer_idx = prune_layer_idx
                model.config.audio_token_num = audio_token_length
                model.config.audio_token_start = audio_token_start
                model.config.audio_prune_ratio = prune_ratio
                model.config.random = use_random
                model.config.frame = use_frame
                
                if use_random or use_frame:
                    model.config.output_attentions = False
                else:
                    model.config.output_attentions = False
                
                pruning_applied = True
                
                if idx == 0:
                    print(f"Audio pruning configuration: start={audio_token_start}, length={audio_token_length}, ratio={prune_ratio}, method={prune_method}")
                    print(f"Model config check: audio_layer_idx={model.config.audio_layer_idx}, audio_prune_ratio={model.config.audio_prune_ratio}")
                    print(f"Flag settings: random={model.config.random}, frame={model.config.frame}")


            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
            prefill_end_event.record()
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0

            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
            
            decode_start_event.record()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    generation_config=generation_config,
                    do_sample=False,
                    return_dict_in_generate=True
                )
            decode_end_event.record()
            torch.cuda.synchronize()
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0

            tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
            output_tokens = len(tokens[0])
            resp = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            
            predicted_choice = extract_answer_choice(resp)

            metrics = evaluate_slue_accuracy(predicted_choice, ground_truth_choice)
            
            accuracy = metrics["accuracy"]
            is_correct = metrics["is_correct"]
            
            total_accuracy += accuracy
            processed_samples += 1

            task_type_total[task_name] += 1
            dataset_total[dataset_name] += 1
            if is_correct:
                task_type_correct[task_name] += 1
                dataset_correct[dataset_name] += 1

            current_avg_acc = total_accuracy / processed_samples
            
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_name[:10],
                    'Dataset': dataset_name[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })
                
                if is_screen_env:
                    print(f"   progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f" accuracy: {current_avg_acc:.3f}")
            else:
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_name[:10],
                    'Dataset': dataset_name[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })

            results.append({
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "filename": sample.get("filename", ""),
                "task_name": task_name,
                "dataset_name": dataset_name,
                "path": sample.get("path", ""),
                "duration": sample.get("duration", 0),
                "question": sample.get("question", ""),
                "choice_a": sample.get("choice_a", ""),
                "choice_b": sample.get("choice_b", ""),
                "choice_c": sample.get("choice_c", ""),
                "choice_d": sample.get("choice_d", ""),
                "ground_truth_choice": ground_truth_choice,
                "predicted_choice": predicted_choice,
                "accuracy": accuracy,
                "is_correct": is_correct,
                "response_text": resp,
                "entity_count": sample.get("entity_count", 0),
                "entity_types": sample.get("entity_types", []),
                "source_count": sample.get("source_count", 0),
                "metrics_detail": metrics
            })

            if idx > 0 and idx <= 100:
                timing_stats.add_record(
                    prefill_time, decode_time, 
                    output_tokens,
                    inputs["input_ids"].shape[1],
                    sample.get("duration", 0),
                    task_name
                )

            del inputs, outputs, out_ids
            
            del audio
            
            torch.cuda.empty_cache()
            
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [ sample {idx+1}]  GPU memory - allocated: {allocated:.2f}GB,  reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"Inference error: {e}")
            traceback.print_exc()
            resp = "ERROR"
            predicted_choice = "error"
            accuracy = 0.0
            is_correct = False
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            
            if 'audio' in locals():
                del audio
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'out_ids' in locals():
                del out_ids
            torch.cuda.empty_cache()
            continue

    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0

    all_predictions = [sample["predicted_choice"] for sample in results]
    all_ground_truths = [sample["ground_truth_choice"] for sample in results]
    
    overall_metrics = calculate_slue_metrics(all_predictions, all_ground_truths)

    task_type_accuracies = {}
    task_type_metrics = {}
    for task_name in task_type_stats.keys():
        if task_type_total[task_name] > 0:
            task_type_accuracies[task_name] = task_type_correct[task_name] / task_type_total[task_name]
            
            task_samples = [sample for sample in results if sample.get("task_name") == task_name]
            if task_samples:
                task_preds = [sample["predicted_choice"] for sample in task_samples]
                task_gts = [sample["ground_truth_choice"] for sample in task_samples]
                task_type_metrics[task_name] = calculate_slue_metrics(task_preds, task_gts)

    dataset_accuracies = {}
    dataset_metrics = {}
    for dataset_name in dataset_stats.keys():
        if dataset_total[dataset_name] > 0:
            dataset_accuracies[dataset_name] = dataset_correct[dataset_name] / dataset_total[dataset_name]
            
            dataset_samples = [sample for sample in results if sample.get("dataset_name") == dataset_name]
            if dataset_samples:
                dataset_preds = [sample["predicted_choice"] for sample in dataset_samples]
                dataset_gts = [sample["ground_truth_choice"] for sample in dataset_samples]
                dataset_metrics[dataset_name] = calculate_slue_metrics(dataset_preds, dataset_gts)

    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "f1_score": overall_metrics["f1_score"],
        "precision": overall_metrics["precision"], 
        "recall": overall_metrics["recall"],
        "macro_f1": overall_metrics["macro_f1"],
        "valid_samples": overall_metrics["valid_samples"],
        "task_type_stats": dict(task_type_stats),
        "dataset_stats": dict(dataset_stats),
        "task_type_accuracies": task_type_accuracies,
        "task_type_metrics": task_type_metrics,
        "dataset_accuracies": dataset_accuracies,
        "dataset_metrics": dataset_metrics,
        "task_type_correct": dict(task_type_correct),
        "task_type_total": dict(task_type_total),
        "dataset_correct": dict(dataset_correct),
        "dataset_total": dict(dataset_total),
        "config": {
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "slue_json_file": slue_json_file,
            "audio_base_dir": audio_base_dir
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== SLUE NERtask evaluation result summary ===")
    print(f"Total samples: {len(results)}")
    print(f"processed samples: {processed_samples}")
    print(f"Valid samples: {overall_metrics['valid_samples']}")
    print(f"Overall accuracy: {final_accuracy:.3f}")
    print(f"F1 score: {overall_metrics['f1_score']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"Macro average F1: {overall_metrics['macro_f1']:.4f}")
    print(f"Task type count: {len(task_type_stats)}")
    print(f"datasetcount: {len(dataset_stats)}")
    
    print(f"\nDetailed metrics for each task type:")
    for task_name, acc in task_type_accuracies.items():
        correct_num = task_type_correct[task_name]
        total_num = task_type_total[task_name]
        print(f"  {task_name}:")
        print(f"     accuracy: {acc:.3f} ({correct_num}/{total_num})")
        if task_name in task_type_metrics:
            metrics = task_type_metrics[task_name]
            print(f"    F1 score: {metrics['f1_score']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
    
    print(f"\nEachdatasetdetailedmetrics:")
    for dataset_name, acc in dataset_accuracies.items():
        correct_num = dataset_correct[dataset_name]
        total_num = dataset_total[dataset_name]
        print(f"  {dataset_name}:")
        print(f"     accuracy: {acc:.3f} ({correct_num}/{total_num})")
        if dataset_name in dataset_metrics:
            metrics = dataset_metrics[dataset_name]
            print(f"    F1 score: {metrics['f1_score']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\ntimestatistics (based onfirst100 samples，Skip1):")
    print(f"statistics samples: {overall_summary.get('total_samples', 0)}")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/ seconds")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()
