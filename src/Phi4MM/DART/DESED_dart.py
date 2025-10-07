import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
import argparse
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attn_implementation')
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--image_token_start_index', type=int, default=None, help='image_token_start_index')
    parser.add_argument('--image_token_length', type=int, default=None, help='image_token_length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    return parser.parse_args()

def configure_DART(model, args):
    if args.sparse:
        DART_config = {
            "K": args.pruned_layer,
            "image_token_start_index": args.image_token_start_index,
            "image_token_length": args.image_token_length,
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            "reduction_ratio": args.reduction_ratio,
            "pivot_image_token": args.pivot_audio_token,
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            "text_length": 1,
        }
        model.config.DART_config = DART_config
    else:
        model.config.DART_config = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

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
    """Track inference timing statistics for DESED sound event detection task"""
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
        """Get overall summary statistics"""
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
        """Export timing statistics to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_file

def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    Load data from new DESED task JSON file

    Args:
        json_file: path to DESED task JSON file
        audio_base_dir: base directory for audio files

    Returns:
        dataset: list containing task data
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
        print(f"Error: Incorrect JSON file format, expected dictionary with 'tasks' field")
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
            print(f"Failed to read audio file info {full_audio_path}: {e}")
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
        print(f"Warning: {missing_files} audio files do not exist")

    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    return dataset

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Properly process audio file as in HAD_test.py"""

    try:
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"soundfile loading failed: {e}")

            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav = temp_file.name

                print(f"Converting with ffmpeg: {audio_path} -> {temp_wav}")
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
                print(f"ffmpeg conversion succeeded: shape={audio.shape}, samplerate={sample_rate}Hz")

            except Exception as e:
                print(f"ffmpeg conversion failed: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sample_rate = target_sr
                print("Generated silent substitute audio")

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.flatten()
        print(f"Converted to mono: shape={audio.shape}")

        if sample_rate != target_sr and sample_rate > 0:
            audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
            sample_rate = target_sr
            print(f"Resampled to {target_sr}Hz: new length={len(audio)}")

        if len(audio) == 0:
            print("Warning: Audio is empty, creating 3 seconds of silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)

        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return [(audio, sample_rate)]

    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [(silence, target_sr)]

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

    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    return f"{user_prompt}<|audio_1|>{prompt_text}{prompt_suffix}{assistant_prompt}"

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

def calculate_desed_metrics(predictions, ground_truths):
    """Calculate DESED sound event detection metrics: Accuracy, Precision, Recall, F1 score"""

    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths)
                   if p in ['A', 'B', 'C', 'D'] and t in ['A', 'B', 'C', 'D']]

    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'valid_samples': 0,
            'total_samples': len(predictions)
        }

    valid_predictions, valid_ground_truths = zip(*valid_pairs)

    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions),
        'label_mapping': label_map
    }

def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate accuracy for sound event detection task"""
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

    args = parse_arguments()

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    print(f"Using GPU ID: {gpu_id}")

    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0.0))
    prune_method = os.environ.get("PRUNE_METHOD", "fast_v")

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
        print(f"Sample limit set to: {sample_limit}")

    qa_json_file = "/data/to/your/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json"
    audio_base_dir = "/data/to/your/DESED_dataset/concatenated_audio"

    print(f"QA JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")

    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)

    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/DESED_Results')
    os.makedirs(result_dir, exist_ok=True)

    output_file = f'{result_dir}/desed_sound_event_detection_results.json'
    timing_output_file = f'{result_dir}/desed_sound_event_detection_timing_stats_{method_is}_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    _AUDIO_SPECIAL_TOKEN_ID = 200011

    timing_stats = DESEDTimingStats()

    print(f"\n=== DESED Sound Event Detection Evaluation Config ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Task JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    print("Loading Phi-4-multimodal-instruct model...")
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_path)

    configure_DART(model, args)
    device="cuda"

    print(f"Using dataset: {len(samples)} samples")

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, processing {len(samples)} samples")

    task_type_stats = defaultdict(int)
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_stats[task_type] += 1

    print(f"Task type statistics: {dict(task_type_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0

    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)

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

    all_predictions = []
    all_ground_truths = []

    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="DESED Sound Event Detection Evaluation", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:

            audio_path = sample["audio"]["path"]
            audio = prepare_audio_for_processor(audio_path)

            ground_truth_choice = sample.get("answer_gt", "")
            task_type = sample.get("task_type", "unknown")

            prompt = create_qa_prompt(sample)

            inputs = processor(
                text=prompt,
                audios=audio,
                return_tensors="pt",
            ).to(device)
            inputs['input_mode'] = torch.tensor([2])

            audio_token_length = 0
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end - audio_token_start + 1

                if hasattr(model.config, 'DART_config') and model.config.DART_config is not None:
                    model.config.DART_config['audio_token_start_index'] = audio_token_start
                    model.config.DART_config['audio_token_length'] = audio_token_length

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

            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)

            decode_start_event.record()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    generation_config=generation_config,
                    do_sample=False,
                    return_dict_in_generate=True,
                    return_legacy_cache=True
                )
            decode_end_event.record()

            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0
            full_time = prefill_time + decode_time

            tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
            output_tokens = len(tokens[0])
            resp = processor.batch_decode(tokens, skip_special_tokens=True)[0]

            predicted_choice = extract_answer_choice(resp)

            metrics = evaluate_qa_accuracy(predicted_choice, ground_truth_choice)

            accuracy = metrics["accuracy"]
            is_correct = metrics["is_correct"]

            if predicted_choice in ['A', 'B', 'C', 'D'] and ground_truth_choice in ['A', 'B', 'C', 'D']:
                all_predictions.append(predicted_choice)
                all_ground_truths.append(ground_truth_choice)

            total_accuracy += accuracy
            processed_samples += 1

            task_type_total[task_type] += 1
            if is_correct:
                task_type_correct[task_type] += 1

            current_avg_acc = total_accuracy / processed_samples

            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1

            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_type[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })

                if is_screen_env:

                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_avg_acc:.3f}")
            else:
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_type[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })

            results.append({
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
                "ground_truth_choice": ground_truth_choice,
                "predicted_choice": predicted_choice,
                "accuracy": accuracy,
                "is_correct": is_correct,
                "response_text": resp,
                "original_events": sample.get("original_events", []),
                "metrics_detail": metrics
            })

            if idx > 0:
                timing_stats.add_record(
                    prefill_time, decode_time,
                    output_tokens,
                    inputs["input_ids"].shape[1],
                    sample.get("duration", 0),
                    task_type
                )

            del inputs, outputs, out_ids

            del audio

            torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

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
            audio_token_length = 0

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

    overall_metrics = calculate_desed_metrics(all_predictions, all_ground_truths)

    task_type_accuracies = {}
    for task_type in task_type_stats.keys():
        if task_type_total[task_type] > 0:
            task_type_accuracies[task_type] = task_type_correct[task_type] / task_type_total[task_type]

    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "task_type_stats": dict(task_type_stats),
        "task_type_accuracies": task_type_accuracies,
        "task_type_correct": dict(task_type_correct),
        "task_type_total": dict(task_type_total),
        "overall_metrics": overall_metrics,
        "config": {
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "task_json_file": qa_json_file,
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

    print(f"\n=== DESED Sound Event Detection Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Overall Accuracy: {final_accuracy:.4f}")
    print(f"F1 Score: {overall_metrics['f1_score']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"Valid samples: {overall_metrics['valid_samples']}/{overall_metrics['total_samples']}")
    print(f"Number of task types: {len(task_type_stats)}")

    print(f"\nAccuracy by task type:")
    for task_type, acc in task_type_accuracies.items():
        correct_num = task_type_correct[task_type]
        total_num = task_type_total[task_type]
        print(f"  {task_type}: {acc:.3f} ({correct_num}/{total_num})")

    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\nAverage inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds (excluding first sample)")
    print(f"Average prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds (excluding first sample)")
    print(f"Average decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds (excluding first sample)")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()