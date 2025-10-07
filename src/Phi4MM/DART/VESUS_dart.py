#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VESUS Emotion Recognition Model Evaluation Script
Used to evaluate model performance on the VESUS emotion recognition task, supporting audio pruning methods.
"""

import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
random.seed(42)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"


_AUDIO_SPECIAL_TOKEN_ID = 200011

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attn_implementation')
    parser.add_argument('--sparse', type=str_to_bool, default=True, help='Enable sparse mode')
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
    """Configure DART sparse attention mechanism"""
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


gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")
random.seed(42)


sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")


data_path = os.environ.get("VESUS_DATA_PATH",
    "/data/to/your/vesus/path")
emotion_json_file = os.path.join(data_path, "audio_emotion_dataset.json")
result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/results/path')
os.makedirs(result_dir, exist_ok=True)




_AUDIO_SPECIAL_TOKEN_ID = 200011

def calculate_emotion_metrics(predictions, ground_truths, emotion_labels):
    """Calculate emotion classification metrics: Accuracy, Precision, Recall, and F1 score"""

    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths)
                   if p in emotion_labels and t in emotion_labels]

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

    label_map = {label: idx for idx, label in enumerate(sorted(emotion_labels))}
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

class VESUSTimingStats:
    """Track inference timing statistics for VESUS emotion recognition task, supporting CUDA Event measurement"""
    def __init__(self):
        self.timing_records = []
        self.emotion_stats = defaultdict(list)
        self.person_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.use_cuda_events = torch.cuda.is_available()

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
        """Export statistics data to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_file


def load_vesus_dataset(root_dir):
    """Load VESUS speech emotion analysis dataset from concatenated_audio directory"""

    meta_file = os.path.join(root_dir, "audio_emotion_dataset.json")

    all_samples = []

    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} sample metadata from {meta_file}")

        for item in metadata:

            rel_path = item["path"]
            wav_path = os.path.join(root_dir, rel_path)

            if not os.path.exists(wav_path):
                print(f"Warning: File does not exist {wav_path}")
                continue

            emotion_label = item["emotion_label"]
            answer_gt = item["answer_gt"].strip()
            person_id = item["person_id"]

            # Skip person2 and person10's "Happy" emotion data
            if emotion_label == "happy":
                if person_id == "2" or person_id == "10":
                    print(f"Skipping person{person_id} Happy emotion data: {rel_path}")
                    continue

            all_samples.append({
                "emotion": emotion_label,
                "audio_path": wav_path,
                "question": item["question"],
                "choices": [item["choice_a"], item["choice_b"], item["choice_c"], item["choice_d"]],
                "answer_gt": answer_gt,
                "task": "Speech_Emotion_Recognition",
                "person_id": person_id
            })
    else:
        print(f"Warning: Metadata file does not exist {meta_file}")
        return []

    print(f"Total loaded {len(all_samples)} valid audio samples")

    emotion_counts = {}
    for sample in all_samples:
        group = sample["emotion"]
        emotion_counts[group] = emotion_counts.get(group, 0) + 1

    print("Emotion distribution:")
    for group, count in emotion_counts.items():
        print(f"  {group}: {count} samples")

    random.shuffle(all_samples)

    return all_samples


def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Process audio file"""
    try:

        full_audio_path = os.path.join(data_path, audio_path)

        if not os.path.exists(full_audio_path):
            print(f"Audio file does not exist: {full_audio_path}")
            return None

        audio_data, sample_rate = sf.read(full_audio_path)

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        if sample_rate != target_sr:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            except ImportError:
                print("Warning: librosa not installed, skipping resampling")

                if sample_rate != target_sr:
                    try:
                        import scipy.signal
                        audio_data = scipy.signal.resample(audio_data,
                                                         int(len(audio_data) * target_sr / sample_rate))
                    except ImportError:
                        print("Warning: scipy also not installed, keeping original sample rate")

        return [(audio_data, target_sr)]

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None
def extract_emotion_answer(text, choices):
    """Extract emotion answer from model output text"""
    text_lower = text.lower().strip()

    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return "A"
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return "B"
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return "C"
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return "D"

    option_patterns = {
        'A': ["option a", "choice a", "a)", "(a)"],
        'B': ["option b", "choice b", "b)", "(b)"],
        'C': ["option c", "choice c", "c)", "(c)"],
        'D': ["option d", "choice d", "d)", "(d)"]
    }

    for option, patterns in option_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return option

    emotion_keywords = {
        'angry': ['anger', 'frustrated', 'mad', 'furious'],
        'happy': ['joy', 'cheerful', 'pleased', 'delighted'],
        'sad': ['sadness', 'melancholy', 'depressed', 'sorrow'],
        'fearful': ['fear', 'anxiety', 'scared', 'afraid'],
        'monotone': ['flat', 'emotionless', 'neutral', 'bland']
    }

    for choice_key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
        if choice_key in choices:
            choice_text = choices[choice_key].lower()
            for emotion, keywords in emotion_keywords.items():
                if emotion in choice_text or any(keyword in choice_text for keyword in keywords):
                    if any(keyword in text_lower for keyword in keywords) or emotion in text_lower:
                        return choice_key[-1].upper()

    return ""

def create_emotion_prompt(sample):
    """Create prompt for emotion recognition task"""
    question = sample.get("question", "What emotion is expressed in this audio segment?")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")

    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    prompt = f"""{user_prompt}<|audio_1|>{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D).{prompt_suffix}{assistant_prompt}"""

    return prompt

def main():
    args = parse_arguments()
    print(f"\n=== VESUS Emotion Recognition Evaluation Config ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Data path: {data_path}")
    print(f"JSON file: {emotion_json_file}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    output_file = f'{result_dir}/VESUS_results_gpu{gpu_id}_prune_{args.reduction_ratio}.json'
    timing_output_file = f'{result_dir}/VESUS_timing_stats_gpu{gpu_id}_prune_{args.reduction_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    print("Loading Phi-4-multimodal-instruct model...")
    model_path = args.model_path

    model_revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"

    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=model_revision,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=model_revision,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )

    generation_config = GenerationConfig.from_pretrained(
        model_path,
        revision=model_revision
    )

    configure_DART(model, args)
    print("Model loaded successfully")

    samples = load_vesus_dataset(data_path)

    if not samples:
        print("Error: No data samples found")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, processing {len(samples)} samples")

    timing_stats = VESUSTimingStats()

    results = []
    total_correct = 0
    emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    person_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    print(f"Start evaluating {len(samples)} samples...")
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="VESUS Evaluation")

    for idx, sample in progress_bar:
        try:

            audio_path = sample.get("path", "")
            audio = prepare_audio_for_processor(audio_path)

            if audio is None:
                continue

            emotion_label = sample.get("emotion_label", "unknown")
            person_id = sample.get("person_id", "unknown")
            answer_gt = sample.get("answer_gt", "").upper()

            prompt = create_emotion_prompt(sample)

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

                if args.sparse:
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
            out_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                generation_config=generation_config,
                do_sample=False,
                use_cache=True
            )
            decode_end_event.record()

            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0

            resp = processor.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            predicted_answer = extract_emotion_answer(resp, sample)
            is_correct = (predicted_answer == answer_gt)

            if is_correct:
                total_correct += 1

            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1

            if is_correct:
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1

            current_accuracy = total_correct / (idx + 1)
            progress_bar.set_postfix({
                'Acc': f'{current_accuracy:.3f}',
                'Emotion': emotion_label[:8],
                'Person': person_id
            })

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
                "timing": {
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
            })

            timing_stats.add_record(
                prefill_time, decode_time,
                out_ids.shape[1] - inputs["input_ids"].shape[1],
                inputs["input_ids"].shape[1],
                emotion_label, person_id
            )

            del inputs, outputs, out_ids
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    all_predictions = [result["predicted_answer"] for result in results]
    all_ground_truths = [result["answer_gt"] for result in results]
    all_emotion_labels = list(set(all_ground_truths))

    emotion_metrics = calculate_emotion_metrics(all_predictions, all_ground_truths, all_emotion_labels)

    emotion_accuracies = {}
    for emotion, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracies[emotion] = stats["correct"] / stats["total"]

    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]

    summary = {
        "total_samples": total_samples,
        "correct_samples": total_correct,
        "overall_accuracy": overall_accuracy,
        "metrics": emotion_metrics,
        "emotion_stats": dict(emotion_stats),
        "emotion_accuracies": emotion_accuracies,
        "person_stats": dict(person_stats),
        "person_accuracies": person_accuracies,
        "config": {
            "gpu_id": gpu_id,
            "prune_layer_idx": args.pruned_layer,
            "prune_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file
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

    print(f"\n=== VESUS Emotion Recognition Evaluation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"F1 Score: {emotion_metrics['f1_score']:.4f}")
    print(f"Precision: {emotion_metrics['precision']:.4f}")
    print(f"Recall: {emotion_metrics['recall']:.4f}")
    print(f"Valid samples: {emotion_metrics['valid_samples']}/{emotion_metrics['total_samples']}")
    print(f"Accuracy by emotion:")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")

    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== Timing Statistics (CUDA Events) ===")
    print(f"Avg inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Avg Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Avg Decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Avg throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()