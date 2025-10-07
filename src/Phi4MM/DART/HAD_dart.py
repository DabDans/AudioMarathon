#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HAD Audio Authenticity Detection DART Version Evaluation Script
Integrated DART sparse attention mechanism, supports audio pruning
"""

import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd

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

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

data_path_root = '/data/to/your/concatenated_audio/path'
result_dir = '/data/to/your/HAD_Results/path'
os.makedirs(result_dir, exist_ok=True)

class HADTimingStats:
    """Track inference timing statistics for HAD task using precise CUDA Event measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, label=None):
        """Add a timing record"""
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "label": label
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get overall statistics summary"""
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        df = pd.DataFrame(self.timing_records)
        summary = {
            "total_samples": len(df),
            "avg_total_time": df["total_time"].mean(),
            "avg_prefill_time": df["prefill_time"].mean(),
            "avg_decode_time": df["decode_time"].mean(),
            "avg_decode_tokens_per_sec": df["decode_tokens_per_sec"].mean(),
            "prefill_percentage": (df["prefill_time"].sum() / df["total_time"].sum()) * 100,
            "decode_percentage": (df["decode_time"].sum() / df["total_time"].sum()) * 100
        }
        
        return summary
    
    def export_to_json(self, output_file):
        """Export timing statistics to JSON file"""
        summary = self.get_summary()
        data = {
            "summary": summary,
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Properly process audio file as per official example"""
    try:
        audio_data, orig_sr = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        if orig_sr != target_sr:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        audio_data = audio_data.astype(np.float32)
        return [(audio_data, target_sr)]
    except Exception as e:
        print(f"Audio processing error {audio_path}: {e}")
        return None

def load_had_dataset(root_dir):
    """Load HAD dataset and balance real/fake sample counts"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    all_samples = []
    if os.path.exists(real_dir):
        for audio_file in os.listdir(real_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(real_dir, audio_file)
                all_samples.append({
                    "audio_path": audio_path,
                    "label": "real",
                    "id": f"real_{audio_file}"
                })
    if os.path.exists(fake_dir):
        for audio_file in os.listdir(fake_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(fake_dir, audio_file)
                all_samples.append({
                    "audio_path": audio_path,
                    "label": "fake",
                    "id": f"fake_{audio_file}"
                })
    print(f"Total loaded {len(all_samples)} audio samples")
    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"Original sample count: real={len(real_samples)}, fake={len(fake_samples)}")
    min_samples_per_category = min(len(real_samples), len(fake_samples))
    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)
    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)
    balanced_samples = real_samples + fake_samples
    random.shuffle(balanced_samples)
    print(f"Balanced sample count: real={len(real_samples)}, fake={len(fake_samples)}, total={len(balanced_samples)}")
    return balanced_samples

def extract_authenticity_answer(text, choice_a="real", choice_b="fake"):
    """Extract authenticity answer from model output text"""
    text_lower = text.lower().strip()
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b
    if choice_a_lower == "real" and choice_b_lower == "fake":
        real_keywords = ["real", "authentic", "genuine", "original", "natural"]
        fake_keywords = ["fake", "synthetic", "artificial", "generated", "deepfake"]
        real_count = sum(1 for keyword in real_keywords if keyword in text_lower)
        fake_count = sum(1 for keyword in fake_keywords if keyword in text_lower)
        if real_count > fake_count:
            return "real"
        elif fake_count > real_count:
            return "fake"
    return ""

def calculate_had_metrics(predictions, ground_truths):
    """Calculate F1 score and other metrics for HAD task"""
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['real', 'fake'] and t in ['real', 'fake']]
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
    label_map = {'real': 0, 'fake': 1}
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
        'total_samples': len(predictions)
    }

def create_had_prompt(sample):
    """Create prompt for HAD task"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    instruction = "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'."
    format_text = "A) real\nB) fake\n\nPlease select A or B based on the audio content."
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    return prompt

def main():
    args = parse_arguments()
    print(f"\n=== HAD DART Evaluation Config ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART Sparse Mode: {args.sparse}")
    print(f"Pruned Layers: {args.pruned_layer}")
    print(f"Retained Ratio: {args.reduction_ratio}")
    print(f"Data Directory: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample Limit: {sample_limit}")
    print("=" * 40)

    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = f'{result_dir}/HAD_results_dart{sparse_suffix}.json'
    timing_output_file = f'{result_dir}/HAD_timing_stats_dart{sparse_suffix}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = HADTimingStats()

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
    model.eval()

    generation_config = GenerationConfig.from_pretrained(
        model_path, 
        revision=model_revision
    )

    configure_DART(model, args)
    print("Model loaded successfully")

    samples = load_had_dataset(data_path_root)

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")

    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)

    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"Category statistics: real_samples={real_count}, fake_samples={fake_count}")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
            "metrics": {},
            "timing": {}
        }
    }

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    all_predictions = []
    all_ground_truths = []

    with tqdm(total=len(samples), desc="Processing HAD audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                audio = prepare_audio_for_processor(sample["audio_path"])
                if audio is None:
                    continue
                prompt = create_had_prompt(sample)
                inputs = processor(
                    text=prompt,
                    audios=audio,
                    return_tensors="pt"
                ).to("cuda")
                inputs['input_mode'] = torch.tensor([2])
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start_index = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    rev_ids = token_ids[::-1]
                    audio_token_end_index = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end_index - audio_token_start_index + 1
                    if args.sparse:
                        model.config.DART_config['audio_token_start_index'] = audio_token_start_index
                        model.config.DART_config['audio_token_length'] = audio_token_length
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        prefill_start_event = torch.cuda.Event(enable_timing=True)
                        prefill_end_event = torch.cuda.Event(enable_timing=True)
                        decode_start_event = torch.cuda.Event(enable_timing=True)
                        decode_end_event = torch.cuda.Event(enable_timing=True)
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
                        decode_start_event.record()
                        with torch.no_grad():
                            generate_output = model.generate(
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
                    if hasattr(generate_output, 'sequences'):
                        generate_ids = generate_output.sequences
                    else:
                        generate_ids = generate_output
                    input_tokens = inputs['input_ids'].shape[1]
                    output_tokens = generate_ids.shape[1] - input_tokens
                    new_tokens = generate_ids[:, inputs['input_ids'].shape[1]:]
                    response = processor.batch_decode(
                        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    predicted_label = extract_authenticity_answer(response)
                    true_label = sample["label"]
                    is_correct = predicted_label == true_label
                    if predicted_label:
                        all_predictions.append(predicted_label)
                        all_ground_truths.append(true_label)
                    sample_result = {
                        "id": sample["id"],
                        "audio_path": sample["audio_path"],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "response": response,
                        "is_correct": is_correct,
                        "timing": {
                            "prefill_time": prefill_time,
                            "decode_time": decode_time,
                            "total_time": prefill_time + decode_time,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "tokens_per_sec": output_tokens/decode_time if decode_time > 0 else 0
                        }
                    }
                    results["samples"].append(sample_result)
                    timing_stats.add_record(
                        prefill_time,
                        decode_time,
                        output_tokens,
                        input_tokens,
                        label=true_label
                    )
                    results["summary"]["total_samples"] += 1
                    if is_correct:
                        results["summary"]["correct_samples"] += 1
                    if true_label == "real":
                        results["summary"]["real_total"] += 1
                        if is_correct:
                            results["summary"]["real_correct"] += 1
                    else:
                        results["summary"]["fake_total"] += 1
                        if is_correct:
                            results["summary"]["fake_correct"] += 1
                    pbar.set_postfix({
                        'Acc': f"{results['summary']['correct_samples']/results['summary']['total_samples']:.3f}",
                        'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                    })
                pbar.update(1)
                del inputs, outputs, generate_output, new_tokens, audio
                torch.cuda.empty_cache()
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                pbar.update(1)
                for var_name in ['inputs', 'outputs', 'generate_output', 'generate_ids', 'new_tokens', 'audio']:
                    try:
                        if var_name in locals():
                            del locals()[var_name]
                    except:
                        pass
                torch.cuda.empty_cache()
                continue

    if len(all_predictions) > 0:
        metrics = calculate_had_metrics(all_predictions, all_ground_truths)
        results["summary"]["metrics"] = metrics
        print(f"\n=== HAD Evaluation Results ===")
        print(f"Total Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")

    timing_summary = timing_stats.get_summary()
    results["summary"]["timing"] = timing_summary

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

    timing_stats.export_to_json(timing_output_file)
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()