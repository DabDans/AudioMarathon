#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TAU Acoustic Scene Classification DART Evaluation Script
Integrates DART sparse attention mechanism and supports audio pruning
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

def calculate_tau_metrics(all_sample_results):
    """Calculate TAU acoustic scene classification evaluation metrics"""
    predictions = [result.get('prediction', '') for result in all_sample_results]
    labels = [result.get('ground_truth', '') for result in all_sample_results]
    

    accuracy = accuracy_score(labels, predictions)
    

    try:
        weighted_precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    except:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    

    try:
        macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    except:
        macro_precision = macro_recall = macro_f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"


random.seed(42)


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

def calculate_acoustic_metrics(predictions, ground_truths, scene_labels):
    """Calculate acoustic scene classification metrics: Accuracy, Precision, Recall, and F1 score"""

    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in scene_labels and t in scene_labels]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'valid_samples': 0,
            'total_samples': len(predictions),
            'label_mapping': {}
        }
    
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    

    label_map = {label: idx for idx, label in enumerate(sorted(scene_labels))}
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

class TAUTimingStats:
    """Track TAU task inference timing statistics using CUDA Event for precise measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, scene_label=None):
        """Add a timing record"""
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "scene_label": scene_label
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get overall timing statistics summary"""
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
    
    def print_summary(self):
        """Print statistics summary"""
        summary = self.get_summary()
        if "error" not in summary:
            print(f"Average total time: {summary['avg_total_time']:.3f} seconds")
            print(f"Average prefill time: {summary['avg_prefill_time']:.3f} seconds")
            print(f"Average decode time: {summary['avg_decode_time']:.3f} seconds")
            print(f"Average tokens/seconds: {summary['avg_decode_tokens_per_sec']:.2f}")


gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")


sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")


data_path_root = '/data/to/your/tau/path/'
audio_dir = os.path.join(data_path_root, 'to/your/concatenated_resampled/path/')
result_dir = '/data/to/your/tau_results/path/'
os.makedirs(result_dir, exist_ok=True)


def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Process audio file correctly according to official example"""
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

def load_tau_acoustic_scene_dataset(root_dir):
    """Load acoustic scene classification task from TAU dataset"""

    meta_file = os.path.join(root_dir, "to/your/acoustic_scene_task_meta/path/acoustic_scene_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    

    scene_counts = {}
    

    for item in metadata:

        audio_rel_path = item["path"]
        audio_path = os.path.join(root_dir, audio_rel_path)
        
        if os.path.exists(audio_path):
            scene_label = item["scene_label"]
            scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
            

            choices = [
                item["choice_a"],
                item["choice_b"], 
                item["choice_c"],
                item["choice_d"]
            ]
            
            sample = {
                "audio_path": audio_path,
                "scene_label": scene_label,
                "choices": choices,
                "correct_answer": item["answer_gt"],
                "question": item["question"],
                "id": item.get("uniq_id", f"tau_{len(all_samples)}")
            }
            all_samples.append(sample)
        else:
            print(f"Audio file does not exist: {audio_path}")
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    

    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count}")
    

    if sample_limit > 0 and sample_limit < len(all_samples):
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Sample number limited to: {len(all_samples)}")
        

    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """Extract acoustic scene answer option (A/B/C/D) from model output text"""
    text_lower = text.lower().strip()
    

    options = ['a', 'b', 'c', 'd']
    

    if text_lower in options:
        return text_lower.upper()
    

    for opt in options:
        if text_lower.startswith(f"{opt}.") or text_lower.startswith(f"{opt})") or text_lower.startswith(f"{opt}:"):
            return opt.upper()
    

    for opt in options:
        patterns = [f"option {opt}", f"choice {opt}", f"answer {opt}", f"({opt})"]
        if any(pattern in text_lower for pattern in patterns):
            return opt.upper()
    

    if choices:
        for i, choice_text in enumerate(choices):
            if choice_text.lower() in text_lower:
                return chr(65 + i)
    

    return ""

def create_tau_prompt(sample):
    """Create prompt for TAU acoustic scene classification task"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option."
    

    formatted_options = ""
    for i, choice in enumerate(sample["choices"]):
        formatted_options += f"{chr(65+i)}) {choice}\n"
    
    format_text = "Respond with only the letter of your answer (A, B, C, or D)"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def main():

    args = parse_arguments()
    
    print(f"\n=== TAU DART Acoustic Scene Classification Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Data directory: {audio_dir}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)


    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = f'{result_dir}/TAU_results_dart{sparse_suffix}.json'
    timing_output_file = f'{result_dir}/TAU_timing_stats_dart{sparse_suffix}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")


    print("Loading Phi-4-multimodal-instruct model...")
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    

    configure_DART(model, args)
    print("Model loaded successfully")
    

    timing_stats = TAUTimingStats()
    

    generation_config = GenerationConfig.from_pretrained(model_path)
    

    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    

    print(f"Total processing {len(samples)} samples")
    

    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    

    scene_stats = {scene: {"total": 0, "correct": 0} for scene in scene_counts}
    

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    

    print(f"Start processing {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing TAU acoustic scene samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:

                audio = prepare_audio_for_processor(sample["audio_path"])
                if audio is None:
                    continue
                

                prompt = create_tau_prompt(sample)
                

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
                        generation_start_event = torch.cuda.Event(enable_timing=True)
                        generation_end_event = torch.cuda.Event(enable_timing=True)
                        

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
                        

                        generation_start_event.record()
                        with torch.no_grad():
                            generate_output = model.generate(
                                **inputs,
                                max_new_tokens=10,
                                generation_config=generation_config,
                                do_sample=False,
                                use_cache=True,
                                return_dict_in_generate=True,
                                return_legacy_cache=True
                            )
                        generation_end_event.record()
                        
                        torch.cuda.synchronize()
                        
                        prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
                        full_generation_time = generation_start_event.elapsed_time(generation_end_event) / 1000.0
                        decode_time = full_generation_time - prefill_time

                    

                    if hasattr(generate_output, 'sequences'):
                        generate_ids = generate_output.sequences
                    else:
                        generate_ids = generate_output
                    
                    input_tokens = inputs['input_ids'].shape[1]
                    output_tokens = generate_ids.shape[1] - input_tokens
                    

                    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                    response = processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    

                    predicted_option = extract_acoustic_scene_answer(response, sample["choices"])
                    true_option = sample["correct_answer"]
                    

                    if predicted_option and ord(predicted_option) - ord('A') < len(sample["choices"]):
                        predicted_scene = sample["choices"][ord(predicted_option) - ord('A')]
                    else:
                        predicted_scene = "unknown"
                    
                    true_scene = sample["scene_label"]
                    is_correct = predicted_option == true_option
                    

                    all_predictions.append(predicted_scene)
                    all_ground_truths.append(true_scene)
                    

                    scene_stats[true_scene]["total"] += 1
                    if is_correct:
                        scene_stats[true_scene]["correct"] += 1
                    

                    sample_result = {
                        "id": sample["id"],
                        "audio_path": sample["audio_path"],
                        "true_scene": true_scene,
                        "predicted_scene": predicted_scene,
                        "true_option": true_option,
                        "predicted_option": predicted_option,
                        "choices": sample["choices"],
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
                    
                    all_sample_results.append(sample_result)
                    

                    timing_stats.add_record(
                        prefill_time,
                        decode_time,
                        output_tokens,
                        input_tokens,
                        scene_label=true_scene
                    )
                    

                    current_acc = sum(1 for r in all_sample_results if r['is_correct']) / len(all_sample_results)
                    pbar.set_postfix({
                        'Acc': f"{current_acc:.3f}",
                        'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                    })
                
                pbar.update(1)


                del inputs, outputs, generate_output, audio
                torch.cuda.empty_cache()
                

                if (idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                pbar.update(1)
                

                for var_name in ['inputs', 'outputs', 'generate_output', 'audio']:
                    try:
                        if var_name in locals():
                            del locals()[var_name]
                    except:
                        pass
                torch.cuda.empty_cache()
                continue
    


    tau_metrics = calculate_tau_metrics(all_sample_results)
    final_stats = timing_stats.get_summary()
    

    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    

    scene_accuracies = {}
    for scene, stats in scene_stats.items():
        if stats["total"] > 0:
            scene_accuracies[scene] = stats["correct"] / stats["total"]
        else:
            scene_accuracies[scene] = 0.0


    final_results = {
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "overall_accuracy": tau_metrics['accuracy'],
            "scene_accuracies": scene_accuracies,
            "scene_stats": scene_stats,
            "tau_metrics": tau_metrics,
            "config": {
                "gpu_id": gpu_id,
                "sparse": args.sparse,
                "pruned_layer": args.pruned_layer,
                "reduction_ratio": args.reduction_ratio,
                "sample_limit": sample_limit
            },
            "timing": final_stats
        },
        "samples": all_sample_results
    }


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")


    timing_stats.export_to_json(timing_output_file)
    print(f"Timing statistics saved to: {timing_output_file}")

    print(f"\n=== TAU DART Evaluation Results ===")
    print(f"Overall Accuracy: {tau_metrics['accuracy']:.4f}")
    print(f"Weighted F1 Score: {tau_metrics['weighted_f1']:.4f}")
    print(f"Macro F1 Score: {tau_metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {tau_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {tau_metrics['weighted_recall']:.4f}")
    print(f"Macro Precision: {tau_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {tau_metrics['macro_recall']:.4f}")
    print(f"Valid samples: {total_samples}")

if __name__ == "__main__":
    main()