#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vox Age Classification DART Evaluation Script
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import soundfile as sf
import numpy as np


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

def calculate_metrics(predictions, ground_truths):
    """Calculate classification metrics: Accuracy, Precision, Recall, and F1 score"""

    valid_age_groups = ['Young Adult (18-30)', 'Early Career (31-40)', 'Mid Career (41-50)', 'Senior (51-70)', 'Elderly (71+)']
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in valid_age_groups and t in valid_age_groups]
    
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
    

    label_map = {
        'Young Adult (18-30)': 0,
        'Early Career (31-40)': 1,
        'Mid Career (41-50)': 2,
        'Senior (51-70)': 3,
        'Elderly (71+)': 4
    }
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions)
    }

class VoxAgeTimingStats:
    """Global timing statistics class, not grouped by folder"""
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_tokens = 0
        self.timing_records = []
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens=0, age_group=None):
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        

        self.timing_records.append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        })
    
    def get_summary(self):
        """Get timing summary"""
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "avg_tokens_per_sec": 0
            }
        
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_decode_time": self.total_decode_time / self.samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        }
    
    def export_to_json(self, output_file):
        """Export timing data to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def print_summary(self):
        """Print timing summary"""
        summary = self.get_summary()
        print(f"\n=== Timing Summary ===")
        print(f"Valid samples: {summary['samples']}")
        print(f"Average Prefill time: {summary['avg_prefill_time']:.4f} seconds")
        print(f"Average Decode time: {summary['avg_decode_time']:.4f} seconds")
        print(f"Average Total time: {summary['avg_total_time']:.4f} seconds")
        print(f"Average tokens/seconds: {summary['avg_tokens_per_sec']:.2f}")


gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")


sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")


data_path_root = '/data/to/your/voxceleb/concatenated_audio'
result_dir = '/data/to/your/vox_results'
os.makedirs(result_dir, exist_ok=True)


def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Process audio file following official example"""
    
    try:

        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"Failed to load soundfile: {e}")
            

            try:
                import subprocess
                import tempfile
                from scipy.io import wavfile
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav = temp_file.name
                
                print(f"Using ffmpeg to convert: {audio_path} -> {temp_wav}")
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
                print(f"ffmpeg conversion success: shape={audio.shape}, sample_rate={sample_rate}Hz")
                
            except Exception as e:
                print(f"ffmpeg conversion failed: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sample_rate = target_sr
                print("Generated silent substitute audio")
        

        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
            print(f"Converted to mono: shape={audio.shape}")
        

        if sample_rate != target_sr and sample_rate > 0:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
            sample_rate = target_sr
            print(f"Resampled to {target_sr}Hz: new length={len(audio)}")
            

        if len(audio) == 0:
            print("Warning: Empty audio, creating 3 seconds silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
        

        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        return [(audio, sample_rate)]
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        import traceback
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [(silence, target_sr)]

def read_text_file(txt_path):
    """Read text file content"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        return ""

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    """Load dataset from concatenated_audio directory, based on age_classification_task_meta.json"""

    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    

    for item in metadata:

        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        

        if not os.path.exists(wav_path):
            print(f"Warning: File does not exist {wav_path}")
            continue
        

        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()
        speaker_age = item.get("speaker_age", 0)
        

        all_samples.append({
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "choice_e": item["choice_e"],
            "answer_gt": age_group,
            "task": "Speaker_Age_Classification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    

    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Applying sample limit: randomly selecting {sample_limit} out of {len(all_samples)}")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Sample count after limit: {len(all_samples)}")
    

    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    

    random.shuffle(all_samples)
    
    return all_samples

def extract_age_answer(text, choices):
    """Extract age group answer from model output text"""
    text_lower = text.lower().strip()
    

    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return choices["choice_d"]
    if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)'):
        return choices["choice_e"]
        

    for option, choice_text in choices.items():
        option_letter = option[-1].lower()
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"{option_letter})" in text_lower:
            return choice_text
    

    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    

    if len(choice_matches) == 1:
        return choice_matches[0]
    

    return ""

def main():

    args = parse_arguments()
    
    print(f"\n=== Vox Age DART Classification Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)


    sparse_suffix = "_sparse" if args.sparse else "_base"
    output_file = f'{result_dir}/VoxCeleb_age_results_dart{sparse_suffix}.json'
    timing_output_file = f'{result_dir}/VoxCeleb_age_timing_stats_dart{sparse_suffix}.json'
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
    

    timing_stats = VoxAgeTimingStats()
    

    generation_config = GenerationConfig.from_pretrained(model_path)
    

    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    

    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    

    print(f"Start processing {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing VoxCeleb Age Classification Samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        

        for i, sample in enumerate(samples):
            wav_path = sample['wav_path']
            speaker_id = sample["speaker_id"]
            ground_truth = sample["age_group"].strip()
            

            choices = {
                "choice_a": sample["choice_a"],
                "choice_b": sample["choice_b"],
                "choice_c": sample["choice_c"],
                "choice_d": sample["choice_d"],
                "choice_e": sample["choice_e"]
            }
            

            instruction = "Listen to this audio and identify the speaker's age group. Choose the most appropriate option: (a) Young Adult (18-30), (b) Early Career (31-40), (c) Mid Career (41-50), (d) Senior (51-70), (e) Elderly (71+). Answer with only the letter (a, b, c, d, or e)."
            full_prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
            
            try:

                audio = prepare_audio_for_processor(wav_path)
              

                inputs = processor(
                    text=full_prompt,
                    audios=audio,
                    return_tensors="pt"
                ).to("cuda")
                inputs['input_mode'] = torch.tensor([2])
                

                audio_token_length = 0
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
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
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        generation_config=generation_config,
                        return_dict_in_generate=True
                    )
                decode_end_event.record()
                

                torch.cuda.synchronize()
                prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
                decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0
                

                tokens = generate_ids.sequences[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                output = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                

                predicted_age_group = extract_age_answer(output, choices)
                

                all_predictions.append(predicted_age_group)
                all_ground_truths.append(ground_truth)
                

                is_correct = predicted_age_group == ground_truth
                

                if i > 0:
                    timing_stats.add_record(prefill_time, decode_time, output_tokens, inputs['input_ids'].shape[1], age_group=ground_truth)
                
            except Exception as e:
                print(f"Inference error: {e}")
                import traceback
                traceback.print_exc()
                output = "ERROR"
                predicted_age_group = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                

                all_predictions.append(predicted_age_group)
                all_ground_truths.append(ground_truth)
            

            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_age_group,
                "is_correct": is_correct,
                "audio_tokens": audio_token_length if 'audio_token_length' in locals() else 0,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            }
            
            all_sample_results.append(sample_result)
            torch.cuda.empty_cache()
            

            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p != "error" and t != "error") / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p != "error" and t != "error"))
            
            pbar.set_postfix({
                'Samples': f'{i+1}/{len(samples)}',
                'Accuracy': f'{current_accuracy:.3f}',
                'Speaker': speaker_id[:8] + '...' if len(speaker_id) > 8 else speaker_id
            })
            
            pbar.update()
    

    metrics_result = calculate_metrics(all_predictions, all_ground_truths)
    final_stats = timing_stats.get_summary()
    

    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    

    age_group_results = {}
    for result in all_sample_results:
        group = result['ground_truth']
        if group not in age_group_results:
            age_group_results[group] = {'total': 0, 'correct': 0}
        age_group_results[group]['total'] += 1
        if result['is_correct']:
            age_group_results[group]['correct'] += 1
    

    results = {
        "samples": all_sample_results,
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": correct_samples / total_samples if total_samples > 0 else 0,
            "age_group_results": {
                group: {
                    "total": stats['total'],
                    "correct": stats['correct'],
                    "accuracy": stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                }
                for group, stats in age_group_results.items()
            },
            "metrics": metrics_result,
            "timing": final_stats,
            "config": {
                "gpu_id": gpu_id,
                "sparse": args.sparse,
                "pruned_layer": args.pruned_layer,
                "reduction_ratio": args.reduction_ratio,
                "sample_limit": sample_limit
            }
        }
    }


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")


    timing_stats.export_to_json(timing_output_file)
    print(f"Timing statistics saved to: {timing_output_file}")
    

    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Total Accuracy: {results['summary']['accuracy']:.2%}")
    

    for group, stats in results['summary']['age_group_results'].items():
        print(f"{group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"F1 Score: {metrics_result['f1_score']:.4f}")
    print(f"Precision: {metrics_result['precision']:.4f}")  
    print(f"Recall: {metrics_result['recall']:.4f}")
    print(f"Average inference time: {final_stats['avg_total_time']:.4f} seconds (excluding first sample)")
    print(f"Average Prefill time: {final_stats['avg_prefill_time']:.4f} seconds (excluding first sample)")
    print(f"Average Decode time: {final_stats['avg_decode_time']:.4f} seconds (excluding first sample)")

if __name__ == "__main__":
    main()