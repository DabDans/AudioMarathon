import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
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
import librosa
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import random


random.seed(42)


sys.path.append("/data/to/your/Modeling/path/")
from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)


from qwen_omni_utils import process_mm_info


_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      




os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"


logging.set_verbosity_error()
warnings.filterwarnings("ignore")


gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")


prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 1))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")


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

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

def calculate_acoustic_metrics(predictions, ground_truths, scene_labels):
    """Calculate acoustic scene classification metrics: accuracy, precision, recall, and F1 score"""
    
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in scene_labels and t in scene_labels]
    
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
    
    
    label_map = {label: idx for idx, label in enumerate(sorted(scene_labels))}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions),
        'label_mapping': label_map
    }

class GlobalTimingStats:
    """Global timing statistics class, statistics for the first 100 samples (excluding the first one)"""
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_time = 0.0
        self.total_input_tokens = 0
        self.total_audio_tokens = 0
        self.timing_records = []
        self.max_samples = 100  
    
    def add_record(self, prefill_time, total_time, output_tokens, input_tokens, audio_tokens, sample_index):
        
        if sample_index == 0:
            return
        
        if self.samples >= self.max_samples:
            return
            
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_time += total_time
        self.total_input_tokens += input_tokens
        self.total_audio_tokens += audio_tokens
        
        
        self.timing_records.append({
            "sample_index": sample_index,
            "prefill_time": prefill_time,
            "total_time": total_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "audio_tokens": audio_tokens
        })
    
    def get_summary(self):
        """Get statistics summary"""
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_total_time": 0,
                "avg_input_tokens": 0,
                "avg_audio_tokens": 0
            }
        
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_total_time": self.total_time / self.samples,
            "avg_input_tokens": self.total_input_tokens / self.samples,
            "avg_audio_tokens": self.total_audio_tokens / self.samples
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def print_summary(self):
        """Print statistics summary"""
        summary = self.get_summary()
        print(f"\n=== Time Statistics Summary (First 100 samples, excluding the first) ===")
        print(f"Statistical sample count: {summary['samples']}")
        print(f"Average total time: {summary['avg_total_time']:.4f} seconds")
        print(f"Average Prefill time: {summary['avg_prefill_time']:.4f} seconds")
        print(f"Average input token count: {summary['avg_input_tokens']:.1f}")
        print(f"Average audio token count: {summary['avg_audio_tokens']:.1f}")

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files according to Qwen2.5-Omni requirements"""
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Librosa loading successful: shape={audio.shape}, sample rate={sr}Hz")
        except Exception as e:
            print(f"Librosa loading failed: {e}")
            
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"Soundfile processing successful: shape={audio.shape}, sample rate={sr}Hz")
                
            except Exception as e:
                print(f"Soundfile loading also failed: {e}")
                
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generating silent replacement audio")
        
        
        if len(audio) == 0:
            print("Warning: Audio is empty, creating 3-second silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            
        
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def load_tau_acoustic_scene_dataset(root_dir):
    """Load acoustic scene classification tasks from TAU dataset"""
    
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    
    scene_counts = {}
    
    
    for item in metadata:
        
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)
        
        
        if not os.path.exists(wav_path):
            print(f"Warning: File does not exist {wav_path}")
            continue
        
        
        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"] 
        
        
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        
        
        all_samples.append({
            "scene_label": scene_label,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    
    
    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count} samples ({count/len(all_samples)*100:.1f}%)")
    
    
    if sample_limit > 0 and sample_limit < len(all_samples):
        print(f"Due to sample limit setting, randomly selecting {sample_limit} samples for evaluation")
        all_samples = random.sample(all_samples, sample_limit)
        
    
    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """Extract acoustic scene answer options (A/B/C/D) from model output text"""
    text_lower = text.lower().strip()
    
    
    options = ['a', 'b', 'c', 'd']
    
    
    if text_lower in options:
        return text_lower.upper()
    
    
    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    
    
    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    
    
    if choices:
        best_match = None
        max_overlap = 0
        
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()
            
            if choice_lower in text_lower:
                return chr(65 + i)  
            
            
            keywords = choice_lower.split(' - ')[0].split()  
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        
        if best_match and max_overlap > 1:  
            return best_match
    
    
    return ""

def group_samples_by_scene(samples):
    """将样本按场景分组"""
    grouped = {}
    for sample in samples:
        scene = sample["scene_label"]
        if scene not in grouped:
            grouped[scene] = []
        grouped[scene].append(sample)
    return grouped

def main():
    
    data_path_root = '/path/to/your/subsetTAU'  
    audio_dir = os.path.join(data_path_root, 'concatenated_resampled')  
    result_dir = './TAU_Results'
    os.makedirs(result_dir, exist_ok=True)

    
    output_file = f'{result_dir}/TAU_results_qwen25_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
    timing_output_file = f'{result_dir}/TAU_timing_stats_qwen25_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")

    print(f"\n=== TAU声学场景分类配置 (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"剪枝层索引: {prune_layer_idx}")
    print(f"剪枝比例: {prune_ratio}")
    print(f"剪枝方法: {method_is}")
    print(f"数据目录: {audio_dir}")
    print(f"结果目录: {result_dir}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 50)

    
    print("加载Qwen2.5-Omni模型...")
    model_path = "/path/to/your/model"  
    device_map = {"": 0}  
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    model.disable_talker()
    
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        
        if not hasattr(model.thinker.model.config, 'image_layer_idx'):
            model.thinker.model.config.image_layer_idx = False
        if not hasattr(model.thinker.model.config, 'audio_layer_idx'):
            model.thinker.model.config.audio_layer_idx = None
        if not hasattr(model.thinker.model.config, 'audio_token_num'):
            model.thinker.model.config.audio_token_num = None
        if not hasattr(model.thinker.model.config, 'audio_token_start'):
            model.thinker.model.config.audio_token_start = None
        if not hasattr(model.thinker.model.config, 'audio_prune_ratio'):
            model.thinker.model.config.audio_prune_ratio = 0
        if not hasattr(model.thinker.model.config, 'random'):
            model.thinker.model.config.random = False
        if not hasattr(model.thinker.model.config, 'frame'):
            model.thinker.model.config.frame = False
        
    
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    
    timing_stats = GlobalTimingStats()
    
    
    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    
    
    print(f"Samples: {len(samples)} ")
    
    
    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    
    
    scene_stats = {scene: {"total": 0, "correct": 0} for scene in scene_counts}
    
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    
    tqdm_kwargs = {
        'ascii': True,        
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }


    print(f"Samples: {len(samples)}")


    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    with tqdm(total=len(samples), desc="Processing TAU acoustic scene samples (Qwen2.5)", position=0, leave=True, **tqdm_kwargs) as pbar:

        
        for i, sample in enumerate(samples):
            try:
                wav_path = sample['wav_path']
                scene_label = sample["scene_label"]
                ground_truth = sample["answer_gt"].upper()
                
                
                instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.\n"
                instruction += f"A: {sample['choice_a']}\nB: {sample['choice_b']}\nC: {sample['choice_c']}\nD: {sample['choice_d']}\n"
                instruction += "Respond with only the letter of your answer (A, B, C, or D)."
                
                
                audio_path_for_inference = wav_path  
                
                
                task_instruction = "You are a helpful assistant that analyzes urban soundscape audio to identify acoustic scenes. Please listen to the audio carefully and classify the scene type."
                full_user_prompt = f"{task_instruction}\n\n{instruction}"
                
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                        ]
                    },
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path_for_inference},
                        {"type": "text", "text": full_user_prompt},
                    ]}
                ]
                
                
                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                
                if isinstance(text, list):
                    text = text[0] if len(text) > 0 else ""
                
                
                inputs = processor(
                    text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=True
                )
                inputs = inputs.to(model.device).to(model.dtype)
                
                
                audio_token_length = 0
                audio_token_start = 0
                input_token_length = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
                
                
                audio_detected = False
                
                
                if hasattr(inputs, 'input_ids'):
                    token_ids = inputs.input_ids[0].tolist()
                    
                    
                    bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                    eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                    
                    if bos_positions and eos_positions:
                        
                        audio_token_start = bos_positions[0]
                        audio_token_end = eos_positions[0]
                        audio_token_length = audio_token_end - audio_token_start + 1
                        
                        audio_detected = True
                        
                        
                        model.thinker.model.config.image_layer_idx = False  
                        model.thinker.model.config.audio_layer_idx = prune_layer_idx
                        model.thinker.model.config.audio_token_num = audio_token_length
                        model.thinker.model.config.audio_token_start = audio_token_start
                        model.thinker.model.config.audio_prune_ratio = prune_ratio
                        model.thinker.model.config.random = use_random
                        model.thinker.model.config.frame = use_frame 
                        print(f"Detected audio tokens from position {audio_token_start} to {audio_token_end} (length {audio_token_length})")
                        model.thinker.model.config.random = use_random
                        model.thinker.model.config.frame = use_frame

                    
                    elif not audio_detected and _AUDIO_TOKEN_ID in token_ids:
                        audio_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_TOKEN_ID]
                        if audio_positions:
                            audio_token_start = audio_positions[0]
                            audio_token_length = len(audio_positions)
                            
                            audio_detected = True
                            
                            
                            if hasattr(model, 'config'):
                                model.config.image_layer_idx = None
                                model.config.audio_layer_idx = prune_layer_idx
                                model.config.audio_token_num = audio_token_length
                                model.config.audio_token_start = audio_token_start
                                model.config.audio_prune_ratio = prune_ratio
                                model.config.random = use_random
                                model.config.frame = use_frame
                
                if not audio_detected:
                    if hasattr(model, 'config'):
                        model.config.audio_layer_idx = None
                        model.config.audio_prune_ratio = 0
                
                
                prefill_start_event = torch.cuda.Event(enable_timing=True)
                prefill_end_event = torch.cuda.Event(enable_timing=True)
                
                prefill_start_event.record()
                
                
                audio_tokens = 0
                if hasattr(processor.tokenizer, 'audio_bos_token_id') and hasattr(processor.tokenizer, 'audio_eos_token_id'):
                    input_ids = inputs['input_ids'][0]
                    audio_bos_positions = (input_ids == processor.tokenizer.audio_bos_token_id).nonzero(as_tuple=True)[0]
                    audio_eos_positions = (input_ids == processor.tokenizer.audio_eos_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(audio_bos_positions) > 0 and len(audio_eos_positions) > 0:
                        for bos_pos in audio_bos_positions:
                            eos_candidates = audio_eos_positions[audio_eos_positions > bos_pos]
                            if len(eos_candidates) > 0:
                                eos_pos = eos_candidates[0]
                                audio_tokens += eos_pos - bos_pos - 1
                                
                    
                    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
                        if hasattr(model.thinker.model.config, 'sparse_attention_config'):
                            model.thinker.model.config.sparse_attention_config['audio_tokens'] = audio_tokens.item() if hasattr(audio_tokens, 'item') else audio_tokens
                
                with torch.no_grad():
                    prefill_output = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        thinker_max_new_tokens=1,  
                        thinker_do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                prefill_end_event.record()
                
                
                total_start_event = torch.cuda.Event(enable_timing=True)
                total_end_event = torch.cuda.Event(enable_timing=True)

                total_start_event.record()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        thinker_max_new_tokens=5,  
                        thinker_do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                total_end_event.record()
                
                
                torch.cuda.synchronize()
                prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
                total_time = total_start_event.elapsed_time(total_end_event) / 1000.0  
                
                
                output_text = processor.batch_decode(
                    output, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                
                if "assistant\n" in output_text:
                    
                    assistant_start = output_text.rfind("assistant\n") + len("assistant\n")
                    output_text = output_text[assistant_start:].strip()
                
                
                if hasattr(output, 'shape') and len(output.shape) > 1:
                    output_tokens = output.shape[1] - inputs["input_ids"].shape[1]
                else:
                    output_tokens = 0  
                
                
                output_text = output_text.strip()
                
                
                choices = [sample['choice_a'], sample['choice_b'], sample['choice_c'], sample['choice_d']]
                predicted_answer = extract_acoustic_scene_answer(output_text, choices)
                
                
                is_correct = (predicted_answer == ground_truth)
                
                
                all_predictions.append(predicted_answer if predicted_answer else "ERROR")
                all_ground_truths.append(ground_truth)
                
                
                scene_stats[scene_label]["total"] += 1
                if is_correct:
                    scene_stats[scene_label]["correct"] += 1
                
                
                timing_stats.add_record(prefill_time, total_time, output_tokens, input_token_length, audio_token_length, i)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                traceback.print_exc()
                
                
                output_text = ""
                predicted_answer = "ERROR"
                is_correct = False
                prefill_time = 0
                total_time = 0
                output_tokens = 0
                
                all_predictions.append("ERROR")
                all_ground_truths.append(ground_truth)
                scene_stats[scene_label]["total"] += 1
                
                
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            
            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "scene_label": scene_label,
                "ground_truth": ground_truth,
                "model_output": output_text,
                "extracted_answer": predicted_answer,
                "is_correct": is_correct,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "total_time": total_time
            }
            
            all_sample_results.append(sample_result)
            
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p != "ERROR" and t != "ERROR") / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p != "ERROR" and t != "ERROR"))
            
            
            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                pbar.set_postfix({
                    'Samples': f'{sample_count}/{len(samples)}',
                    'Accuracy': f'{current_accuracy:.3f}',
                    'Scene': scene_label[:12] + '...' if len(scene_label) > 12 else scene_label
                })
                
                if is_screen_env:

                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.3f}")
            else:
                pbar.set_postfix({
                    'Samples': f'{sample_count}/{len(samples)}',
                    'Accuracy': f'{current_accuracy:.3f}',
                    'Scene': scene_label[:12] + '...' if len(scene_label) > 12 else scene_label
                })
            
            
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                
                if (i + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {i+1}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            pbar.update()
    
    
    all_scene_labels = list(set(all_ground_truths))
    acoustic_metrics = calculate_acoustic_metrics(all_predictions, all_ground_truths, all_scene_labels)
    final_stats = timing_stats.get_summary()
    
    
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    
    
    for scene in scene_stats:
        if scene_stats[scene]["total"] > 0:
            scene_stats[scene]["accuracy"] = scene_stats[scene]["correct"] / scene_stats[scene]["total"]
        else:
            scene_stats[scene]["accuracy"] = 0.0
    
    
    results = {
        "samples": all_sample_results,
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": correct_samples / total_samples if total_samples > 0 else 0,
            "scene_stats": scene_stats,
            "metrics": acoustic_metrics,
            "timing": final_stats,
            "config": {
                "gpu_id": gpu_id,
                "model_path": model_path,
                "prune_layer_idx": prune_layer_idx,
                "prune_ratio": prune_ratio,
                "prune_method": method_is,
                "sample_limit": sample_limit,
                "timing_sample_count": final_stats["samples"]
            }
        }
    }
    
    
    json_output_file = f'{result_dir}/TAU_results_qwen25_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    
    timing_stats.export_to_json(timing_output_file)
    

    print("\n=== TAU Acoustic Scene Classification Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total Samples: {total_samples}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 Score: {acoustic_metrics['f1_score']:.4f}")
    print(f"Precision: {acoustic_metrics['precision']:.4f}")
    print(f"Recall: {acoustic_metrics['recall']:.4f}")
    print(f"Valid Samples: {acoustic_metrics['valid_samples']}/{acoustic_metrics['total_samples']}")

    
    sorted_scenes = sorted(
        [(scene, stats["accuracy"], stats["correct"], stats["total"]) 
         for scene, stats in results["summary"]["scene_stats"].items()],
        key=lambda x: x[1], reverse=True
    )

    print("\nScene Accuracy:")
    for scene, acc, correct, total in sorted_scenes:
        print(f"  {scene}: {acc:.2%} ({correct}/{total})")

    print(f"\nPruning Method: {method_is}, Pruning Ratio: {prune_ratio}")
    timing_sample_count = final_stats["samples"]
    print(f"\n=== Timing Statistics (First 100 Samples, Excluding First) ===")
    print(f"Number of Samples: {timing_sample_count}")
    print(f"Average Total Time: {final_stats['avg_total_time']:.4f} seconds")
    print(f"Average Prefill Time: {final_stats['avg_prefill_time']:.4f} seconds")
    print(f"Average Input Tokens: {final_stats['avg_input_tokens']:.1f}")
    print(f"Average Audio Tokens: {final_stats['avg_audio_tokens']:.1f}")
    print(f"Results saved to: {json_output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()
