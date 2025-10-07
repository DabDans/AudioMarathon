#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import argparse

random.seed(42)

sys.path.append("/data/to/your/Qwen_2.5/folder")
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

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    def process_mm_info(messages, use_audio_in_video=True):
        audios = []
        images = []
        videos = []
        
        for message in messages:
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if content_item.get("type") == "audio":
                        audio_data = content_item.get("audio")
                        if isinstance(audio_data, str):
                            audio = prepare_audio_for_qwen_omni(audio_data)
                            audios.append(audio)
                        else:
                            audios.append(audio_data)
        
        return audios, images, videos

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")

prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
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

data_path_root = '/data/to/your/dataset/path//VoxCeleb/concatenated_audio'  
result_dir = './Vox_Results'
os.makedirs(result_dir, exist_ok=True)

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

def calculate_metrics(predictions, ground_truths):
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['male', 'female'] and t in ['male', 'female']]
    
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
    
    label_map = {'male': 0, 'female': 1}
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

class GlobalTimingStats:
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_tokens = 0
        self.timing_records = []
    
    def add_record(self, prefill_time, decode_time, output_tokens):
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
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def print_summary(self):
        summary = self.get_summary()
        print(f"\n=== Timing Statistics Summary ===")
        print(f"Valid samples: {summary['samples']}")
        print(f"Average Prefill time: {summary['avg_prefill_time']:.4f} seconds")
        print(f"Average Decode time: {summary['avg_decode_time']:.4f} seconds")
        print(f"Average total time: {summary['avg_total_time']:.4f} seconds")
        print(f"Average tokens/sec: {summary['avg_tokens_per_sec']:.2f}")

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    
    try:
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio.shape}, sample_rate={sr}Hz")
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
                print(f"Soundfile processing successful: shape={audio.shape}, sample_rate={sr}Hz")
                
            except Exception as e:
                print(f"Soundfile loading also failed: {e}")
                
                try:
                    from scipy.io import wavfile
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_wav = temp_file.name
                    
                    print(f"Using ffmpeg conversion: {audio_path} -> {temp_wav}")
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
                    sr = target_sr
                    print(f"FFmpeg conversion successful: shape={audio.shape}, sample_rate={sr}Hz")
                    
                except Exception as e:
                    print(f"FFmpeg conversion failed: {e}")
                    audio = np.zeros(target_sr * 3, dtype=np.float32)  
                    sr = target_sr
                    print("Generated silence as replacement audio")
        
        if len(audio) == 0:
            print("Warning: audio is empty, creating 3 seconds of silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
        
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    meta_file = os.path.join(root_dir, "gender_id_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(wav_path):
            print(f"Warning: file does not exist: {wav_path}")
            continue
        
        speaker_id = item["speaker_id_original"]
        gender = item["answer_gt"].lower().strip()  
        
        all_samples.append({
            "speaker_id": speaker_id,
            "gender": gender,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "answer_gt": gender,
            "task": "Speaker_Gender_Identification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    
    male_samples = [sample for sample in all_samples if sample["gender"].lower() == "male"]
    female_samples = [sample for sample in all_samples if sample["gender"].lower() == "female"]
    print(f"Original sample count: male={len(male_samples)}, female={len(female_samples)}")
    
    min_samples_per_gender = min(len(male_samples), len(female_samples))
    
    if sample_limit > 0:
        max_per_gender = sample_limit // 2
        min_samples_per_gender = min(min_samples_per_gender, max_per_gender)
        print(f"Applied sample limit: maximum {min_samples_per_gender} samples per gender")
    
    if len(male_samples) > min_samples_per_gender:
        male_samples = random.sample(male_samples, min_samples_per_gender)
    
    if len(female_samples) > min_samples_per_gender:
        female_samples = random.sample(female_samples, min_samples_per_gender)
    
    balanced_samples = male_samples + female_samples
    
    random.shuffle(balanced_samples)
    
    print(f"Final sample count: male={len(male_samples)}, female={len(female_samples)}, total={len(balanced_samples)}")
    
    return balanced_samples

def extract_gender_answer(text, choice_a="male", choice_b="female"):
    text_lower = text.lower().strip()
    
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a_lower
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b_lower
        
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a_lower
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b_lower
    
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a_lower
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b_lower
    
    import re
    if choice_a_lower == "male" and choice_b_lower == "female":
        male_match = re.search(r'\bmale\b', text_lower) is not None
        female_match = re.search(r'\bfemale\b', text_lower) is not None
        
        if male_match and not female_match:
            return "male"
        if female_match and not male_match:
            return "female"
    
    return ""

def main():
    print(f"\n=== VoxCeleb Gender Recognition Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)
    
    output_file = f'{result_dir}/VoxCeleb_results_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.jsonl'
    timing_output_file = f'{result_dir}/VoxCeleb_timing_stats_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    
    print("Loading Qwen2.5-Omni model...")
    model_path = "/data/to/your/Qwen_2.5_Model/path/"
    device_map = {"": 0}  
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    

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
    print(f"Initialized thinker.model.config pruning configuration parameters")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("Model loaded successfully")
    
    timing_stats = GlobalTimingStats()
    
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    
    male_count = sum(1 for s in samples if s["gender"].lower() == "male")
    female_count = sum(1 for s in samples if s["gender"].lower() == "female")
    print(f"Gender statistics: male samples={male_count}, female samples={female_count}")
    
    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,      
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    print(f"Starting to process {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing VoxCeleb samples (Qwen2.5)", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        for i, sample in enumerate(samples):
            wav_path = sample['wav_path']
            speaker_id = sample["speaker_id"]
            ground_truth = sample["gender"].lower().strip()
            
            instruction = "Listen to this audio and identify the speaker's gender. Is this a male or female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'."
            
            try:
                audio_np = prepare_audio_for_qwen_omni(wav_path, target_sr=16000)
                
                if audio_np is None:
                    print(f"Skipping sample {i}: unable to load audio {wav_path}")
                    continue
                
                task_instruction = "You are a helpful assistant that analyzes speech audio to classify speaker gender. Please listen to the voice carefully and determine the speaker's gender."
                full_user_prompt = f"{task_instruction}\n\n{instruction}"

                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_np},
                            {"type": "text", "text": full_user_prompt}
                        ]
                    }
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
                        
                if not audio_detected:
                    model.thinker.model.config.audio_layer_idx = None
                    model.thinker.model.config.audio_prune_ratio = 0

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
                    outputs = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        thinker_max_new_tokens=1,  
                        thinker_do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                prefill_end_event.record()
                
                decode_start_event = torch.cuda.Event(enable_timing=True)
                decode_end_event = torch.cuda.Event(enable_timing=True)
                
                decode_start_event.record()
                out_ids = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=5,
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                decode_end_event.record()
                
                torch.cuda.synchronize()
                prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
                decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0  
                
                tokens = out_ids[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                resp = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                if not resp.strip():
                    resp = processor.batch_decode(
                        out_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                    output_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]
                
                predicted_gender = extract_gender_answer(resp, sample.get("choice_a", "male"), sample.get("choice_b", "female"))
                is_correct = (predicted_gender == ground_truth)
                
                if i < 5:
                    print(f"\n=== Sample {i} Debug Information ===")
                    print(f"Model raw output: '{resp}'")
                    print(f"Extracted answer: '{predicted_gender}'")
                    print(f"Correct answer: '{ground_truth}'")
                    print(f"Is correct: {is_correct}")
                    print(f"Output token count: {output_tokens}")
                    print("=" * 30)
                
                all_predictions.append(predicted_gender)
                all_ground_truths.append(ground_truth)
                
                if i > 0:
                    timing_stats.add_record(prefill_time, decode_time, output_tokens)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                traceback.print_exc()
                
                resp = "ERROR"
                predicted_gender = ""
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                
                all_predictions.append(predicted_gender)
                all_ground_truths.append(ground_truth)
            
            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "model_output": resp,
                "extracted_answer": predicted_gender,
                "is_correct": is_correct,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            }
            
            all_sample_results.append(sample_result)
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if (i + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {i+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p in ['male', 'female'] and t in ['male', 'female']) / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p in ['male', 'female'] and t in ['male', 'female']))
            
            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                pbar.set_postfix({
                    'Sample': f'{sample_count}/{len(samples)}',
                    'Accuracy': f'{current_accuracy:.3f}',
                    'Speaker': speaker_id[:8] + '...' if len(speaker_id) > 8 else speaker_id,
                    'Predicted': predicted_gender,
                    'True': ground_truth
                })
                
                if is_screen_env:
                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.3f}")
            else:
                pbar.set_postfix({
                    'Sample': f'{sample_count}/{len(samples)}',
                    'Accuracy': f'{current_accuracy:.3f}',
                    'Speaker': speaker_id[:8] + '...' if len(speaker_id) > 8 else speaker_id,
                    'Predicted': predicted_gender,
                    'True': ground_truth
                })
            
            pbar.update()
    
    metrics_result = calculate_metrics(all_predictions, all_ground_truths)
    final_stats = timing_stats.get_summary()
    
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    
    male_samples = [r for r in all_sample_results if r['ground_truth'] == 'male']
    female_samples = [r for r in all_sample_results if r['ground_truth'] == 'female']
    
    male_correct = sum(1 for r in male_samples if r['is_correct'])
    female_correct = sum(1 for r in female_samples if r['is_correct'])
    
    results = {
        "samples": all_sample_results,
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": correct_samples / total_samples if total_samples > 0 else 0,
            "male_total": len(male_samples),
            "male_correct": male_correct,
            "male_accuracy": male_correct / len(male_samples) if len(male_samples) > 0 else 0,
            "female_total": len(female_samples),
            "female_correct": female_correct,
            "female_accuracy": female_correct / len(female_samples) if len(female_samples) > 0 else 0,
            "metrics": metrics_result,
            "timing": final_stats,
            "config": {
                "gpu_id": gpu_id,
                "model_path": model_path,
                "prune_layer_idx": prune_layer_idx,
                "prune_ratio": prune_ratio,
                "prune_method": method_is,
                "sample_limit": sample_limit,
                "data_path": data_path_root,
                "timing_sample_count": final_stats["samples"]
            }
        }
    }
    
    json_output_file = f'{result_dir}/VoxCeleb_results_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n=== Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Male accuracy: {results['summary']['male_accuracy']:.2%} ({results['summary']['male_correct']}/{results['summary']['male_total']})")
    print(f"Female accuracy: {results['summary']['female_accuracy']:.2%} ({results['summary']['female_correct']}/{results['summary']['female_total']})")
    print(f"F1 Score: {metrics_result['f1_score']:.4f}")
    print(f"Precision: {metrics_result['precision']:.4f}")  
    print(f"Recall: {metrics_result['recall']:.4f}")
    print(f"Valid predictions: {metrics_result['valid_samples']}/{metrics_result['total_samples']}")
    
    print(f"\n=== Inference Time Statistics ===")
    print(f"Statistical sample count: {final_stats['samples']} (excluding first sample)")
    print(f"Average inference time: {final_stats['avg_total_time']:.4f} seconds")
    print(f"Average Prefill time: {final_stats['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {final_stats['avg_decode_time']:.4f} seconds")
    print(f"Average throughput: {final_stats['avg_tokens_per_sec']:.2f} tokens/second")
    
    print(f"Results saved to: {json_output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()