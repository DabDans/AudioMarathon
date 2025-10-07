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

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON compatibility"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

_AUDIO_TOKEN_ID = 151646          
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    def process_mm_info(messages, use_audio_in_video=True):
        """Simplified multimodal information processing function"""
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

from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)

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

data_path = os.environ.get("VESUS_DATA_PATH", 
    "/data/to/your/dataset/path//VESUS")
emotion_json_file = os.path.join(data_path, "audio_emotion_dataset.json")
result_dir = os.environ.get("RESULTS_DIR", './VESUS_Results')
os.makedirs(result_dir, exist_ok=True)

def get_gpu_memory_usage():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

def calculate_emotion_metrics(predictions, ground_truths, emotion_labels):
    """Calculate emotion classification metrics: accuracy, precision, recall and F1 score"""
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

class VESUSTimingStats:
    """Track inference time statistics for VESUS emotion recognition tasks, supporting CUDA Event accurate measurement"""
    def __init__(self):
        self.timing_records = []
        self.emotion_stats = defaultdict(list)
        self.person_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.use_cuda_events = torch.cuda.is_available()
    
    def add_record(self, prefill_time, total_time, output_tokens, input_tokens, 
                   emotion_label=None, person_id=None):
        
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += total_time  
        self.total_tokens += output_tokens
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": total_time,  
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / total_time if total_time > 0 else 0,
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
                emotion_prefill = sum(r["prefill_time"] for r in records)
                emotion_decode = sum(r["decode_time"] for r in records)
                emotion_tokens = sum(r["output_tokens"] for r in records)
                
                emotion_summaries[emotion] = {
                    "samples": len(records),
                    "avg_prefill_time": emotion_prefill / len(records),
                    "avg_decode_time": emotion_decode / len(records),
                    "avg_total_time": (emotion_prefill + emotion_decode) / len(records),
                    "avg_tokens": emotion_tokens / len(records),
                    "avg_tokens_per_sec": emotion_tokens / emotion_decode if emotion_decode > 0 else 0
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

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file according to Qwen2.5-Omni requirements"""
    
    try:
        full_audio_path = os.path.join(data_path, audio_path)
        
        if not os.path.exists(full_audio_path):
            print(f"Audio file does not exist: {full_audio_path}")
            return None
        
        try:
            audio, sr = librosa.load(full_audio_path, sr=target_sr, mono=True)
            print(f"Loaded successfully with librosa: shape={audio.shape}, sample rate {sr}Hz")
        except Exception as e:
            print(f"Librosa loading failed: {e}")
            
            try:
                audio, sample_rate = sf.read(full_audio_path)
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"Processed successfully with soundfile: shape={audio.shape}, sample rate {sr}Hz")
                
            except Exception as e:
                print(f"Soundfile loading also failed: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generated silent audio as fallback")
        

        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def load_vesus_dataset(json_file_path):
    
    if not os.path.exists(json_file_path):
        print(f"Error: Dataset file does not exist: {json_file_path}")
        return []
    
    print(f"Loading VESUS emotion dataset: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        valid_samples = []
        filtered_count = 0
        
        for item in data:
            if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
                person_id_str = str(item.get('person_id', '')).strip()
                emotion_label_lower = str(item.get('emotion_label', '')).strip().lower()

                if person_id_str in ['2', '10'] and emotion_label_lower == 'happy':
                    filtered_count += 1
                    continue
                
                valid_samples.append(item)

        print(f"Filtered {filtered_count} samples (person 2 and person 10 with happy emotion)")
        print(f"Loaded {len(valid_samples)} valid samples")

        emotion_counts = defaultdict(int)
        person_emotion_counts = defaultdict(int)
        for sample in valid_samples:
            emotion = sample.get('emotion_label', 'unknown')
            person = sample.get('person_id', 'unknown')
            emotion_counts[emotion] += 1
            person_emotion_counts[str(person)] += 1 
        
        print(f"Emotion distribution: {dict(emotion_counts)}")
        print(f"Person sample count statistics:")
        for person, count in person_emotion_counts.items():
            print(f"  person {person}: {count} samples")
        
        return valid_samples
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []
    
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
                if any(keyword in text_lower for keyword in keywords) and any(keyword in choice_text for keyword in keywords):
                    option_index = ord(choice_key[-1]) - ord('a')  
                    return chr(65 + option_index)  
    
    return ""

def create_emotion_prompt(sample):
    """Create emotion recognition task prompt (consistent with original version)"""
    question = sample.get("question", "What emotion is expressed in this audio segment?")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")
    
    prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D)."""
    
    return prompt

def main():
    print(f"\n=== VESUS Emotion Recognition Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {data_path}")
    print(f"JSON file: {emotion_json_file}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)
    
    output_file = f'{result_dir}/VESUS_results_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    timing_output_file = f'{result_dir}/VESUS_timing_stats_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
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
    model.eval()
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
        print(f"Initialized thinker.model.config pruning configuration parameters")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("Model loaded successfully")

    samples = load_vesus_dataset(emotion_json_file)
    
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
    print(f"GPU memory after model loading - Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")

    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="VESUS Evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            audio_path = sample.get("path", "")
            
            full_audio_path = os.path.join(data_path, audio_path)
            if not os.path.exists(full_audio_path):
                print(f"Skipping sample {idx+1}/{len(samples)}: Audio file does not exist - {full_audio_path}")
                progress_bar.update()
                continue
            
            emotion_label = sample.get("emotion_label", "unknown")
            person_id = sample.get("person_id", "unknown")
            answer_gt = sample.get("answer_gt", "").upper()
            
            prompt_text = create_emotion_prompt(sample)

            task_instruction = "You are a helpful assistant that analyzes speech audio to recognize emotions. Please listen to the voice carefully and identify the emotional state of the speaker."
            full_user_prompt = f"{task_instruction}\n\n{prompt_text}"

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
                        {"type": "audio", "audio": os.path.join(data_path, audio_path)},  
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
                    
                    print(f"Sample {idx+1}: Detected audio tokens - Start position: {audio_token_start}, Length: {audio_token_length}")
                    
                    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                        model.thinker.model.config.image_layer_idx = False  
                        model.thinker.model.config.audio_layer_idx = prune_layer_idx
                        model.thinker.model.config.audio_token_num = audio_token_length
                        model.thinker.model.config.audio_token_start = audio_token_start
                        model.thinker.model.config.audio_prune_ratio = prune_ratio
                        model.thinker.model.config.random = use_random
                        model.thinker.model.config.frame = use_frame 
                    
            if not audio_detected:
                print(f"Sample {idx+1}: No audio tokens detected")
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                    model.thinker.model.config.audio_layer_idx = None
                    model.thinker.model.config.audio_prune_ratio = 0

            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            
            with torch.no_grad():
                _ = model.generate(
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
                    thinker_max_new_tokens=10,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            total_end_event.record()
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
            total_gpu_time = total_start_event.elapsed_time(total_end_event) / 1000.0  
            
            response = processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            if "assistant\n" in response:
                assistant_start = response.rfind("assistant\n") + len("assistant\n")
                response = response[assistant_start:].strip()
            
            output_tokens = len(output[0]) - len(inputs["input_ids"][0])
            input_tokens = len(inputs["input_ids"][0])
            
            choices = {
                'choice_a': sample.get('choice_a', ''),
                'choice_b': sample.get('choice_b', ''),
                'choice_c': sample.get('choice_c', ''),
                'choice_d': sample.get('choice_d', '')
            }
            predicted_answer = extract_emotion_answer(response, choices)
            
            is_correct = (predicted_answer == answer_gt)
            if is_correct:
                total_correct += 1
            
            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1
            
            if is_correct:
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1
            
            result = {
                "idx": idx,
                "audio_path": audio_path,
                "emotion_label": emotion_label,
                "person_id": person_id,
                "question": sample.get("question", ""),
                "choices": choices,
                "answer_gt": answer_gt,
                "predicted_answer": predicted_answer,
                "response": response,
                "is_correct": is_correct,
                "prefill_time": prefill_time,
                "total_time": total_gpu_time,
                "output_tokens": output_tokens,
                "input_tokens": input_tokens
            }
            results.append(result)
            
            if idx > 0:
                timing_stats.add_record(
                    prefill_time, total_gpu_time, output_tokens, input_tokens,
                    emotion_label, person_id
                )
            
            current_accuracy = total_correct / (idx + 1) if (idx + 1) > 0 else 0
            
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'acc': f'{current_accuracy:.3f}',
                    'pred': predicted_answer,
                    'gt': answer_gt,
                    'emotion': emotion_label[:8]
                })
                
                if is_screen_env:
                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.3f}")
            else:
                progress_bar.set_postfix({
                    'acc': f'{current_accuracy:.3f}',
                    'pred': predicted_answer,
                    'gt': answer_gt,
                    'emotion': emotion_label[:8]
                })
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing sample {idx}: {error_msg}")
            traceback.print_exc()
            
            if "No such file" in error_msg or "does not exist" in error_msg or "FileNotFoundError" in error_msg:
                print(f"Skipping sample {idx+1}/{len(samples)}: Audio file processing failed - {sample.get('path', '')}")
                continue
            
            result = {
                "idx": idx,
                "audio_path": sample.get("path", ""),
                "emotion_label": sample.get("emotion_label", "unknown"),
                "person_id": sample.get("person_id", "unknown"),
                "question": sample.get("question", ""),
                "choices": {},
                "answer_gt": sample.get("answer_gt", "").upper(),
                "predicted_answer": "ERROR",
                "response": "",
                "is_correct": False,
                "prefill_time": 0,
                "decode_time": 0,
                "total_time": 0,
                "output_tokens": 0,
                "input_tokens": 0
            }
            results.append(result)
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
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
        else:
            emotion_accuracies[emotion] = 0.0

    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]
        else:
            person_accuracies[person] = 0.0

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
            "model_path": model_path,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file,
            "timing_sample_count": timing_stats.total_samples
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

    print(f"\n=== VESUS Emotion Recognition Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"F1 Score: {emotion_metrics['f1_score']:.4f}")
    print(f"Precision: {emotion_metrics['precision']:.4f}")
    print(f"Recall: {emotion_metrics['recall']:.4f}")
    print(f"Valid samples: {emotion_metrics['valid_samples']}/{emotion_metrics['total_samples']}")
    print(f"Emotion-wise accuracy:")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    timing_sample_count = summary["config"]["timing_sample_count"]
    print(f"\n=== Timing Statistics (CUDA Events Accurate Measurement) ===")
    print(f"Timing sample count: {timing_sample_count} (excluding first sample)")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()