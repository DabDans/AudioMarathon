


"""
VoxCeleb Age Classification Model Evaluation Script (Qwen2.5-Omni version)
For evaluating Qwen2.5-Omni model performance on VoxCeleb age classification tasks
"""

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


sys.path.append("path/to/your/Modeling")
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
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")


data_path_root = '/path/to/your/subsetVoxCeleb/concatenated_audio'  
result_dir = './Vox_Results'
os.makedirs(result_dir, exist_ok=True)

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0


class FolderTimingStats:
    """Track inference timing statistics for each folder"""
    def __init__(self):
        self.folder_stats = {}
        self.current_folder = None
    
    def set_current_folder(self, folder_name):
        self.current_folder = folder_name
        if folder_name not in self.folder_stats:
            self.folder_stats[folder_name] = {
                "samples": 0,
                "total_prefill_time": 0.0,
                "total_decode_time": 0.0,
                "total_tokens": 0,
                "timing_records": []
            }
    
    def add_record(self, prefill_time, decode_time, output_tokens):
        if self.current_folder is None:
            return
        
        folder_data = self.folder_stats[self.current_folder]
        folder_data["samples"] += 1
        folder_data["total_prefill_time"] += prefill_time
        folder_data["total_decode_time"] += decode_time
        folder_data["total_tokens"] += output_tokens
        
        
        folder_data["timing_records"].append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        })
    
    def export_to_json(self, output_file):
        """Export statistics data to JSON file"""
        result = {
            "folder_summaries": {
                folder: {
                    "folder": folder,
                    "samples": stats["samples"],
                    "avg_prefill_time": stats["total_prefill_time"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_decode_time": stats["total_decode_time"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_total_time": (stats["total_prefill_time"] + stats["total_decode_time"]) / stats["samples"] if stats["samples"] > 0 else 0,
                    "total_tokens": stats["total_tokens"],
                    "avg_tokens": stats["total_tokens"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_tokens_per_sec": stats["total_tokens"] / stats["total_decode_time"] if stats["total_decode_time"] > 0 else 0
                }
                for folder, stats in self.folder_stats.items() if stats["samples"] > 0
            },
            "detailed_records": self.folder_stats
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files according to Qwen2.5-Omni requirements"""
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"使用librosa加载成功: 形状={audio.shape}, 采样率={sr}Hz")
        except Exception as e:
            print(f"librosa加载失败: {e}")
            
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"soundfile处理成功: 形状={audio.shape}, 采样率={sr}Hz")
                
            except Exception as e:
                print(f"soundfile加载也失败: {e}")
                
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("生成静音替代音频")
        
        
        if len(audio) == 0:
            print("警告: 音频为空")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            
        
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
        
    except Exception as e:
        print(f"音频处理出错: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

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
            print(f"警告: 文件不存在:{wav_path}")
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
        print(f"Apply sample limit: randomly selecting {sample_limit} from {len(all_samples)} samples")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"限制后样本数量: {len(all_samples)}")
    
    
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
    """从模型输出文本中提取年龄组答案，处理直接回复a/b/c/d/e的情况"""
    text_lower = text.lower().strip()
    
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)') or text_lower.endswith(' a'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)') or text_lower.endswith(' b'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)') or text_lower.endswith(' c'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)') or text_lower.endswith(' d'):
        return choices["choice_d"]
    if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)') or text_lower.endswith(' e'):
        return choices["choice_e"]
    
    
    if re.search(r'\ba\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['b', 'c', 'd', 'e']):
        return choices["choice_a"]
    if re.search(r'\bb\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'c', 'd', 'e']):
        return choices["choice_b"]
    if re.search(r'\bc\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'd', 'e']):
        return choices["choice_c"]
    if re.search(r'\bd\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'c', 'e']):
        return choices["choice_d"]
    if re.search(r'\be\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'c', 'd']):
        return choices["choice_e"]
        
    
    for option, choice_text in choices.items():
        option_letter = option[-1].lower()  
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"answer {option_letter}" in text_lower:
            return choice_text
    
    
    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    
    
    if len(choice_matches) == 1:
        return choice_matches[0]
    
    
    return ""

def calculate_age_classification_metrics(y_true, y_pred, age_groups=None):
    """
    计算年龄分类的详细评估指标
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表 
        age_groups: 年龄组列表，如果为None则自动从数据中获取
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    
    if age_groups is None:
        age_groups = list(set(y_true))
        age_groups.sort()  
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in age_groups and pred_label in age_groups:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'per_class_metrics': {},
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true),
            'age_groups': age_groups
        }
    
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=age_groups, average=None, zero_division=0
    )
    
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=age_groups, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=age_groups, average='weighted', zero_division=0
    )
    
    
    per_class_metrics = {}
    for i, age_group in enumerate(age_groups):
        per_class_metrics[age_group] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=age_groups,
        target_names=[f"Age Group: {ag}" for ag in age_groups],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true),
        'age_groups': age_groups
    }

def create_age_prompt(sample):
    """创建年龄分类任务的提示词（与原版保持一致）"""
    question = sample.get("question", "")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")
    choice_e = sample.get("choice_e", "")
    
    
    prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
E) {choice_e}

Please select the correct answer (A, B, C, D, or E)."""
    
    return prompt

def main():
    print(f"\n=== VoxCeleb年龄分类评测配置 (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"剪枝层索引: {prune_layer_idx}")
    print(f"剪枝比例: {prune_ratio}")
    print(f"剪枝方法: {method_is}")
    print(f"数据路径: {data_path_root}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 40)
    
    
    output_file = f'{result_dir}/VoxCeleb_age_results_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.jsonl'
    timing_output_file = f'{result_dir}/VoxCeleb_age_timing_stats_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")
    
    
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
        print(f"初始化thinker.model.config剪枝配置参数")
    
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("模型加载成功")
    
    
    timing_stats = FolderTimingStats()
    
    
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    
    
    grouped_samples = {}
    for sample in samples:
        speaker_id = sample["speaker_id"]
        if speaker_id not in grouped_samples:
            grouped_samples[speaker_id] = []
        grouped_samples[speaker_id].append(sample)
    
    
    age_group_counts = {}
    for s in samples:
        group = s["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("年龄组统计:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count}个样本")
    
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "age_group_stats": {},
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
    
    
    for group in age_group_counts.keys():
        results["summary"]["age_group_stats"][group] = {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0
        }

    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("检测到screen或非交互式环境，使用简化进度显示")
    
    
    tqdm_kwargs = {
        'ascii': True,      
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }
    
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    
    with tqdm(total=len(grouped_samples), desc="处理说话人组", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        
        
        for speaker_id, items in grouped_samples.items():
            folder_count += 1
            pbar_folders.set_description(f"处理说话人[{folder_count}/{total_folders}]: {speaker_id}")
            
            
            timing_stats.set_current_folder(speaker_id)
            
            
            sample_count = 0
            total_samples = len(items)
            
            
            for i, item in enumerate(items):
                sample_count += 1
                try:
                    wav_path = item["wav_path"]
                    age_group = item["age_group"]
                    
                    
                    audio_np = prepare_audio_for_qwen_omni(wav_path, target_sr=16000)
                    
                    if audio_np is None:
                        print(f"跳过样本: 无法加载音频 {wav_path}")
                        continue
                    
                    
                    prompt_text = create_age_prompt(item)

                    
                    task_instruction = "You are a helpful assistant that analyzes speech audio to estimate speaker age. Please listen to the voice carefully and classify the speaker's age group."
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
                                {"type": "audio", "audio": wav_path},  
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
                    
                    
                    
                    generated_tokens = out_ids[:, inputs["input_ids"].shape[-1]:]
                    response = processor.batch_decode(
                        generated_tokens, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    
                    if not response.strip():
                        response = processor.batch_decode(
                            out_ids, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )[0]
                    
                    
                    output_tokens = len(out_ids[0]) - len(inputs["input_ids"][0])
                    input_tokens = len(inputs["input_ids"][0])
                    
                    
                    choices = {
                        "choice_a": item["choice_a"],
                        "choice_b": item["choice_b"],
                        "choice_c": item["choice_c"],
                        "choice_d": item["choice_d"],
                        "choice_e": item["choice_e"]
                    }
                    
                    extracted_answer = extract_age_answer(response, choices)
                    is_correct = (extracted_answer == age_group)
                    
                    
                    if results["summary"]["total_samples"] < 5:
                        print(f"\n=== 年龄分类样本 {results['summary']['total_samples']} 调试信息 ===")
                        print(f"模型原始输出: '{response}'")
                        print(f"提取的答案: '{extracted_answer}'")
                        print(f"正确答案: '{age_group}'")
                        print(f"是否正确: {is_correct}")
                        print(f"输出token数量: {output_tokens}")
                        print("=" * 40)
                    
                    
                    results["summary"]["total_samples"] += 1
                    results["summary"]["age_group_stats"][age_group]["total"] += 1
                    
                    if is_correct:
                        results["summary"]["correct_samples"] += 1
                        results["summary"]["age_group_stats"][age_group]["correct"] += 1
                    
                    
                    if results["summary"]["total_samples"] > 1:
                        results["summary"]["timing"]["total_prefill_time"] += prefill_time
                        results["summary"]["timing"]["total_decode_time"] += decode_time
                        results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                        timing_stats.add_record(prefill_time, decode_time, output_tokens)
                    
                    
                    result_item = {
                        "speaker_id": speaker_id,
                        "age_group": age_group,
                        "wav_path": wav_path,
                        "question": item["question"],
                        "choices": choices,
                        "ground_truth": age_group,
                        "extracted_answer": extracted_answer,
                        "response": response,
                        "is_correct": is_correct,
                        "prefill_time": prefill_time,
                        "decode_time": decode_time,
                        "total_time": prefill_time + decode_time,
                        "output_tokens": output_tokens,
                        "input_tokens": input_tokens,
                        "audio_tokens": int(audio_tokens) if isinstance(audio_tokens, (int, float)) else (audio_tokens.item() if hasattr(audio_tokens, 'item') else 0)
                    }
                    results["samples"].append(result_item)
                    
                    
                    if is_screen_env and sample_count % 10 == 0:
                        current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                        print(f"      样本进度: {sample_count}/{total_samples}, 准确率: {current_accuracy:.3f}, 预测: {extracted_answer}, 真实: {age_group}")
                    
                    
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    
                    if sample_count % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        
                        if sample_count % 100 == 0:
                            allocated, reserved = get_gpu_memory_usage()
                            print(f"      [样本 {sample_count}] GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
                    
                except Exception as e:
                    print(f"处理样本时出错({speaker_id}[{sample_count}]): {e}")
                    traceback.print_exc()
                    
                    
                    result_item = {
                        "speaker_id": speaker_id,
                        "age_group": item.get("age_group", "unknown"),
                        "wav_path": item.get("wav_path", ""),
                        "question": item.get("question", ""),
                        "choices": {},
                        "ground_truth": item.get("age_group", "unknown"),
                        "extracted_answer": "ERROR",
                        "response": "",
                        "is_correct": False,
                        "prefill_time": 0,
                        "decode_time": 0,
                        "total_time": 0,
                        "output_tokens": 0
                    }
                    results["samples"].append(result_item)
                    results["summary"]["total_samples"] += 1
                    
                    
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    continue
            
            
            pbar_folders.update()
    
    
    total_samples = results["summary"]["total_samples"]
    timing_sample_count = max(0, total_samples - 1)  
    if timing_sample_count > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / timing_sample_count
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / timing_sample_count
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / timing_sample_count
    else:
        results["summary"]["timing"]["avg_prefill_time"] = 0
        results["summary"]["timing"]["avg_decode_time"] = 0
        results["summary"]["timing"]["avg_total_time"] = 0
    
    
    results["summary"]["timing"]["timing_sample_count"] = timing_sample_count
    
    
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    
    
    for age_group, stats in results["summary"]["age_group_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    
    y_true = [sample["ground_truth"] for sample in results["samples"]]
    y_pred = [sample["extracted_answer"] for sample in results["samples"]]
    
    
    all_age_groups = list(results["summary"]["age_group_stats"].keys())
    all_age_groups.sort()  
    
    
    detailed_metrics = calculate_age_classification_metrics(y_true, y_pred, all_age_groups)
    
    
    results["summary"]["sklearn_metrics"] = detailed_metrics
    
    
    json_output_file = f'{result_dir}/VoxCeleb_age_results_qwen25_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    
    timing_stats.export_to_json(timing_output_file)
    
    
    print("\n=== 年龄分类评测结果摘要 (Qwen2.5-Omni) ===")
    print(f"总样本数: {total_samples}")
    print(f"总准确率: {results['summary']['accuracy']:.2%}")
    
    
    metrics = results["summary"]["sklearn_metrics"]
    print(f"\n=== 详细评估指标 (sklearn) ===")
    print(f"准确率(Accuracy): {metrics['accuracy']:.4f}")
    print(f"F1分数 (宏平均): {metrics['f1_macro']:.4f}")
    print(f"F1分数 (加权平均): {metrics['f1_weighted']:.4f}")
    print(f"精度 (宏平均): {metrics['precision_macro']:.4f}")
    print(f"召回率(宏平均): {metrics['recall_macro']:.4f}")
    
    print(f"\n=== 各年龄组评估指标 ===")
    for age_group, per_class_metrics in metrics['per_class_metrics'].items():
        print(f"{age_group}:")
        print(f"  精度: {per_class_metrics['precision']:.4f}")
        print(f"  召回率: {per_class_metrics['recall']:.4f}")
        print(f"  F1分数: {per_class_metrics['f1_score']:.4f}")
        print(f"  样本数: {per_class_metrics['support']}")
    
    print("\n=== 传统准确率统计 ===")
    for age_group, stats in results["summary"]["age_group_stats"].items():
        print(f"  {age_group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"\n=== 推理时间统计 ===")
    print(f"统计样本数: {timing_sample_count} (排除第一个样本)")
    print(f"平均推理时间: {results['summary']['timing']['avg_total_time']:.4f}秒")
    print(f"平均 Prefill 时间: {results['summary']['timing']['avg_prefill_time']:.4f}秒")
    print(f"平均 Decode 时间: {results['summary']['timing']['avg_decode_time']:.4f}秒")
    
    print(f"\n=== 分类详细报告 ===")
    print(metrics['classification_report'])
    
    print(f"\n结果已保存到: {json_output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

    
    try:
        
        input_tokens_list = [s.get('input_tokens', 0) for s in results['samples'] if 'input_tokens' in s]
        audio_tokens_list = [s.get('audio_tokens', 0) for s in results['samples'] if 'audio_tokens' in s]
        avg_input_tokens = float(sum(input_tokens_list) / len(input_tokens_list)) if input_tokens_list else 0.0
        avg_audio_tokens = float(sum(audio_tokens_list) / len(audio_tokens_list)) if audio_tokens_list else 0.0

        metrics = results['summary'].get('sklearn_metrics', {})
        timing_block = results['summary'].get('timing', {})
        simple_summary = {
            "task": "VoxCeleb_Age",
            "macro_f1": metrics.get('f1_macro', 0.0),
            "prefill_time_avg": timing_block.get('avg_prefill_time', 0.0),
            "decode_time_avg": timing_block.get('avg_decode_time', 0.0),
            "total_time_avg": timing_block.get('avg_total_time', 0.0),
            "avg_input_tokens": avg_input_tokens,
            "avg_audio_tokens": avg_audio_tokens
        }
        simple_path = os.path.join(os.path.dirname(json_output_file), 'VoxCeleb_age_simple_summary.json')
        with open(simple_path, 'w', encoding='utf-8') as sfp:
            json.dump(simple_summary, sfp, ensure_ascii=False, indent=2)
        print(f"精简summary已保存到: {simple_path}")
    except Exception as e:
        print(f"保存精简summary出错: {e}")

if __name__ == "__main__":
    main()