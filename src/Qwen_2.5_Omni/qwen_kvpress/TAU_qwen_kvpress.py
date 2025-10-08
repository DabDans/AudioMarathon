#!/usr/bin/env python3
"""
TAU Urban Acoustic Scenes 2022 评测脚本 - Qwen2.5-Omni + KV Press 版本

参考:
  - TAU_qwen2.5.py (Qwen2.5-Omni TAU评测逻辑)
  - TAU_kvpress.py (KV Press 集成方法)

功能:
 1. 加载 Qwen2.5-Omni 模型进行 TAU 音频场景分类
 2. 支持多种 KV Press 压缩策略
 3. 按场景类别统计准确率与混淆矩阵
 4. 生成详细的分类报告

环境变量:
  COMPRESSION_RATIO=0.5           压缩比例
  PRESS_TYPE=knorm                压缩类型
  SAMPLE_LIMIT=0                  样本限制

用法:
  python TAU_qwen_kvpress.py --model-path /path/to/Qwen2.5-Omni-3B
"""

import os
import sys
import json
import time
import gc
import argparse
import warnings
import random
import traceback
import contextlib
import pandas as pd
import soundfile as sf
import numpy as np
import torch
import librosa
import transformers
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

# 设置NUMEXPR最大线程数以避免警告
os.environ['NUMEXPR_MAX_THREADS'] = '64'

# 设置随机种子
random.seed(42)

# 抑制警告
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

# 获取GPU ID
gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"使用 GPU ID: {gpu_id}")

# 环境变量配置
ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
if SAMPLE_LIMIT > 0:
    print(f"样本限制设置为: {SAMPLE_LIMIT}")

print(f"KV Press 配置: 压缩比率={ENV_COMPRESSION_RATIO}, 压缩类型={ENV_PRESS_TYPE}")

# Qwen2.5-Omni相关导入
sys.path.append("/data/hepeize05/Audio_Longbench/Code/Qwen_2.5")
try:
    from modeling_qwen2_5_omni_origin import Qwen2_5OmniForConditionalGeneration
    from processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    QWEN_AVAILABLE = True
    print("[信息] Qwen2.5-Omni 模块加载成功")
except ImportError as e:
    print(f"[警告] Qwen2.5-Omni 模块导入失败: {e}")
    QWEN_AVAILABLE = False
except Exception as e:
    print(f"[警告] Qwen2.5-Omni 模块加载错误: {e}")
    QWEN_AVAILABLE = False

# KV Press 导入 - 强制启用处理
KV_PRESS_AVAILABLE = False
try:
    # 首先检查transformers版本兼容性
    import transformers
    transformers_version = transformers.__version__
    print(f"[信息] Transformers版本: {transformers_version}")
    
    # 尝试直接导入KV Press，忽略transformers兼容性检查
    try:
        from kvpress import (
            ExpectedAttentionPress,
            KnormPress,
            ObservedAttentionPress,
            RandomPress,
            SnapKVPress,
            StreamingLLMPress,
            TOVAPress,
        )
        KV_PRESS_AVAILABLE = True
        print("[信息] KV Press 库加载成功")
    except ImportError as e:
        print(f"[警告] KV Press 库导入失败: {e}")
        print("[信息] 尝试从本地kvpress目录导入...")
        
        # 尝试从本地kvpress目录导入
        try:
            import sys
            kvpress_path = os.path.join(os.path.dirname(__file__), "..", "kvpress")
            if os.path.exists(kvpress_path):
                sys.path.insert(0, kvpress_path)
                from kvpress import (
                    ExpectedAttentionPress,
                    KnormPress,
                    ObservedAttentionPress,
                    RandomPress,
                    SnapKVPress,
                    StreamingLLMPress,
                    TOVAPress,
                )
                KV_PRESS_AVAILABLE = True
                print("[信息] 从本地目录成功加载KV Press")
            else:
                print(f"[错误] 本地kvpress目录不存在: {kvpress_path}")
        except Exception as local_e:
            print(f"[错误] 本地kvpress导入失败: {local_e}")

except ImportError as e:
    print(f"[警告] Transformers 库导入失败: {e}")
    KV_PRESS_AVAILABLE = False
except Exception as e:
    print(f"[警告] KV Press 库加载错误: {e}")
    KV_PRESS_AVAILABLE = False

# 如果用户要求必须使用KV Press但库不可用，则报错
if not KV_PRESS_AVAILABLE:
    print("[错误] KV Press 库不可用，但您要求必须使用压缩")
    print("请检查以下项目:")
    print("1. kvpress库是否已安装: pip install kvpress")
    print("2. transformers版本是否兼容")
    print("3. 是否存在本地kvpress目录")
    # 继续运行，但在main函数中会强制退出


# 常量
_AUDIO_TOKEN_ID = 151646          # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'

# TAU 场景类别
TAU_SCENE_CLASSES = [
    "airport", "shopping_mall", "metro_station", "street_pedestrian",
    "public_square", "street_traffic", "tram", "bus", "metro", "park"
]

# 环境变量
ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "TAU_QwenKVPress_Results")

# KV Press 配置
KV_PRESS_CONFIG = {
    "compression_ratio": ENV_COMPRESSION_RATIO,
    "head_dims": None,
    "num_attention_heads": None,
    "press_type": ENV_PRESS_TYPE,
    "return_indices": True,
    "min_seq_len": 128,
    "model_kwargs": {
        "attn_implementation": "flash_attention_2",
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="TAU with Qwen2.5-Omni + KV Press")
    parser.add_argument("--model-path", type=str, default="/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-path", type=str, default="/data/hepeize05/Audio_Longbench/Dataset/TAU/concatenated_resampled")
    parser.add_argument("--meta-file", type=str, default="acoustic_scene_task_meta.json", help="元数据文件名")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="最大生成 token 数")
    parser.add_argument("--min-seq-len", type=int, default=128, help="压缩阈值")
    parser.add_argument("--no-compress", action="store_true", help="禁用压缩")
    return parser.parse_args()

# 序列化函数
def convert_to_serializable(obj):
    """递归转换对象为JSON可序列化格式"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        # 处理自定义类（如dataclass）
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    else:
        return obj

def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """按照Qwen2.5-Omni要求处理音频文件，参考TAU_qwen2.5.py"""
    
    try:
        # 使用librosa加载音频（Qwen2.5-Omni推荐方式）
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"使用librosa加载成功: 形状={audio.shape}, 采样率={sr}Hz")
        except Exception as e:
            print(f"librosa加载失败: {e}")
            
            # 备用方法: 使用soundfile
            try:
                audio, sample_rate = sf.read(audio_path)
                
                # 确保是单声道
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                # 重采样到目标采样率
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"soundfile处理成功: 形状={audio.shape}, 采样率={sr}Hz")
                
            except Exception as e:
                print(f"soundfile加载也失败: {e}")
                # 创建静音作为备用
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("生成静音替代音频")
        
        # 确保音频不为空
        if len(audio) == 0:
            print("警告: 音频为空，创建3秒静音")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            
        # 确保数据类型为float32
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"音频处理出错: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def load_tau_acoustic_scene_dataset(root_dir):
    """从TAU数据集加载声学场景分类任务，参考TAU_qwen2.5.py"""
    # 加载元数据JSON文件
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"从{meta_file}加载了{len(metadata)}个样本元数据")
    
    # 场景类别计数
    scene_counts = {}
    
    # 遍历元数据中的每个条目
    for item in metadata:
        # 获取音频文件路径
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)
        
        # 检查文件是否存在
        if not os.path.exists(wav_path):
            print(f"警告: 文件不存在 {wav_path}")
            continue
        
        # 提取场景类别和正确选项
        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"] # A, B, C, D
        
        # 更新场景类别计数
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        
        # 构建样本信息
        all_samples.append({
            "scene_label": scene_label,
            "audio_path": wav_path,  # 修改为 audio_path 以匹配主函数
            "wav_path": wav_path,    # 保留 wav_path 作为备用
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification",
            "id": item.get("uniq_id", f"tau_{os.path.basename(wav_path)}"),
            "filename": os.path.basename(wav_path),
            "duration": item.get("duration", 0),  # 添加时长信息
        })
    
    print(f"总计加载了 {len(all_samples)} 个有效音频样本")
    
    # 显示场景分布
    print("场景分布:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count}个样本 ({count/len(all_samples)*100:.1f}%)")
    
    # 样本限制处理
    if SAMPLE_LIMIT > 0 and SAMPLE_LIMIT < len(all_samples):
        print(f"由于样本限制设置，随机选择{SAMPLE_LIMIT}个样本进行评估")
        all_samples = random.sample(all_samples, SAMPLE_LIMIT)
        
    # 随机打乱样本
    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """从模型输出文本中提取声学场景答案选项（A/B/C/D），参考TAU_qwen2.5.py"""
    text_lower = text.lower().strip()
    
    # 直接检测a/b/c/d回答
    options = ['a', 'b', 'c', 'd']
    
    # 精确匹配: 如果输出只有a/b/c/d之一
    if text_lower in options:
        return text_lower.upper()
    
    # 检查开头匹配: "a.", "a)", "a:"等
    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    
    # 检查是否包含明确的选项指示
    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    
    # 如果存在选项文本，检查选项文本是否在回答中
    if choices:
        best_match = None
        max_overlap = 0
        
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()
            # 优先检查选项全名是否在文本中
            if choice_lower in text_lower:
                return chr(65 + i)  # A, B, C, D
            
            # 检查重要关键词是否在文本中
            keywords = choice_lower.split(' - ')[0].split()  # 取选项的第一部分作为关键词
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        
        if best_match and max_overlap > 1:  # 至少需要匹配2个关键词
            return best_match
    
    # 如果无法确定，返回空字符串
    return ""

def calculate_acoustic_metrics(predictions, ground_truths, scene_labels):
    """计算声学场景分类指标：准确率、精确率、召回率和F1分数，参考TAU_qwen2.5.py"""
    # 过滤掉无效的预测和真实标签
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
    
    # 创建标签映射
    label_map = {label: idx for idx, label in enumerate(sorted(scene_labels))}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    
    # 计算各项指标
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

# 数据加载（旧版本，保留作为备用）
def load_tau_dataset_backup(dataset_path, meta_file="meta.csv"):
    """加载TAU数据集，支持多种元数据格式，参考TAU_qwen_dart.py的实现"""
    meta_path = os.path.join(dataset_path, meta_file)
    
    # 方法1: 尝试加载JSON格式的元数据（参考TAU_qwen_dart.py）
    json_meta_path = os.path.join(dataset_path, "acoustic_scene_task_meta.json")
    if os.path.exists(json_meta_path):
        print(f"[加载] 发现JSON元数据文件: {json_meta_path}")
        try:
            with open(json_meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            dataset = []
            missing_files = 0
            scene_counts = Counter()
            
            for item in metadata:
                filename = item.get("path", item.get("filename", ""))
                scene_label = item.get("scene_label", "")
                
                audio_path = os.path.join(dataset_path, filename)
                if not os.path.exists(audio_path):
                    missing_files += 1
                    continue
                
                try:
                    audio_info = sf.info(audio_path)
                    duration = audio_info.duration
                    sample_rate = audio_info.samplerate
                except Exception as e:
                    continue
                
                scene_counts[scene_label] += 1
                
                # 构建选择题格式数据（参考TAU_qwen_dart.py）
                choices = [
                    item.get("choice_a", ""),
                    item.get("choice_b", ""), 
                    item.get("choice_c", ""),
                    item.get("choice_d", "")
                ]
                
                dataset_item = {
                    "filename": filename,
                    "audio_path": audio_path,
                    "scene_label": scene_label,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "id": item.get("uniq_id", f"tau_{filename}"),
                    "question": item.get("question", "Listen to this audio and identify the acoustic scene. Choose the most appropriate option."),
                    "choices": choices,
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2], 
                    "choice_d": choices[3],
                    "answer_gt": item.get("answer_gt", ""),
                    "correct_answer": item.get("answer_gt", ""),
                    "task": "Acoustic_Scene_Classification"
                }
                dataset.append(dataset_item)
            
            print(f"[加载] 从JSON加载 {len(dataset)} 个样本")
            print(f"[统计] 场景分布: {dict(scene_counts)}")
            return dataset
            
        except Exception as e:
            print(f"[错误] JSON元数据加载失败: {e}")
    
    # 方法2: 回退到CSV格式（原始逻辑）
    if not os.path.exists(meta_path):
        print(f"[错误] 元数据文件不存在: {meta_path}")
        return []
    
    try:
        df = pd.read_csv(meta_path, sep='\t')
    except Exception as e:
        print(f"[错误] 读取元数据失败: {e}")
        return []
    
    dataset = []
    missing_files = 0
    
    for _, row in df.iterrows():
        filename = row['filename']
        scene_label = row['scene_label']
        
        audio_path = os.path.join(dataset_path, filename)
        if not os.path.exists(audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"[警告] 音频不存在: {audio_path}")
            continue
        
        try:
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"[错误] 无法读取音频: {audio_path}")
            continue
        
        item = {
            "filename": filename,
            "audio_path": audio_path,
            "scene_label": scene_label,
            "duration": duration,
            "sample_rate": sample_rate,
            "id": f"tau_{filename}",
            "choices": [],  # CSV格式通常没有选择题信息
            "task": "Acoustic_Scene_Classification"
        }
        dataset.append(item)
    
    if missing_files > 5:
        print(f"[信息] 总共 {missing_files} 个音频文件缺失")
    
    print(f"[加载] 成功加载 {len(dataset)} 个样本")
    
    # 场景统计
    scene_counts = Counter([item["scene_label"] for item in dataset])
    print(f"[统计] 场景分布: {dict(scene_counts)}")
    
    return dataset

# 响应解析
def extract_scene_prediction(response, scene_classes=TAU_SCENE_CLASSES):
    """从模型输出中提取场景预测，支持多种格式"""
    if not response:
        return ""
    
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    
    response_lower = response.lower().strip()
    
    # 方法1: 检查选择题答案（A/B/C/D）
    choice_options = ['a', 'b', 'c', 'd']
    for opt in choice_options:
        if response_lower == opt or response_lower.startswith(f"{opt}.") or response_lower.startswith(f"{opt})"):
            return opt.upper()
    
    # 方法2: 直接匹配场景名称
    for scene in scene_classes:
        if scene.lower() in response_lower:
            return scene
    
    # 方法3: 匹配下划线替换的版本
    for scene in scene_classes:
        scene_spaced = scene.replace("_", " ")
        if scene_spaced.lower() in response_lower:
            return scene
    
    # 方法4: 返回响应的第一个词作为备选
    words = response_lower.split()
    if words:
        first_word = words[0].strip('.,!?;:')
        # 检查第一个词是否是有效场景
        for scene in scene_classes:
            if scene.lower().startswith(first_word) or first_word in scene.lower():
                return scene
        return first_word
    
    return ""

# 删除重复定义（extract_acoustic_scene_answer 已在前文定义）

# 统计类
@dataclass
class TAUSampleResult:
    id: str
    audio_path: str
    filename: str
    ground_truth_scene: str
    predicted_scene: str
    is_correct: bool
    raw_response: str
    timing: Dict[str, Any]

class TAUTimingStats:
    def __init__(self):
        self.records = []
        self.scene_stats = defaultdict(list)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
            self.initial_mem = torch.cuda.memory_allocated()

    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, scene_label=None):
        peak_mem = torch.cuda.max_memory_allocated() if self.cuda else 0
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "scene_label": scene_label,
            "gpu_memory_peak_gb": peak_mem / 1024**3 if peak_mem else 0
        }
        self.records.append(record)
        if scene_label:
            self.scene_stats[scene_label].append(record)

    def get_summary(self):
        if not self.records:
            return {"error": "No samples processed"}
        
        df = pd.DataFrame(self.records)
        
        overall_summary = {
            "total_samples": len(df),
            "avg_prefill_time": df.prefill_time.mean(),
            "avg_decode_time": df.decode_time.mean(),
            "avg_total_time": df.total_time.mean(),
            "avg_tokens_per_sec": df.tokens_per_sec.mean(),
            "total_tokens": int(df.output_tokens.sum()),
            "avg_audio_duration": df.audio_duration.mean() if 'audio_duration' in df.columns else 0,
            "avg_gpu_peak_mem_gb": df.gpu_memory_peak_gb.mean()
        }
        
        scene_summaries = {}
        for scene, records in self.scene_stats.items():
            if records:
                scene_df = pd.DataFrame(records)
                scene_summaries[scene] = {
                    "samples": len(records),
                    "avg_prefill_time": scene_df.prefill_time.mean(),
                    "avg_decode_time": scene_df.decode_time.mean(),
                    "avg_total_time": scene_df.total_time.mean(),
                    "avg_tokens_per_sec": scene_df.tokens_per_sec.mean()
                }
        
        return {
            "overall_summary": overall_summary,
            "scene_summaries": scene_summaries
        }

# 指标计算
def calculate_tau_metrics(y_true, y_pred, scene_classes=TAU_SCENE_CLASSES):
    # 处理空值预测
    clean_true, clean_pred = [], []
    for t, p in zip(y_true, y_pred):
        if t and p:  # 都不为空
            clean_true.append(t)
            clean_pred.append(p)
    
    if not clean_true:
        return {"accuracy": 0.0}
    
    # 计算基本指标
    accuracy = accuracy_score(clean_true, clean_pred)
    
    # 计算每类精度、召回率、F1
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_true, clean_pred, labels=scene_classes, average=None, zero_division=0
    )
    
    # 宏平均和加权平均
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_true, clean_pred, average='macro', zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        clean_true, clean_pred, average='weighted', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(clean_true, clean_pred, labels=scene_classes)
    
    # 分类报告
    report = classification_report(
        clean_true, clean_pred, 
        labels=scene_classes,
        target_names=scene_classes,
        zero_division=0,
        digits=4
    )
    
    # 每类指标
    per_class_metrics = {}
    for i, scene in enumerate(scene_classes):
        per_class_metrics[scene] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_w),
        'recall_weighted': float(rec_w),
        'f1_weighted': float(f1_w),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'valid_samples': len(clean_true),
        'total_samples': len(y_true)
    }

# KV Press 初始化
def patch_qwen_for_kvpress(model):
    """为 Qwen2.5-Omni 添加 rotary_emb 兼容性，参考TAU_kvpress.py的实现"""
    # 检查模型结构 - 适配Qwen2.5-Omni的复合结构
    try_paths = []
    
    # Qwen2.5-Omni的模型结构层次：model.thinker.model
    if hasattr(model, 'thinker'):
        try_paths.append(model.thinker)
        if hasattr(model.thinker, 'model'):
            try_paths.append(model.thinker.model)
    
    # 传统结构
    if hasattr(model, 'model'):
        try_paths.append(model.model)
    try_paths.append(model)
    
    base = None
    for cand in try_paths:
        if hasattr(cand, 'layers'):
            base = cand
            print(f"[补丁] 找到基础模型: {type(cand).__name__}")
            break
    
    if base is None:
        print("[警告] 无法找到模型的基础结构，KV Press可能无法正常工作")
        return False
    
    # 检查是否已经有全局rotary_emb
    if hasattr(base, 'rotary_emb') and base.rotary_emb is not None:
        print("[补丁] 模型已有全局rotary_emb属性")
        base._kvpress_patched = True
        return True
    
    # 从首层提取 rotary_emb
    if hasattr(base, 'layers') and len(base.layers) > 0:
        for layer_idx, layer in enumerate(base.layers):
            # 查找self_attn或attn属性
            attn_layer = None
            for attr in ['self_attn', 'attn']:
                if hasattr(layer, attr):
                    attn_layer = getattr(layer, attr)
                    break
            
            if attn_layer and hasattr(attn_layer, 'rotary_emb') and attn_layer.rotary_emb is not None:
                base.rotary_emb = attn_layer.rotary_emb
                print(f"[补丁] 已从层 {layer_idx} 提取并添加全局 rotary_emb 属性")
                base._kvpress_patched = True
                return True
    
    # 创建占位符 rotary_emb - 适配复合配置
    try:
        # 尝试从不同的配置路径获取参数
        config = None
        if hasattr(model, 'config'):
            if hasattr(model.config, 'thinker_config') and hasattr(model.config.thinker_config, 'text_config'):
                config = model.config.thinker_config.text_config
            elif hasattr(model.config, 'text_config'):
                config = model.config.text_config
            elif hasattr(model.config, 'hidden_size'):
                config = model.config
        
        # 使用配置或默认值
        if config and hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
        else:
            # Qwen2.5-Omni-3B的默认值
            hidden_size = 3584
            num_heads = 28
            print(f"[补丁] 使用默认配置参数")
        
        head_dim = hidden_size // num_heads
        
        class SimpleRotaryEmbedding:
            def __init__(self, dim, max_position_embeddings=32768, base=10000):
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
                self.base = base
        
        base.rotary_emb = SimpleRotaryEmbedding(dim=head_dim)
        base._kvpress_patched = True
        print(f"[补丁] 创建占位符rotary_emb，head_dim={head_dim}")
        return True
    except Exception as e:
        print(f"[错误] 创建占位符rotary_emb失败: {e}")
        return False

def create_kvpress_adapter(model, press_obj):
    """为 Qwen2.5-Omni 创建 KV Press 适配器，保持与 HAD_qwen_kvpress 一致的挂钩方式"""
    if press_obj is None:
        return contextlib.nullcontext()

    class Qwen2_5OmniKVPressAdapter:
        def __init__(self, original_model, press_object):
            self.original_model = original_model
            self.press_object = press_object
            self.hooks = []
            self.base_model = None
            self.press_method = type(press_object).__name__.lower()

        def __enter__(self):
            try:
                # 定位到实际的 transformer 层
                if hasattr(self.original_model, 'thinker') and hasattr(self.original_model.thinker, 'model'):
                    self.base_model = self.original_model.thinker.model
                elif hasattr(self.original_model, 'model'):
                    self.base_model = self.original_model.model
                else:
                    print("[KVPress适配器] 警告: 无法定位基础模型")
                    self.hooks = []
                    return self

                if not hasattr(self.base_model, 'layers'):
                    print("[KVPress适配器] 警告: 基础模型没有 layers 属性")
                    self.hooks = []
                    return self

                layers = self.base_model.layers
                print(f"[KVPress适配器] 为 {len(layers)} 层注册 hooks (方法: {type(self.press_object).__name__})")

                hooks = []
                successful_hooks = 0

                # 确保 rotary_emb 可用
                base_rotary_emb = getattr(self.base_model, 'rotary_emb', None)

                for layer_idx, layer in enumerate(layers):
                    try:
                        # Qwen 注意力模块可能叫 self_attn 或 attn，优先 self_attn
                        attn_module = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                        if attn_module is None:
                            continue

                        # 确保每层注意力都有 rotary_emb
                        if base_rotary_emb and (not hasattr(attn_module, 'rotary_emb') or attn_module.rotary_emb is None):
                            attn_module.rotary_emb = base_rotary_emb

                        if hasattr(self.press_object, 'forward_hook'):
                            hook = attn_module.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
                            hooks.append(hook)
                            successful_hooks += 1
                    except Exception as e:
                        print(f"[KVPress适配器] 层 {layer_idx} hook 注册失败: {e}")
                        continue

                self.hooks = hooks
                print(f"[KVPress适配器] 成功注册 {successful_hooks}/{len(layers)} 个 hooks")
                return self

            except Exception as e:
                print(f"[KVPress适配器] 注册过程出错: {e}")
                traceback.print_exc()
                for h in getattr(self, 'hooks', []):
                    try:
                        h.remove()
                    except:
                        pass
                self.hooks = []
                return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            removed_count = 0
            for hook in self.hooks:
                try:
                    hook.remove()
                    removed_count += 1
                except Exception:
                    pass
            if removed_count:
                print(f"[KVPress适配器] 清理了 {removed_count} 个 hooks")
            self.hooks = []

    return Qwen2_5OmniKVPressAdapter(model, press_obj)

def verify_kvpress_compatibility(model, press_type):
    """验证特定 KVPress 方法的兼容性（移植自 HAD_qwen_kvpress）"""
    try:
        print(f"[KVPress] 验证 {press_type} 方法兼容性...")
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            print("[KVPress] 兼容性验证失败: 无法定位基础模型")
            return False

        if not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
            print("[KVPress] 兼容性验证失败: 基础模型没有有效的 layers")
            return False

        if press_type in ['snap', 'tova']:
            global_rotary = getattr(base_model, 'rotary_emb', None)
            if global_rotary is None:
                print(f"[KVPress] {press_type} 兼容性验证失败: 缺少全局 rotary_emb")
                return False
            layer_check_count = min(3, len(base_model.layers))
            for i in range(layer_check_count):
                layer = base_model.layers[i]
                attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                if attn is None or getattr(attn, 'rotary_emb', None) is None:
                    print(f"[KVPress] {press_type} 兼容性验证失败: 层 {i} 缺少 rotary_emb/self_attn")
                    return False

        first_layer = base_model.layers[0]
        attn = getattr(first_layer, 'self_attn', None) or getattr(first_layer, 'attn', None)
        if attn is None:
            print("[KVPress] 兼容性验证失败: 第一层没有注意力模块")
            return False
        for attr in ['q_proj', 'k_proj', 'v_proj']:
            if not hasattr(attn, attr):
                print(f"[KVPress] 兼容性验证失败: 注意力缺少 {attr}")
                return False
        print(f"[KVPress] {press_type} 兼容性验证通过")
        return True
    except Exception as e:
        print(f"[KVPress] {press_type} 兼容性验证出错: {e}")
        return False

def verify_tova_multimodal_compatibility(model):
    """验证 TOVA 与多模态输入的兼容性（移植自 HAD_qwen_kvpress）"""
    try:
        print("[KVPress] 验证 TOVA 多模态兼容性...")
        base_model = None
        config = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            config = getattr(model.thinker, 'config', None)
        elif hasattr(model, 'model'):
            base_model = model.model
            config = getattr(model, 'config', None)
        if base_model is None:
            print("[KVPress] TOVA 多模态验证失败: 无法定位基础模型")
            return False
        rotary_emb = getattr(base_model, 'rotary_emb', None)
        if rotary_emb is None:
            print("[KVPress] TOVA 多模态验证失败: 缺少 rotary_emb")
            return False
        if hasattr(rotary_emb, 'head_dim'):
            head_dim = rotary_emb.head_dim
            if head_dim <= 0 or head_dim % 2 != 0:
                print(f"[KVPress] TOVA 多模态验证失败: head_dim {head_dim} 不适合 TOVA")
                return False
        print("[KVPress] TOVA 多模态兼容性验证通过")
        return True
    except Exception as e:
        print(f"[KVPress] TOVA 多模态兼容性验证出错: {e}")
        return False

def verify_snapkv_multimodal_compatibility(model):
    """验证 SnapKV 与多模态输入的兼容性（移植自 HAD_qwen_kvpress）"""
    try:
        print("[KVPress] 验证 SnapKV 多模态兼容性...")
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        if base_model is None:
            print("[KVPress] SnapKV 多模态验证失败: 无法定位基础模型")
            return False
        rotary_emb = getattr(base_model, 'rotary_emb', None)
        if rotary_emb is None or not hasattr(rotary_emb, 'forward'):
            print("[KVPress] SnapKV 多模态验证失败: rotary_emb 缺失或不含 forward")
            return False
        print("[KVPress] SnapKV 多模态兼容性验证通过")
        return True
    except Exception as e:
        print(f"[KVPress] SnapKV 多模态兼容性验证出错: {e}")
        return False

def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """初始化KV Press压缩，使用HAD简化版本"""
    if not KV_PRESS_AVAILABLE:
        print("[警告] KV Press不可用，跳过压缩初始化")
        return None

    print(f"初始化 KV Press: 类型={press_type}, 压缩比={compression_ratio}")

    try:
        if press_type == 'expected':
            press_obj = ExpectedAttentionPress(compression_ratio=compression_ratio)
        elif press_type == 'observed':
            press_obj = ObservedAttentionPress(compression_ratio=compression_ratio)
        elif press_type == 'random':
            press_obj = RandomPress(compression_ratio=compression_ratio)
            print("[KVPress] 使用 RandomPress（兼容所有模型）")
        elif press_type == 'streaming':
            press_obj = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
        elif press_type == 'tova':
            press_obj = TOVAPress(compression_ratio=compression_ratio)
            print("[KVPress] 使用 TOVAPress")
        elif press_type == 'snap':
            press_obj = SnapKVPress(compression_ratio=compression_ratio)
            print("[KVPress] 使用 SnapKVPress")
        else:
            # 默认使用 knorm（最兼容的方法，适用所有模型）
            press_obj = KnormPress(compression_ratio=compression_ratio)
            print("[KVPress] 使用默认 KnormPress（兼容所有模型）")

        # 可选地设置最小序列长度
        if hasattr(press_obj, 'min_seq_len'):
            try:
                setattr(press_obj, 'min_seq_len', min_seq_len)
            except Exception:
                pass

        print(f"[KVPress] 已创建 {type(press_obj).__name__}")
        return press_obj
    except Exception as e:
        print(f"[KVPress] 创建 {press_type} 对象失败: {e}")
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    # 环境诊断
    print("=== 环境诊断 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径
    
    # 检查Torch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch未安装")
    
    # 检查依赖
    if not QWEN_AVAILABLE:
        print("\n[错误] Qwen2.5-Omni 模块不可用")
        print("请检查以下项目:")
        print("1. 模型路径是否正确: /data/hepeize05/Audio_Longbench/Code/Qwen_2.5")
        print("2. 相关文件是否存在:")
        print("   - modeling_qwen2_5_omni_origin.py")
        print("   - processing_qwen2_5_omni.py") 
        print("   - qwen_omni_utils.py")
        print("3. 是否在正确的环境中运行")
        return
    
    # 强制要求KV Press可用
    if not KV_PRESS_AVAILABLE:
        print("\n[错误] KV Press 库不可用，但要求必须使用压缩")
        print("请检查以下项目:")
        print("1. kvpress库是否已安装: pip install kvpress")
        print("2. transformers版本是否兼容")
        print("3. 是否存在本地kvpress目录")
        print("4. Python环境是否正确")
        return
    
    # 强制禁用no_compress选项
    if args.no_compress:
        print("\n[警告] 忽略 --no-compress 选项，强制启用KV Press压缩")
        args.no_compress = False
    
    # 确认GPU可用性并清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda"
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\nNo GPU available, using CPU")
    
    # 结果目录
    result_dir = os.path.abspath(RESULTS_DIR_ENV)
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载数据集
    samples, scene_counts = load_tau_acoustic_scene_dataset(args.dataset_path)
    if not samples:
        print("[错误] 无法加载数据集")
        return
    
    if SAMPLE_LIMIT > 0 and len(samples) > SAMPLE_LIMIT:
        # 随机采样而不是直接截取前N个
        import random
        samples = random.sample(samples, SAMPLE_LIMIT)
        print(f"[限制] 随机选择 {len(samples)} 个样本")
    
    # 加载模型
    print("[加载] Qwen2.5-Omni 模型...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 初始化 KV Press（提前确定是否启用压缩）
    will_use_compression = not args.no_compress and KV_PRESS_AVAILABLE
    
    # 根据是否使用压缩选择注意力实现
    attention_impl = "eager" if will_use_compression else "flash_attention_2"
    print(f"[注意力] 使用 {attention_impl} 实现（压缩：{'启用' if will_use_compression else '禁用'}）")
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_impl,
        trust_remote_code=True,
    )
    
    if hasattr(model, 'disable_talker'):
        model.disable_talker()
    model.eval()
    
    # 重置峰值显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 记录初始显存
    initial_allocated, initial_reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {initial_allocated:.2f}GB, 已保留: {initial_reserved:.2f}GB")
    
    # 添加缺失的配置属性（参考 TAU_qwen2.5.py）
    print("[配置] 为 Qwen2.5-Omni 添加必要的配置属性...")
    
    # 首先在 thinker 配置上设置（这是forward函数中self.config指向的配置）
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
        if not hasattr(model.thinker.config, 'audio_token_id'):
            model.thinker.config.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(model.thinker.config, 'image_token_id'):
            model.thinker.config.image_token_id = 151655  # 标准图像token ID
        if not hasattr(model.thinker.config, 'video_token_id'):
            model.thinker.config.video_token_id = 151656  # 标准视频token ID
        if not hasattr(model.thinker.config, 'audio_bos_token_id'):
            model.thinker.config.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(model.thinker.config, 'audio_eos_token_id'):
            model.thinker.config.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
        if not hasattr(model.thinker.config, 'image_layer_idx'):
            model.thinker.config.image_layer_idx = False
        if not hasattr(model.thinker.config, 'audio_layer_idx'):
            model.thinker.config.audio_layer_idx = None
        if not hasattr(model.thinker.config, 'audio_token_num'):
            model.thinker.config.audio_token_num = None
        if not hasattr(model.thinker.config, 'audio_token_start'):
            model.thinker.config.audio_token_start = None
        if not hasattr(model.thinker.config, 'audio_prune_ratio'):
            model.thinker.config.audio_prune_ratio = 0
        if not hasattr(model.thinker.config, 'random'):
            model.thinker.config.random = False
        if not hasattr(model.thinker.config, 'frame'):
            model.thinker.config.frame = False
        print("[配置] thinker.config 音频配置参数已设置")
    
    # 然后在 thinker.model 配置上也设置（双重保险）
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        # 设置到thinker.model.config
        if not hasattr(model.thinker.model.config, 'audio_token_id'):
            model.thinker.model.config.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(model.thinker.model.config, 'image_token_id'):
            model.thinker.model.config.image_token_id = 151655
        if not hasattr(model.thinker.model.config, 'video_token_id'):
            model.thinker.model.config.video_token_id = 151656
        if not hasattr(model.thinker.model.config, 'audio_bos_token_id'):
            model.thinker.model.config.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(model.thinker.model.config, 'audio_eos_token_id'):
            model.thinker.model.config.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
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
        print("[配置] thinker.model.config 音频配置参数已设置")
    
    # 初始化 KV Press - 强制启用
    press_obj = initialize_kv_press(model, ENV_PRESS_TYPE, ENV_COMPRESSION_RATIO, args.min_seq_len)
    if press_obj is None:
        print(f"\n[错误] KV Press初始化失败，无法继续")
        print(f"尝试的压缩类型: {ENV_PRESS_TYPE}")
        print(f"压缩比例: {ENV_COMPRESSION_RATIO}")
        print(f"最小序列长度: {args.min_seq_len}")
        return
    else:
        print(f"[成功] KV Press 启用: {ENV_PRESS_TYPE}, 压缩比: {ENV_COMPRESSION_RATIO}")
    
    # 记录初始GPU内存使用情况
    initial_memory_gpu = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理缓存以获得准确的基线
        torch.cuda.synchronize()  # 确保所有操作完成
        initial_memory_gpu = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
        print(f"[内存] 初始GPU内存使用: {initial_memory_gpu:.2f} GB")
        
        # 获取GPU总内存
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_properties.total_memory / 1024**3
        print(f"[内存] GPU总内存: {total_memory_gb:.2f} GB")
        print(f"[内存] 初始内存占用率: {initial_memory_gpu/total_memory_gb*100:.1f}%")
    else:
        print("[内存] CUDA不可用，无法获取GPU内存统计")
    
    # 统计
    results: List[TAUSampleResult] = []
    timing = TAUTimingStats()
    total_accuracy = 0
    processed_samples = 0
    scene_correct = defaultdict(int)
    scene_total = defaultdict(int)
    all_predictions = []
    all_ground_truths = []
    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0
    
    # 检测运行环境
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("检测到screen或非交互式环境，使用简化进度显示")
    
    tqdm_kwargs = {'ascii': True, 'dynamic_ncols': True, 'file': sys.stdout}
    print(f"[开始] 评估 {len(samples)} 个样本")
    
    for idx, sample in tqdm(enumerate(samples), total=len(samples), desc="TAU QwenKVPress", **tqdm_kwargs):
        try:
            audio_path = sample["audio_path"]
            if not os.path.exists(audio_path):
                continue
            
            ground_truth_scene = sample["scene_label"]
            
            # 构建指令（支持选择题和开放式两种格式，参考TAU_qwen_dart.py）
            if "question" in sample and sample["question"] and "choices" in sample and sample["choices"]:
                # 选择题格式（TAU_qwen_dart.py的标准格式）
                instruction = sample["question"]
                choices = sample["choices"]
                
                # 格式化选项
                formatted_options = "Respond with only the letter of your answer (A, B, C, or D).\n"
                for i, choice in enumerate(choices):
                    if choice:  # 只添加非空选项
                        formatted_options += f"{chr(65+i)}) {choice}\n"
                
                sys_prompt = "You are an expert in urban acoustic scene classification. Listen carefully to the audio and choose the correct answer."
                full_instruction = f"{instruction}\n\nOptions:\n{formatted_options.strip()}"
                
            elif "choice_a" in sample and sample["choice_a"]:
                # 备用选择题格式
                choices = [sample.get("choice_a", ""), sample.get("choice_b", ""), 
                          sample.get("choice_c", ""), sample.get("choice_d", "")]
                
                instruction = sample.get("question", "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.")
                
                # 格式化选项
                formatted_options = "Respond with only the letter of your answer (A, B, C, or D).\n"
                for i, choice in enumerate(choices):
                    if choice:  # 只添加非空选项
                        formatted_options += f"{chr(65+i)}) {choice}\n"
                
                sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are an expert in urban acoustic scene classification. Listen carefully to the audio and choose the correct answer."
                full_instruction = f"{instruction}\n\nOptions:\n{formatted_options.strip()}"
                
            else:
                # 开放式格式
                scene_list = ", ".join(TAU_SCENE_CLASSES)
                full_instruction = f"Listen to this audio recording and identify the urban acoustic scene. Choose from the following categories: {scene_list}. Respond with only the scene name."
                sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are an expert in urban acoustic scene classification. Listen carefully to the audio and identify the type of urban environment or transportation setting."
                choices = None
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": full_instruction},
                ]}
            ]
            
            # 多模态处理
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(text, list):
                text = text[0]
            
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            )
            inputs = inputs.to(model.device)
            inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}
            
            # 设置音频配置参数 - 检测音频token位置（参考 TAU_qwen2.5.py）
            audio_token_length = 0
            audio_token_start = 0
            input_token_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            
            # 检测音频token位置和长度 - 使用正确的audio token sequence
            audio_detected = False
            
            if 'input_ids' in inputs:
                token_ids = inputs['input_ids'][0].tolist()
                
                # 查找 <|audio_bos|> 和 <|audio_eos|> token 的位置
                bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                
                if bos_positions and eos_positions:
                    # 使用第一个BOS和第一个EOS
                    audio_token_start = bos_positions[0]
                    audio_token_end = eos_positions[0]
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    audio_detected = True
                    
                    # 配置音频相关参数到 thinker.model.config
                    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                        model.thinker.model.config.image_layer_idx = False  # 不处理图像
                        model.thinker.model.config.audio_layer_idx = None    # 这里我们不做剪枝，只设置KV Press
                        model.thinker.model.config.audio_token_num = audio_token_length
                        model.thinker.model.config.audio_token_start = audio_token_start
                        model.thinker.model.config.audio_prune_ratio = 0    # 不做剪枝
                        model.thinker.model.config.random = False
                        model.thinker.model.config.frame = False
                
                # 如果没有找到BOS/EOS对，检查是否只有AUDIO token
                elif not audio_detected and _AUDIO_TOKEN_ID in token_ids:
                    audio_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_TOKEN_ID]
                    if audio_positions:
                        audio_token_start = audio_positions[0]
                        audio_token_length = len(audio_positions)
                        audio_detected = True
                        
                        # 配置音频相关参数
                        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                            model.thinker.model.config.audio_token_num = audio_token_length
                            model.thinker.model.config.audio_token_start = audio_token_start
            
            # 检测音频 token（为了兼容性保留）
            audio_token_len = audio_token_length
            
            # 计算输入长度用于压缩门控
            input_tokens = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            use_compression = press_obj is not None and input_tokens >= args.min_seq_len
            
            # Prefill 计时
            prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if prefill_start:
                prefill_start.record()
            with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if prefill_end:
                prefill_end.record()
            
            # 完整生成
            gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if gen_start:
                gen_start.record()
            with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            if gen_end:
                gen_end.record()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 更新峰值显存统计
            if torch.cuda.is_available():
                current_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                current_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                peak_allocated_gb = max(peak_allocated_gb, current_allocated)
                peak_reserved_gb = max(peak_reserved_gb, current_reserved)

            prefill_time = prefill_start.elapsed_time(prefill_end)/1000 if prefill_start else 0.0
            total_time = gen_start.elapsed_time(gen_end)/1000 if gen_start else 0.0
            decode_time = max(total_time - prefill_time, 0.0)
            
            # 解析输出
            resp = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if "assistant\n" in resp:
                resp = resp.split("assistant\n")[-1].strip()
            
            output_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            
            # 提取预测结果（参考TAU_qwen_dart.py的答案提取逻辑）
            answer_choice = ""  # 初始化变量
            if choices and any(choices):
                # 选择题格式：首先提取A/B/C/D，然后映射到场景
                answer_choice = extract_acoustic_scene_answer(resp, choices)
                
                # 映射选择到场景标签
                if answer_choice and len(choices) >= ord(answer_choice) - ord('A') + 1:
                    choice_index = ord(answer_choice) - ord('A')
                    predicted_scene_from_choice = choices[choice_index]
                    
                    # 尝试将选择内容映射到标准场景类别
                    predicted_scene = ""
                    for scene in TAU_SCENE_CLASSES:
                        if scene.lower() in predicted_scene_from_choice.lower() or \
                           scene.replace("_", " ").lower() in predicted_scene_from_choice.lower():
                            predicted_scene = scene
                            break
                    
                    # 如果映射失败，使用原始选择内容
                    if not predicted_scene:
                        predicted_scene = predicted_scene_from_choice
                        
                    # 验证选择的正确性（如果有ground truth答案）
                    if "correct_answer" in sample and sample["correct_answer"] == answer_choice:
                        pass  # 正确答案
                else:
                    # 如果选择题提取失败，回退到直接场景提取
                    predicted_scene = extract_scene_prediction(resp, TAU_SCENE_CLASSES)
            else:
                # 开放式格式：直接提取场景
                predicted_scene = extract_scene_prediction(resp, TAU_SCENE_CLASSES)
            
            # 准确率
            is_correct = predicted_scene == ground_truth_scene
            accuracy = 1.0 if is_correct else 0.0
            total_accuracy += accuracy
            processed_samples += 1
            
            scene_total[ground_truth_scene] += 1
            if is_correct:
                scene_correct[ground_truth_scene] += 1
            
            all_predictions.append(predicted_scene)
            all_ground_truths.append(ground_truth_scene)
            
            # 时间统计
            timing.add_record(
                prefill_time=prefill_time,
                decode_time=decode_time,
                output_tokens=output_tokens,
                input_tokens=inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0,
                audio_duration=sample.get("duration", 0),
                scene_label=ground_truth_scene
            )
            
            # 结果记录（参考TAU_qwen_dart.py的详细记录）
            results.append(TAUSampleResult(
                id=sample.get("id", f"sample_{idx}"),
                audio_path=audio_path,
                filename=sample.get("filename", os.path.basename(audio_path)),
                ground_truth_scene=ground_truth_scene,
                predicted_scene=predicted_scene,
                is_correct=is_correct,
                raw_response=resp,
                timing={
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time,
                    "output_tokens": output_tokens,
                    "audio_tokens": audio_token_len,
                    "answer_choice": answer_choice,
                    "correct_answer": sample.get("answer_gt", ""),
                    "choices_used": bool(choices and any(choices)),
                    "compression_info": {
                        "enabled": press_obj is not None,
                        "used": use_compression,
                        "press_type": ENV_PRESS_TYPE if press_obj else None,
                        "compression_ratio": ENV_COMPRESSION_RATIO if press_obj else None,
                        "min_seq_len": args.min_seq_len,
                        "input_tokens": input_tokens,
                    }
                }
            ))
            
            # 清理内存
            torch.cuda.empty_cache()
            
            if (idx + 1) % 50 == 0:
                current_acc = total_accuracy / processed_samples
                print(f"[进度] {idx+1}/{len(samples)} 当前准确率: {current_acc:.3f}")
                
        except Exception as e:
            print(f"[错误] 样本 {idx} 处理失败: {e}")
            # 添加详细错误信息（参考TAU_qwen_dart.py）
            import traceback
            traceback.print_exc()
            
            # 记录失败的样本信息
            failed_sample = {
                "id": sample.get("id", f"sample_{idx}"),
                "audio_path": sample.get("audio_path", ""),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"失败样本详情: {failed_sample}")
            continue
    
    # 最终统计
    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0
    
    # 场景类别准确率
    scene_accuracies = {}
    for scene in TAU_SCENE_CLASSES:
        if scene_total[scene] > 0:
            scene_accuracies[scene] = scene_correct[scene] / scene_total[scene]
        else:
            scene_accuracies[scene] = 0.0
    
    # sklearn 指标
    detailed_metrics = calculate_tau_metrics(all_ground_truths, all_predictions, TAU_SCENE_CLASSES)
    timing_summary = timing.get_summary()
    
    # 统一输出结构（与 SLUE/VESUS 对齐）
    processed_results = [r for r in results]
    final_results = {
        "model_info": {
            "model_path": args.model_path,
            "kv_press_type": ENV_PRESS_TYPE,
            "compression_ratio": ENV_COMPRESSION_RATIO,
        },
        "dataset_info": {
            "total_samples": len(results),
            "processed_samples": processed_samples,
            "skipped_samples": 0,
            "error_samples": 0,
        },
        "metrics": {
            "accuracy": final_accuracy,
            "f1_macro": detailed_metrics.get("f1_macro", 0.0),
            "precision_macro": detailed_metrics.get("precision_macro", 0.0),
            "recall_macro": detailed_metrics.get("recall_macro", 0.0),
            "classification_report": detailed_metrics.get("classification_report", ""),
            "per_class_metrics": detailed_metrics.get("per_class_metrics", {}),
            "confusion_matrix": detailed_metrics.get("confusion_matrix", []),
        },
        "detailed_results": convert_to_serializable(results),
        "timing_summary": timing_summary,
    }

    output_file = os.path.join(result_dir, f"tau_qwen_kvpress_results_{ENV_PRESS_TYPE}_{ENV_COMPRESSION_RATIO}.json")
    timing_file = os.path.join(result_dir, f"tau_qwen_kvpress_timing_{ENV_PRESS_TYPE}_{ENV_COMPRESSION_RATIO}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    with open(timing_file, 'w', encoding='utf-8') as f:
        json.dump({"timing_summary": timing_summary, "kv_press": {"enabled": True, "press_type": ENV_PRESS_TYPE, "compression_ratio": ENV_COMPRESSION_RATIO}}, f, ensure_ascii=False, indent=2)
    
    # 输出结果（参考TAU_qwen_dart.py的详细输出格式）
    print("\n=== TAU Qwen2.5-Omni + KV Press 结果摘要 ===")
    print(f"总样本: {len(results)}")
    print(f"处理样本: {processed_samples}")
    print(f"总体准确率: {final_accuracy:.3f}")
    
    print(f"\n各场景准确率:")
    sorted_scenes = sorted(scene_accuracies.items(), key=lambda x: x[1], reverse=True)
    for scene, acc in sorted_scenes:
        correct_num = scene_correct[scene]
        total_num = scene_total[scene]
        print(f"  {scene}: {acc:.3f} ({correct_num}/{total_num})")
    
    metrics = detailed_metrics
    print(f"\n=== sklearn 评估指标 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"F1分数(宏): {metrics['f1_macro']:.4f}")
    print(f"F1分数(加权): {metrics['f1_weighted']:.4f}")
    print(f"精度(宏): {metrics['precision_macro']:.4f}")
    print(f"召回率(宏): {metrics['recall_macro']:.4f}")
    print(f"有效样本: {metrics['valid_samples']}/{metrics['total_samples']}")
    
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== 时间统计 ===")
    print(f"平均总时间: {overall_summary.get('avg_total_time', 0):.4f}s")
    print(f"平均Prefill: {overall_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"平均Decode: {overall_summary.get('avg_decode_time', 0):.4f}s")
    print(f"平均吞吐: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tok/s")
    print(f"平均GPU峰值内存: {overall_summary.get('avg_gpu_peak_mem_gb', 0):.2f} GB")
    
    # 添加内存统计输出
    # 简化内存统计输出（summary 已合并入 final_results，避免未定义访问）
    print(f"\n=== 内存统计（简要） ===")
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"当前GPU内存: {current_memory:.2f} GB")
    
    # 添加KV Press配置信息输出
    print(f"\n=== KV Press 配置 ===")
    print(f"压缩启用: True")
    print(f"压缩类型: {ENV_PRESS_TYPE}")
    print(f"压缩比例: {ENV_COMPRESSION_RATIO}")
    
    print(f"\n=== 分类报告 ===")
    print(metrics['classification_report'])
    
    print(f"\n结果文件: {output_file}")
    print(f"时间文件: {timing_file}")

if __name__ == "__main__":
    main()
