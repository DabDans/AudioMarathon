#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VESUS情感识别模型KV Press评测脚本
用于评测模型在VESUS情感识别任务上的性能，使用KV Press压缩技术
"""

import os
import sys
import warnings
import torch
import time
import json
import random
import gc
import re
import contextlib
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.io import wavfile
from scipy import signal

# KV Press 相关导入
from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
)
from kvpress.pipeline import KVPressAudioTranscriptionPipeline
import torch.nn.functional as F

# 禁用警告
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
logging.set_verbosity_error()

# 音频特殊token ID
_AUDIO_SPECIAL_TOKEN_ID = 200011

# KV Press 配置
kv_press_config = {
    "compression_ratio": float(os.environ.get("COMPRESSION_RATIO", 0.5)),
    "head_dims": None,
    "num_attention_heads": None,
    "press_type": os.environ.get("PRESS_TYPE", "knorm"),
    "return_indices": True,
    "min_seq_len": 128,
    "model_kwargs": {
        "attn_implementation": "sdpa",
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False
    }
}

def patch_phi4mm_for_kvpress(model):
    """为 Phi4MM 模型添加 rotary_emb 属性以便与 KV Press 兼容"""
    if not hasattr(model, "model") or not hasattr(model.model, "layers") or len(model.model.layers) == 0:
        print("警告: 模型结构不符合预期，无法应用KV Press补丁")
        return False
    
    first_layer = model.model.layers[0]
    if not hasattr(first_layer, "self_attn"):
        print("警告: 注意力层结构不符合预期，无法应用KV Press补丁")
        return False
    
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        print("模型已有全局rotary_emb属性，无需添加补丁")
        return True
    
    rotary_found = False
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "rotary_emb") and layer.self_attn.rotary_emb is not None:
            model.model.rotary_emb = layer.self_attn.rotary_emb
            print(f"已从层 {layer_idx} 提取并添加全局 rotary_emb 属性")
            rotary_found = True
            break
    
    if not rotary_found:
        try:
            config = model.config
            head_dim = config.hidden_size // config.num_attention_heads
            
            # 创建简单的RotaryEmbedding占位符
            class SimpleRotaryEmbedding:
                def __init__(self, dim, max_position_embeddings=32768, base=10000):
                    self.dim = dim
                    self.max_position_embeddings = max_position_embeddings
                    self.base = base
            
            model.model.rotary_emb = SimpleRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, 'rope_theta', 10000)
            )
            
            print("已手动创建并添加全局 rotary_emb 属性")
            rotary_found = True
        except Exception as e:
            print(f"创建rotary_emb时出错: {str(e)}")
            return False
    
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model._kvpress_patched = True
        return True
    else:
        print("补丁应用失败，无法找到或创建适用的rotary_emb")
        return False

def initialize_kv_press_simplified(model, config):
    """初始化 KV Press 压缩组件，基于配置选择合适的压缩类型
    
    Args:
        model: 需要应用 KV Press 的模型
        config: 包含压缩配置的字典
    
    Returns:
        KV Press 实例，如果初始化失败则返回 None
    """
    # 获取压缩参数
    press_type = config.get("press_type", "knorm").lower()
    compression_ratio = config.get("compression_ratio", 0.5)
    window_size = config.get("window_size", 1024)  # StreamingLLM 论文中的默认值

    print(f"正在初始化 KV Press 类型: {press_type}, 压缩比: {compression_ratio}")

    # 检查模型是否成功应用了 KV Press 补丁
    if not getattr(model.model, "_kvpress_patched", False):
        print("警告: KV Press 补丁未成功应用或未运行，KV Press 可能无法正常工作")

    # 根据配置的类型选择 Press 实现
    if press_type == "knorm":
        print("使用 KnormPress (基于 Key-norm 的压缩方法，最稳定)")
        return KnormPress(compression_ratio=compression_ratio)
    
    elif press_type == "expected":
        print("使用 Phi4MM适配版 ExpectedAttentionPress (基于 RoPE 的注意力压缩)")
        return ExpectedAttentionPress(compression_ratio=compression_ratio)

    elif press_type == "random":
        print("使用 RandomPress (随机选择 tokens)")
        return RandomPress(compression_ratio=compression_ratio)
    
    elif press_type == "observed":
        print("使用 ObservedAttentionPress (基于观察到的注意力分数)")
        return ObservedAttentionPress(compression_ratio=compression_ratio)
    
    elif press_type == "tova":
        print("使用 TOVAPress (基于时间顺序的注意力值分析)")
        
        # 创建改进的 TOVAPress 类来修复 position_embeddings 错误
        class CustomTOVAPress(TOVAPress):
            def __init__(self, compression_ratio=0.0):
                super().__init__(compression_ratio=compression_ratio)
                self.output_attentions = False
                
            def compute_window_attention(self, module, hidden_states, keys):
                """计算最后一个token相对于之前序列的注意力分数"""
                # 确保keys有效
                if not isinstance(keys, torch.Tensor) or keys.dim() < 3:
                    # 返回一个安全的默认值
                    return torch.ones((1, 1, 1), device=keys.device if hasattr(keys, 'device') else None)
                    
                seq_length = keys.size(-2)
                
                # 如果序列长度为0，返回空的注意力分数
                if seq_length <= 0:
                    return torch.zeros((keys.size(0), 1, 0), device=keys.device)
                    
                # 获取最后一个位置的位置编码
                if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
                    # 使用ROT-EMB处理
                    positions = torch.arange(seq_length, device=keys.device)
                    window_positions = positions.float()
                else:
                    # 退化为线性衰减
                    window_positions = torch.arange(seq_length, device=keys.device).float()
                    
                # 时间衰减因子 (距离越近权重越大)
                # 使用简单指数衰减替代复杂的注意力计算
                decay = torch.exp(-0.1 * torch.abs(
                    window_positions.unsqueeze(0) - window_positions[-1:]
                ))
                
                # 对衰减因子归一化形成注意力分数
                window_attention = torch.nn.functional.softmax(decay, dim=-1)
                
                # 添加批量维度和头维度
                batch_size = keys.size(0)
                # 首先确保window_attention至少是2D (批量维度和序列维度)
                if window_attention.dim() == 1:
                    window_attention = window_attention.unsqueeze(0)
                    
                # 然后添加头维度
                if window_attention.dim() == 2:
                    window_attention = window_attention.unsqueeze(1)
                    
                # 现在安全地使用repeat而不是expand
                if window_attention.size(0) == 1 and batch_size > 1:
                    window_attention = window_attention.repeat(batch_size, 1, 1)
                    
                return window_attention
                    
            def score(self, module, hidden_states, keys, values, attentions, kwargs):
                """基于时间顺序的注意力评分，增强版"""
                try:
                    # 验证keys张量有效性
                    if not isinstance(keys, torch.Tensor) or keys.dim() < 3:
                        print(f"无效的keys，维度：{keys.dim() if isinstance(keys, torch.Tensor) else type(keys)}")
                        return torch.ones((1, 1, 1), device=module.device if hasattr(module, 'device') else None)
                        
                    # 获取注意力分数
                    if attentions is not None:
                        # 如果模型输出了注意力矩阵，使用最后一个token对之前序列的注意力
                        if attentions.dim() >= 4:
                            attn_weights = attentions[..., -1:, :-1]  # 取最后一个位置对之前位置的注意力
                        else:
                            # 如果维度不符合预期，使用计算的window_attention
                            attn_weights = self.compute_window_attention(module, hidden_states, keys)
                    else:
                        # 如果没有注意力输出，使用compute_window_attention计算
                        attn_weights = self.compute_window_attention(module, hidden_states, keys)
                    
                    # 确保attn_weights张量有效
                    if not isinstance(attn_weights, torch.Tensor) or attn_weights.numel() == 0:
                        print(f"无效的attn_weights: {type(attn_weights)}")
                        # 创建一个有效的默认值
                        attn_weights = torch.ones(keys.size(0), 1, keys.size(-2), device=keys.device)
                    
                    # 对多头注意力取平均 - 增强稳健性
                    if attn_weights.dim() > 2:
                        scores = attn_weights.mean(1)
                    else:
                        scores = attn_weights
                    
                    # 安全检查scores的维度
                    if scores.dim() == 1:
                        # 单维张量，添加必要的批次维度
                        scores = scores.unsqueeze(0)
                    
                    # 将注意力分数重复扩展到所有头 - 使用repeat替代expand
                    num_heads = keys.shape[1]
                    batch_size = keys.shape[0]
                    
                    # 确保scores至少有2D (批次和序列长度)
                    if scores.dim() == 1:
                        scores = scores.unsqueeze(0)
                    
                    # 添加头维度如果需要
                    if scores.dim() == 2:
                        scores = scores.unsqueeze(1)
                    
                    # 使用repeat扩展到所有头
                    if scores.size(1) == 1 and num_heads > 1:
                        scores = scores.repeat(1, num_heads, 1)
                    
                    # 添加对最后一个token的高权重
                    if scores.size(-1) < keys.size(-2):
                        # 使用最大值填充最后一个位置，确保最后一个token被保留
                        max_val = scores.max().item() if scores.numel() > 0 else 1.0
                        scores = torch.nn.functional.pad(scores, (0, 1), value=max_val)
                    
                    return scores
                    
                except Exception as e:
                    print(f"TOVA评分计算错误: {str(e)}")
                    # 出错时使用均匀分布
                    uniform_scores = torch.ones(keys.size(0), keys.size(1), keys.size(-2), device=keys.device)
                    return torch.nn.functional.normalize(uniform_scores, p=1, dim=-1)  # 归一化确保和为1
            
            def forward_hook(self, module, args, kwargs, output):
                """增强的forward_hook，避免position_ids出错"""
                try:
                    # 获取输入张量
                    hidden_states = None
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        hidden_states = args[0]
                    else:
                        if isinstance(output, tuple) and len(output) > 0:
                            hidden_states = output[0] 
                        elif hasattr(output, "last_hidden_state"):
                            hidden_states = output.last_hidden_state
                    
                    # 如果无法获取hidden_states，直接返回原始输出
                    if hidden_states is None:
                        print("警告: 无法获取hidden_states，跳过处理")
                        return output
                    
                    # 检查张量有效性
                    if not isinstance(hidden_states, torch.Tensor):
                        print(f"警告: hidden_states类型异常: {type(hidden_states)}")
                        return output
                    
                    # 确保张量至少是2D
                    if hidden_states.dim() < 2:
                        print(f"警告: hidden_states维度不足: {hidden_states.dim()}")
                        return output
                    
                    # 获取批次大小和序列长度
                    bsz, seq_len = hidden_states.size(0), hidden_states.size(1)
                    if bsz <= 0 or seq_len <= 0:
                        print(f"警告: 无效的批次大小或序列长度: {bsz}, {seq_len}")
                        return output
                    
                    device = hidden_states.device
                    
                    # 创建位置编码
                    position_ids = torch.arange(seq_len, device=device)
                    position_ids = position_ids.unsqueeze(0)  # 添加批次维度
                    
                    # 使用repeat代替expand，更安全
                    if bsz > 1:
                        position_ids = position_ids.repeat(bsz, 1)  # 明确使用repeat复制
                    
                    # 设置位置编码，确保两种格式都提供
                    kwargs['position_ids'] = position_ids
                    if 'position_embeddings' not in kwargs:
                        kwargs['position_embeddings'] = position_ids
                    
                    # 调用父类方法完成处理
                    return super().forward_hook(module, args, kwargs, output)
                    
                except Exception as e:
                    print(f"TOVAPress forward_hook错误详情: {str(e)}")
                    import traceback
                    traceback.print_exc()  # 打印完整堆栈跟踪
                    return output
        
        return CustomTOVAPress(compression_ratio=compression_ratio)
    
    elif press_type == "snap":
        print("使用 SnapKVPress (适用于 LoRA 微调模型)")
        return SnapKVPress(compression_ratio=compression_ratio)
    
    elif press_type == "streaming":
        print(f"使用 StreamingLLMPress (窗口大小: {window_size})")
        num_sink_tokens = 4  # 示例值，实际使用时需要根据模型和任务调整
        print(f"警告: 使用默认 {num_sink_tokens} 个 sink tokens，音频处理可能需要特殊配置")
        return StreamingLLMPress(compression_ratio=compression_ratio, n_sink=num_sink_tokens)
    
    else:
        print(f"未知的压缩类型 '{press_type}'，默认使用 KnormPress")
        return KnormPress(compression_ratio=compression_ratio)

class VESUSTimingStats:
    """跟踪VESUS情感识别任务的推理时间统计，支持CUDA Events和GPU内存监控"""
    def __init__(self):
        self.timing_records = []
        self.emotion_stats = defaultdict(list)
        self.person_stats = defaultdict(list)
        self.cuda_available = torch.cuda.is_available()
        self.initial_memory = 0
        self.peak_memory = 0
        self.total_peak_memory = 0
        
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
            print(f"初始GPU内存使用: {self.initial_memory / 1024**3:.2f} GB")
    
    def record_initial_memory(self):
        """记录初始内存使用"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            gc.collect()
            self.initial_memory = torch.cuda.memory_allocated()
            self.total_peak_memory = 0
            torch.cuda.reset_peak_memory_stats()
        else:
            self.initial_memory = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   emotion_label=None, person_id=None, gpu_memory_peak=None):
        """添加一条时间记录"""
        current_memory = 0
        if self.cuda_available:
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            self.peak_memory = max(self.peak_memory, peak_memory)
            # 更新总峰值内存
            if gpu_memory_peak:
                self.total_peak_memory = max(self.total_peak_memory, gpu_memory_peak)
            else:
                self.total_peak_memory = max(self.total_peak_memory, peak_memory)
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "emotion_label": emotion_label,
            "person_id": person_id,
            "gpu_memory_current": current_memory / 1024**3 if self.cuda_available else 0,
            "gpu_memory_peak": gpu_memory_peak / 1024**3 if gpu_memory_peak else 0,
            "kv_press_enabled": True,
            "compression_ratio": kv_press_config["compression_ratio"]
        }
        
        self.timing_records.append(record)
        
        if emotion_label:
            self.emotion_stats[emotion_label].append(record)
        
        if person_id:
            self.person_stats[person_id].append(record)
    
    def get_summary(self):
        """获取总体统计摘要"""
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        df = pd.DataFrame(self.timing_records)
        
        summary = {
            "total_samples": len(df),
            "avg_prefill_time": df["prefill_time"].mean(),
            "avg_decode_time": df["decode_time"].mean(),
            "avg_total_time": df["total_time"].mean(),
            "avg_tokens_per_sec": df["tokens_per_sec"].mean(),
            "prefill_percentage": (df["prefill_time"].sum() / df["total_time"].sum()) * 100,
            "decode_percentage": (df["decode_time"].sum() / df["total_time"].sum()) * 100,
            "gpu_memory_stats": {
                "initial_memory_gb": self.initial_memory / 1024**3 if self.cuda_available else 0,
                "peak_memory_gb": self.total_peak_memory / 1024**3 if self.cuda_available else 0,
                "avg_current_memory_gb": df["gpu_memory_current"].mean() if self.cuda_available else 0,
                "max_peak_memory_gb": df["gpu_memory_peak"].max() if self.cuda_available else 0,
            },
            "kv_press_stats": {
                "compression_ratio": kv_press_config["compression_ratio"],
                "press_type": kv_press_config["press_type"],
                "samples_with_compression": len(df)
            }
        }
        
        # 添加情感统计
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
        """导出统计数据到JSON文件"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records,
            "kv_press_config": kv_press_config
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def emotion_doc_to_text(doc, kwargs):
    """生成情感识别任务的提示词"""
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    # 从样本中获取信息
    question = doc.get("question", "What emotion is expressed in this audio segment?")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    
    # 情感识别指令
    instruction = f"{question}\n\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}\n\nPlease select the correct answer (A, B, C, or D)."
    format_text = "Your response should be formatted as follows: Answer:"
    
    # 定义提示词结构
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # 构建完整提示词
    return f"{pre_prompt}{user_prompt}<|audio_1|>{instruction} {format_text} <answer here>{prompt_suffix}{assistant_prompt}"

def clean_emotion_response(response):
    """清理情感识别响应"""
    if not response or response.strip() == "":
        return ""
    
    # 清理前缀和特殊标记
    for marker in ["answer:", "emotion:", "content:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    # 移除其他常见标记
    response = re.sub(r'<answer here>', '', response)
    response = re.sub(r'<emotion here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    
    return response.strip()

def extract_emotion_answer_from_response(response, sample):
    """从响应中提取情感答案"""
    # 清理响应
    cleaned_response = clean_emotion_response(response)
    
    # 获取选项
    choices = {
        "A": sample.get("choice_a", "").strip(),
        "B": sample.get("choice_b", "").strip(), 
        "C": sample.get("choice_c", "").strip(),
        "D": sample.get("choice_d", "").strip()
    }
    
    # 首先尝试直接匹配字母答案
    for letter in ["A", "B", "C", "D"]:
        if letter in cleaned_response.upper():
            return letter
    
    # 如果没有找到字母，尝试匹配情感词
    cleaned_lower = cleaned_response.lower()
    for letter, choice in choices.items():
        if choice.lower() in cleaned_lower:
            return letter
    
    # 如果都没找到，返回第一个字母作为默认值
    return "A"

def calculate_emotion_metrics(predictions, ground_truths, emotion_labels):
    """计算情感分类指标：准确率、精确率、召回率和F1分数"""
    # 过滤掉无效的预测和真实标签
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
    
    # 创建标签映射
    label_map = {label: idx for idx, label in enumerate(sorted(emotion_labels))}
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

def load_vesus_dataset(json_file_path):
    """加载VESUS情感识别数据集，跳过person2和person10的happy样本"""
    if not os.path.exists(json_file_path):
        print(f"错误: 数据集文件不存在: {json_file_path}")
        return []
    
    print(f"加载VESUS情感数据集: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 过滤有效的数据项，跳过person2和person10的happy情感样本
        valid_samples = []
        filtered_count = 0
        for item in data:
            if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
                # 过滤掉person2和person10的happy情感样本
                person_id = item.get('person_id', '')
                emotion_label = item.get('emotion_label', '').lower()
                
                # 将person_id转换为字符串进行标准化比较，处理数字和字符串两种格式
                person_id_str = str(person_id).lower()
                
                # 检查多种可能的格式：数字(2,10)、字符串("2","10")、带前缀("person2","person10")
                is_target_person = (
                    person_id_str in ['2', '10'] or  # 数字或数字字符串
                    person_id_str in ['person2', 'person10'] or  # 带前缀的字符串
                    person_id == 2 or person_id == 10  # 直接的数字比较
                )
                
                if (is_target_person and emotion_label == 'happy'):
                    filtered_count += 1
                    print(f"跳过样本: {person_id} - {emotion_label} - {item.get('path', '')}")
                    continue
                
                valid_samples.append(item)
        
        print(f"跳过了 {filtered_count} 个样本（person2和person10的happy情感）")
        print(f"加载了 {len(valid_samples)} 个有效样本")
        
        # 统计情感分布
        emotion_counts = defaultdict(int)
        person_emotion_counts = defaultdict(lambda: defaultdict(int))
        for sample in valid_samples:
            emotion = sample.get('emotion_label', 'unknown')
            person = sample.get('person_id', 'unknown')
            emotion_counts[emotion] += 1
            person_emotion_counts[person][emotion] += 1
        
        print(f"情感分布: {dict(emotion_counts)}")
        print(f"各人员情感分布:")
        for person, emotions in person_emotion_counts.items():
            # 标准化person_id显示格式
            person_str = str(person).lower()
            is_filtered_person = (
                person_str in ['2', '10'] or 
                person_str in ['person2', 'person10'] or 
                person == 2 or person == 10
            )
            if is_filtered_person:
                print(f"  {person}: {dict(emotions)} (已跳过happy样本)")
            else:
                print(f"  {person}: {dict(emotions)}")
        
        return valid_samples
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return []

def prepare_audio_for_processor(audio_path, data_path, target_sr=16000):
    """使用 librosa 处理音频文件"""
    try:
        # 构建完整的音频文件路径
        full_audio_path = os.path.join(data_path, audio_path)
        
        if not os.path.exists(full_audio_path):
            print(f"音频文件不存在: {full_audio_path}")
            # 创建3秒静音作为fallback
            silence = np.zeros(target_sr * 3, dtype=np.float32)
            return [(silence, target_sr)]
        
        # 使用 librosa 读取音频文件（自动转换为单声道和指定采样率）
        audio_data, sample_rate = librosa.load(
            full_audio_path, 
            sr=target_sr,      # 目标采样率
            mono=True,         # 转换为单声道
            dtype=np.float32   # 数据类型
        )
        
        # librosa.load 已经处理了重采样和单声道转换，无需额外处理
        
        # 确保数据类型为float32并归一化（librosa通常已经归一化到[-1,1]）
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        return [(audio_data, target_sr)]
        
    except Exception as e:
        print(f"使用 librosa 处理音频文件时出错 {audio_path}: {e}")
        # 回退到 soundfile
        try:
            # 使用 soundfile 作为备选方案
            audio_data, sample_rate = sf.read(full_audio_path)
            
            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # 重采样到目标采样率
            if sample_rate != target_sr:
                try:
                    # 使用 librosa 重采样
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                except Exception:
                    # 如果 librosa 重采样失败，回退到 scipy
                    audio_data = signal.resample(audio_data, int(len(audio_data) * target_sr / sample_rate))
            
            # 确保数据类型为float32并归一化
            audio_data = audio_data.astype(np.float32)
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val
            
            return [(audio_data, target_sr)]
            
        except Exception as e2:
            print(f"备选方案也失败: {e2}")
            # 返回3秒静音
            silence = np.zeros(target_sr * 3, dtype=np.float32)
            return [(silence, target_sr)]

def extract_advanced_audio_features(audio_data, sample_rate=16000):
    """使用 librosa 提取高级音频特征（可选）"""
    try:
        features = {}
        
        # 1. MFCC 特征
        mfcc = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate,
            n_mfcc=13
        )
        features['mfcc'] = mfcc
        
        # 2. 色度特征
        chroma = librosa.feature.chroma(
            y=audio_data, 
            sr=sample_rate
        )
        features['chroma'] = chroma
        
        # 3. 光谱质心
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=sample_rate
        )
        features['spectral_centroid'] = spectral_centroids
        
        # 4. 光谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, 
            sr=sample_rate
        )
        features['spectral_bandwidth'] = spectral_bandwidth
        
        # 5. 过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        features['zero_crossing_rate'] = zero_crossing_rate
        
        # 6. RMS 能量
        rms = librosa.feature.rms(y=audio_data)
        features['rms'] = rms
        
        # 7. Mel频谱图
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=128
        )
        features['mel_spectrogram'] = mel_spectrogram
        
        return features
        
    except Exception as e:
        print(f"高级特征提取失败: {e}")
        return {}

def preprocess_audio_with_librosa(audio_data, sample_rate=16000):
    """使用 librosa 进行音频预处理"""
    try:
        # 1. 音频归一化
        audio_normalized = librosa.util.normalize(audio_data)
        
        # 2. 预强调滤波器（可选，用于语音处理）
        audio_preemphasized = librosa.effects.preemphasis(audio_normalized)
        
        # 3. 去除静音部分
        audio_trimmed, _ = librosa.effects.trim(
            audio_preemphasized, 
            top_db=20  # 静音阈值（dB）
        )
        
        # 4. 如果音频太短，进行填充
        min_length = sample_rate * 1  # 至少1秒
        if len(audio_trimmed) < min_length:
            audio_trimmed = np.pad(audio_trimmed, (0, min_length - len(audio_trimmed)), mode='constant')
        
        return audio_trimmed
        
    except Exception as e:
        print(f"librosa 音频预处理失败: {e}")
        return audio_data  # 返回原始音频

def extract_emotion_audio_features(audio_data, sample_rate=16000):
    """专门为情感识别提取音频特征"""
    try:
        features = {}
        
        # 基本特征
        features.update(extract_advanced_audio_features(audio_data, sample_rate))
        
        # 情感相关的特征
        
        # 1. 基频（F0）- 用于情感分析
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        features['f0'] = f0
        features['voiced_flag'] = voiced_flag
        
        # 2. 谐波和感知特征
        harmonic, percussive = librosa.effects.hpss(audio_data)
        features['harmonic'] = harmonic
        features['percussive'] = percussive
        
        # 3. 音调特征
        tonnetz = librosa.feature.tonnetz(
            y=harmonic, 
            sr=sample_rate
        )
        features['tonnetz'] = tonnetz
        
        # 4. 统计特征
        features['stats'] = {
            'mean_f0': np.nanmean(f0),
            'std_f0': np.nanstd(f0),
            'mean_energy': np.mean(features['rms']),
            'std_energy': np.std(features['rms']),
            'mean_zcr': np.mean(features['zero_crossing_rate']),
            'std_zcr': np.std(features['zero_crossing_rate'])
        }
        
        return features
        
    except Exception as e:
        print(f"情感音频特征提取失败: {e}")
        return extract_advanced_audio_features(audio_data, sample_rate)

def extract_emotion_answer(text, choices):
    """从模型输出文本中提取情感答案，过滤system prompt信息"""
    if not text:
        return ""
    
    # 移除常见的system prompt模式
    text = re.sub(r'^.*?(?:system|assistant|user).*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^.*?(?:Answer|Response|Output).*?:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*<?/?s?>\s*', '', text)
    
    text_lower = text.lower().strip()
    
    # 优先匹配明确的选项格式
    option_patterns = [
        r'(?:选择|答案|answer|choice|option)?\s*[：:]\s*([ABCD])',
        r'([ABCD])[).]',
        r'([ABCD])\s*[：:]',
        r'(?:选项|option|choice)\s*([ABCD])',
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 直接检测a/b/c/d回答
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return "A"
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return "B"
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return "C"
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return "D"
    
    # 检查是否含有明确的选项指示
    option_patterns_dict = {
        'A': ["option a", "choice a", "a)", "(a)"],
        'B': ["option b", "choice b", "b)", "(b)"],
        'C': ["option c", "choice c", "c)", "(c)"],
        'D': ["option d", "choice d", "d)", "(d)"]
    }
    
    for option, patterns in option_patterns_dict.items():
        if any(pattern in text_lower for pattern in patterns):
            return option
    
    # 尝试通过情感关键词匹配
    emotion_keywords = {
        'angry': ['anger', 'frustrated', 'mad', 'furious'],
        'happy': ['joy', 'cheerful', 'pleased', 'delighted'],
        'sad': ['sadness', 'melancholy', 'depressed', 'sorrow'],
        'fearful': ['fear', 'anxiety', 'scared', 'afraid'],
        'monotone': ['flat', 'emotionless', 'neutral', 'bland']
    }
    
    # 检查选项中的情感关键词
    for choice_key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
        if choice_key in choices:
            choice_text = choices[choice_key].lower()
            for emotion, keywords in emotion_keywords.items():
                if emotion in choice_text or any(keyword in choice_text for keyword in keywords):
                    if any(keyword in text_lower for keyword in keywords) or emotion in text_lower:
                        return choice_key[-1].upper()  # 返回A/B/C/D
    
    return ""

def create_emotion_prompt(sample):
    """创建情感识别任务的提示词"""
    question = sample.get("question", "What emotion is expressed in this audio segment?")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")
    
    # 构建提示词
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
    from types import MethodType
    from kvpress.pipeline import KVPressAudioTranscriptionPipeline
    
    def custom_extract_mfcc(self, audio):
        """使用 librosa 提取音频的 MFCC 特征"""
        if isinstance(audio, tuple):
            audio_data, sampling_rate = audio
        else:
            audio_data = audio
            sampling_rate = 16000
            
        # 确保音频是一维数组
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # 取第一个通道
            
        # 确保音频长度合适
        max_length = sampling_rate * 30  # 最多30秒
        if len(audio_data) > max_length:
            audio_data = audio_data[:max_length]
            
        # 使用 librosa 提取 MFCC 特征
        try:
            # 提取 MFCC 特征
            mfcc_features = librosa.feature.mfcc(
                y=audio_data, 
                sr=sampling_rate,
                n_mfcc=13,          # 13个MFCC系数
                n_fft=2048,         # FFT窗口大小
                hop_length=512,     # 跳跃长度
                n_mels=128,         # Mel滤波器数量
                fmin=0,             # 最小频率
                fmax=sampling_rate/2  # 最大频率（奈奎斯特频率）
            )
            
            # 转置特征矩阵，使时间轴在第一维
            mfcc_features = mfcc_features.T
            
            # 如果特征太长，截断到合理长度
            max_frames = 3000  # 大约30秒的音频
            if mfcc_features.shape[0] > max_frames:
                mfcc_features = mfcc_features[:max_frames]
            
            return mfcc_features
            
        except Exception as e:
            print(f"MFCC 特征提取失败: {e}")
            # 回退到简单的音频数据
            return audio_data
    
    # 替换方法
    KVPressAudioTranscriptionPipeline.extract_mfcc = custom_extract_mfcc
    
    # 确认GPU可用性并清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda"
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU available, using CPU")
    
    # 音频相关特殊token ID
    _AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
    
    # 获取配置参数
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    compression_ratio = kv_press_config["compression_ratio"]
    press_type = kv_press_config["press_type"]
    
    print(f"使用 GPU ID: {gpu_id}")
    print(f"KV Press 配置: 压缩比率={compression_ratio}, 压缩类型={press_type}")
    
    if sample_limit > 0:
        print(f"样本限制设置为: {sample_limit}")
    
    # 数据路径配置
    data_path = os.environ.get("VESUS_DATA_PATH", 
        "/data/hepeize05/Audio_Longbench/Dataset/VESUS")
    emotion_json_file = os.path.join(data_path, "audio_emotion_dataset.json")
    result_dir = os.environ.get("RESULTS_DIR", './VESUS_KVPress_Results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = f'{result_dir}/vesus_kvpress_results_{press_type}_{compression_ratio}.json'
    timing_output_file = f'{result_dir}/vesus_timing_stats_{press_type}_{compression_ratio}.json'
    
    print(f"\n=== VESUS情感识别KV Press评测配置 ===")
    print(f"GPU ID: {gpu_id}")
    print(f"数据路径: {data_path}")
    print(f"JSON文件: {emotion_json_file}")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 50)
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")
    
    # 加载模型
    print("加载Phi-4-multimodal-instruct模型...")
    model_path = "/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B"
    
    # 首先加载 processor
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # 然后加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # 使用 auto 而不是 cuda
        torch_dtype="auto",
        trust_remote_code=True,
        **kv_press_config["model_kwargs"]
    )
    
    # 为 Phi4MM 模型添加KV Press补丁
    patch_phi4mm_for_kvpress(model)
    
    # 初始化 KV Press
    press = initialize_kv_press_simplified(model, kv_press_config)
    
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # 配置参数
    lmms_eval_specific_kwargs = {
        "pre_prompt": "",
        "post_prompt": ""
    }
    
    # 创建pipeline实例
    pipeline = KVPressAudioTranscriptionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        processor=processor,
        audio_special_token_id=_AUDIO_SPECIAL_TOKEN_ID
    )
    
    print("模型加载成功")
    
    # 加载VESUS数据集
    samples = load_vesus_dataset(emotion_json_file)
    
    if not samples:
        print("错误: 未找到任何数据样本")
        return
    
    # 应用样本限制
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"应用样本限制，处理 {len(samples)} 个样本")
    
    # 创建时间统计器
    timing_stats = VESUSTimingStats()
    
    # 记录初始内存
    if hasattr(timing_stats, 'record_initial_memory'):
        timing_stats.record_initial_memory()
    elif torch.cuda.is_available():
        # 如果没有record_initial_memory方法，手动清理和重置内存统计
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        print(f"初始GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    results = []
    total_correct = 0
    emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    person_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    print(f"开始评估 {len(samples)} 个样本...")
    
    # 检测是否在screen或非交互式环境中运行
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("检测到screen或非交互式环境，使用简化进度显示")
    
    # 设置tqdm参数
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="VESUS KV Press评估", **tqdm_kwargs)
    
    for idx, sample in progress_bar:
        # 重置GPU内存统计
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        try:
            # 加载音频数据
            audio_path = sample.get("path", "")
            
            # 检查音频文件是否存在
            full_audio_path = os.path.join(data_path, audio_path)
            if not os.path.exists(full_audio_path):
                print(f"跳过不存在的音频文件: {full_audio_path}")
                # 创建跳过记录
                result_entry = {
                    "path": audio_path,
                    "person_id": sample.get("person_id", "unknown"),
                    "emotion_label": sample.get("emotion_label", "unknown"),
                    "question": sample.get("question", ""),
                    "choices": sample.get("choices", {}),
                    "answer_gt": sample.get("answer_gt", ""),
                    "ground_truth": sample.get("answer_gt", ""),
                    "model_output": "SKIPPED - File not found",
                    "extracted_answer": "skip",
                    "is_correct": False,
                    "output_tokens": 0,
                    "prefill_time": 0.0,
                    "decode_time": 0.0,
                    "total_time": 0.0,
                    "peak_memory_gb": 0.0,
                    "skipped": True,
                    "skip_reason": "Audio file not found"
                }
                results.append(result_entry)
                continue
                
            audio_raw = prepare_audio_for_processor(audio_path, data_path)
            
            if audio_raw is None:
                continue
            
            # 获取样本信息
            emotion_label = sample.get("emotion_label", "unknown")
            person_id = sample.get("person_id", "unknown")
            answer_gt = sample.get("answer_gt", "").upper()
            
            # 生成情感识别提示词
            prompt = emotion_doc_to_text(sample, lmms_eval_specific_kwargs)
            
            # 初始化timing_info变量
            timing_info = {
                "prefill_time": 0.0,
                "generation_time": 0.0,
                "total_time": 0.0
            }
            
            # 使用pipeline处理，应用压缩
            with press(model) if press is not None else contextlib.nullcontext():
                result = pipeline(
                    prompt=prompt,
                    audios=audio_raw,
                    press=press,
                    input_mode=2,
                    measure_time=True,
                    max_new_tokens=64,
                    do_sample=False,
                    return_legacy_cache=True,  # 处理缓存格式警告  
                )
            
            # 获取内存使用信息（从pipeline返回的metrics中获取）
            peak_memory = 0
            current_memory = 0
            memory_increase = 0
            
            if 'metrics' in result and result['metrics']:
                metrics = result['metrics']
                peak_memory_gb = metrics.get('peak_memory_gb', 0.0)
                peak_memory = peak_memory_gb * (1024**3)  # 转换回字节
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_increase = peak_memory - initial_memory
            else:
                # 如果pipeline没有返回内存信息，fallback到手动获取
                peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_increase = peak_memory - initial_memory
            
            # 获取结果
            response = result["text"]
            is_valid = result.get("is_valid", True)
            skip_wer_computation = result.get("skip_wer_computation", False)

            # 如果返回了无效响应标记，输出警告
            if not is_valid or skip_wer_computation or response == "[INVALID_RESPONSE]":
                print(f"\n注意: 样本 {idx} (ID: {sample.get('id', 'unknown')}) 被标记为无效响应")

            # 获取指标
            if "metrics" in result:
                timing_info = {
                    "prefill_time": result["metrics"]["prefill_time"],
                    "generation_time": result["metrics"]["generation_time"],
                    "total_time": result["metrics"]["total_time"],
                    "initial_memory": initial_memory,
                    "peak_memory": peak_memory,
                    "memory_increase": memory_increase
                }
            
            # 提取答案
            predicted_answer = extract_emotion_answer_from_response(response, sample)
            is_correct = (predicted_answer == answer_gt)
            
            if is_correct:
                total_correct += 1
            
            # 更新统计
            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1
            
            if is_correct:
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1
            
            # 更新进度条
            current_accuracy = total_correct / (idx + 1)
            progress_bar.set_postfix({
                'Acc': f'{current_accuracy:.3f}',
                'Emotion': emotion_label[:8],
                'Person': person_id,
                'Press': press_type[:4]
            })
            
            # 保存详细结果
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
                "response_text": response,
                "kv_press_type": press_type,
                "compression_ratio": compression_ratio,
                "timing": timing_info
            })
            
            # 收集timing信息（排除第一个样本的前100个样本）
            if idx > 0 and idx <= 100:
                timing_stats.add_record(
                    timing_info["prefill_time"], 
                    timing_info["generation_time"],
                    len(response.split()),  # 简单的token估计
                    result.get("audio_token_info", {}).get("length", 0),
                    emotion_label, person_id, peak_memory
                )
            
            # 内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            continue
    
    # 计算最终统计
    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 计算F1分数和其他指标
    all_predictions = [result["predicted_answer"] for result in results]
    all_ground_truths = [result["answer_gt"] for result in results]
    all_emotion_labels = list(set(all_ground_truths))
    
    # 计算情感分类指标
    emotion_metrics = calculate_emotion_metrics(all_predictions, all_ground_truths, all_emotion_labels)
    
    # 按情感计算准确率
    emotion_accuracies = {}
    for emotion, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracies[emotion] = stats["correct"] / stats["total"]
    
    # 按人员计算准确率
    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]
    
    # 创建结果摘要
    summary = {
        "total_samples": total_samples,
        "correct_samples": total_correct,
        "overall_accuracy": overall_accuracy,
        "metrics": emotion_metrics,
        "emotion_stats": dict(emotion_stats),
        "emotion_accuracies": emotion_accuracies,
        "person_stats": dict(person_stats),
        "person_accuracies": person_accuracies,
        "kv_press_config": kv_press_config,
        "config": {
            "gpu_id": gpu_id,
            "compression_ratio": compression_ratio,
            "press_type": press_type,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file,
            "filtered_person2_person10_happy": True
        },
        "timing": timing_stats.get_summary()
    }
    
    # 保存结果
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 保存时间统计
    timing_stats.export_to_json(timing_output_file)
    
    # 输出结果摘要
    print(f"\n=== VESUS情感识别KV Press评测结果摘要 ===")
    print(f"总样本数: {total_samples}")
    print(f"总体准确率: {overall_accuracy:.3f}")
    
    # 显示F1 Score结果
    print(f"\n=== F1 Score 指标 ===")
    print(f"Weighted F1 Score: {emotion_metrics['f1_score']:.4f}")
    print(f"Weighted Precision: {emotion_metrics['precision']:.4f}")
    print(f"Weighted Recall: {emotion_metrics['recall']:.4f}")
    print(f"有效样本: {emotion_metrics['valid_samples']}/{emotion_metrics['total_samples']}")
    print(f"跳过了person2和person10的happy样本")
    
    print(f"\n各情感准确率:")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== KV Press配置 ===")
    print(f"压缩类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"最小序列长度: {kv_press_config['min_seq_len']}")
    
    print(f"\n=== 时间统计（CUDA Events精确测量，排除第一个样本的前100个样本）===")
    print(f"统计样本数: {overall_summary.get('total_samples', 0)}")
    print(f"平均推理时间: {overall_summary.get('avg_total_time', 0):.4f}秒")
    print(f"平均Prefill时间: {overall_summary.get('avg_prefill_time', 0):.4f}秒 ({overall_summary.get('prefill_percentage', 0):.1f}%)")
    print(f"平均Decode时间: {overall_summary.get('avg_decode_time', 0):.4f}秒 ({overall_summary.get('decode_percentage', 0):.1f}%)")
    print(f"平均吞吐量: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/秒")
    
    # 打印GPU内存统计
    if 'gpu_memory_stats' in overall_summary:
        gpu_stats = overall_summary['gpu_memory_stats']
        print("\n===== GPU内存统计 =====")
        print(f"初始GPU内存: {gpu_stats['initial_memory_gb']:.2f} GB")
        print(f"峰值GPU内存: {gpu_stats['peak_memory_gb']:.2f} GB")
        print(f"平均当前内存: {gpu_stats['avg_current_memory_gb']:.2f} GB")
        print(f"最大峰值内存: {gpu_stats['max_peak_memory_gb']:.2f} GB")
    
    print(f"\n结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

if __name__ == "__main__":
    random.seed(42)
    main()
