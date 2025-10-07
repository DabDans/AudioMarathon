import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
import contextlib
from scipy.io import wavfile
from scipy import signal
import torch.nn.functional as F

# KV Press imports
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
# 禁用transformers警告
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class CustomTOVAPress(TOVAPress):
    """TOVA压缩的自定义实现，修复position_embeddings相关问题"""
    
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
    
    def score(self, module, hidden_states, keys, values, attentions, **kwargs):
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
            return super().forward_hook(module, args, output, **kwargs)
            
        except Exception as e:
            print(f"TOVAPress forward_hook错误详情: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整堆栈跟踪
            return output

# KV Press配置
kv_press_config = {
    "compression_ratio": float(os.environ.get("COMPRESSION_RATIO", 0.5)),
    "head_dims": None,                # 会在运行时设置
    "num_attention_heads": None,      # 会在运行时设置
    "press_type": os.environ.get("PRESS_TYPE", "knorm"),  # 默认压缩类型: knorm
    "return_indices": True,           # 是否返回保留的索引，用于调试
    "min_seq_len": 128,               # 最小序列长度，低于此长度不压缩
    "model_kwargs": {
        "attn_implementation": "sdpa",  # 使用 sdpa 实现而不是 flash attention
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False
    }
}

def patch_phi4mm_for_kvpress(model):
    """为 Phi4MM 模型添加 rotary_emb 属性以便与 KV Press 兼容"""
    # 检查模型结构
    if not hasattr(model, "model") or not hasattr(model.model, "layers") or len(model.model.layers) == 0:
        print("警告: 模型结构不符合预期，无法应用KV Press补丁")
        return False
    
    # 找到第一个注意力层
    first_layer = model.model.layers[0]
    if not hasattr(first_layer, "self_attn"):
        print("警告: 注意力层结构不符合预期，无法应用KV Press补丁")
        return False
    
    # 检查是否已经有全局rotary_emb
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        print("模型已有全局rotary_emb属性，无需添加补丁")
        return True
    
    # 查找注意力层中是否有可用的rotary_emb
    rotary_found = False
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "rotary_emb") and layer.self_attn.rotary_emb is not None:
            # 找到了rotary_emb，添加到模型全局属性
            model.model.rotary_emb = layer.self_attn.rotary_emb
            print(f"已从层 {layer_idx} 提取并添加全局 rotary_emb 属性")
            rotary_found = True
            break
    
    # 如果没有找到rotary_emb，尝试创建一个
    if not rotary_found:
        try:
            from transformers.models.phi import modeling_phi
            config = model.config
            head_dim = config.hidden_size // config.num_attention_heads
            
            # 创建标准的RotaryEmbedding - 使用Phi模型的RotaryEmbedding类
            if hasattr(modeling_phi, 'PhiRotaryEmbedding'):
                model.model.rotary_emb = modeling_phi.PhiRotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=getattr(config, 'rope_theta', 10000.0)
                )
                print("已手动创建并添加全局 rotary_emb 属性")
                rotary_found = True
            else:
                print("警告: 无法找到合适的RotaryEmbedding类")
                return False
                
        except Exception as e:
            print(f"创建rotary_emb时出错: {str(e)}")
            return False
    
    # 检查是否还需要添加其他KV Press所需的属性或方法
    if not hasattr(model, "config") or not hasattr(model.config, "num_attention_heads"):
        print("警告: 模型缺少KV Press所需的配置参数")
        return False
    
    # 验证补丁是否成功
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        # 添加一个标记，表示已经应用了补丁
        model.model._kvpress_patched = True
        return True
    else:
        print("补丁应用失败，无法找到或创建适用的rotary_emb")
        return False

def initialize_kv_press(model, compression_ratio=None):
    """根据模型特性选择最合适的KV Press类型"""
    if compression_ratio is None:
        compression_ratio = kv_press_config["compression_ratio"]
    
    # 检查模型配置
    if not hasattr(model, "config"):
        print("警告: 模型没有config属性，使用默认KnormPress")
        return KnormPress(compression_ratio=compression_ratio)
    
    # 更新配置
    kv_press_config["compression_ratio"] = compression_ratio
    
    # 获取模型配置信息
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    kv_press_config["head_dims"] = head_dim
    kv_press_config["num_attention_heads"] = config.num_attention_heads
    
    print(f"模型配置: hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, head_dim={head_dim}")
    
    # 检查模型层结构
    if not hasattr(model.model, "layers") or len(model.model.layers) == 0:
        print("警告: 模型层结构异常，使用默认KnormPress")
        return KnormPress(compression_ratio=compression_ratio)
    
    # 确定最佳Press类型
    press_type = kv_press_config["press_type"]
    has_rotary = hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None
    has_kvpress_patch = hasattr(model.model, "_kvpress_patched") and model.model._kvpress_patched
    
    print(f"Press类型配置: {press_type}, 压缩比率: {compression_ratio}")
    
    # 根据不同类型创建对应的Press实例
    if press_type == "knorm":
        print("使用KnormPress (基于Key-norm的注意力压缩)")
        return KnormPress(compression_ratio=compression_ratio)
    elif press_type == "expected" and has_rotary and has_kvpress_patch:
        print("使用ExpectedAttentionPress (基于RoPE的注意力压缩)")
        return ExpectedAttentionPress(compression_ratio=compression_ratio)
    elif press_type == "random":
        print("使用RandomPress (随机丢弃tokens)")
        return RandomPress(compression_ratio=compression_ratio)
    elif press_type == "observed":
        print("使用ObservedAttentionPress (基于观察到的注意力分数)")
        return ObservedAttentionPress(compression_ratio=compression_ratio)
    elif press_type == "tova":
        print("使用CustomTOVAPress (基于时间顺序的注意力值分析)")
        return CustomTOVAPress(compression_ratio=compression_ratio)
    elif press_type == "snap":
        print("使用SnapKVPress (适用于LoRA微调模型)")
        return SnapKVPress(compression_ratio=compression_ratio)
    elif press_type == "streaming":
        print("使用StreamingLLMPress (适用于长文本流式处理)")
        window_size = kv_press_config.get("window_size", 4096)
        return StreamingLLMPress(window_size=window_size)
    else:
        # 默认回退到KnormPress
        print(f"未知压缩类型 {press_type}，默认使用KnormPress")
        return KnormPress(compression_ratio=compression_ratio)

class DESEDTimingStats:
    """跟踪DESED声音事件检测任务的推理时间统计"""
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
        self.total_peak_memory = 0  # 添加峰值内存跟踪
        self.initial_gpu_memory = 0
    
    def record_initial_memory(self):
        """记录初始GPU内存使用情况"""
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            self.initial_gpu_memory = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None, peak_memory_gb=None):
        """添加一条时间记录"""
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        if audio_duration:
            self.total_audio_duration += audio_duration
        
        # 添加峰值内存跟踪
        if peak_memory_gb is not None:
            self.total_peak_memory += peak_memory_gb
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "task_type": task_type,
            "peak_memory_gb": peak_memory_gb if peak_memory_gb else 0.0  # 添加峰值内存
        }
        
        self.timing_records.append(record)
        
        if task_type:
            self.task_type_stats[task_type].append(record)
        
        if task_type:
            self.task_type_stats[task_type].append(record)
    
    def get_summary(self):
        """获取总体统计摘要"""
        if self.total_samples == 0:
            return {"error": "No samples processed"}
        
        avg_prefill = self.total_prefill_time / self.total_samples
        avg_decode = self.total_decode_time / self.total_samples
        avg_total = avg_prefill + avg_decode
        avg_tokens_per_sec = self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        avg_peak_memory = self.total_peak_memory / self.total_samples if self.total_samples > 0 else 0
        
        summary = {
            "total_samples": self.total_samples,
            "avg_prefill_time": avg_prefill,
            "avg_decode_time": avg_decode,
            "avg_total_time": avg_total,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_audio_duration": self.total_audio_duration,
            "avg_audio_duration": self.total_audio_duration / self.total_samples if self.total_samples > 0 else 0,
            "avg_peak_memory_gb": avg_peak_memory  # 添加平均峰值内存
        }
        
        # 添加任务类型统计
        task_summaries = {}
        for task_type, records in self.task_type_stats.items():
            if len(records) > 0:
                task_prefill = sum(r["prefill_time"] for r in records) / len(records)
                task_decode = sum(r["decode_time"] for r in records) / len(records)
                task_tokens = sum(r["output_tokens"] for r in records) / len(records)
                task_peak_memory = sum(r["peak_memory_gb"] for r in records) / len(records)
                
                task_summaries[task_type] = {
                    "samples": len(records),
                    "avg_prefill_time": task_prefill,
                    "avg_decode_time": task_decode,
                    "avg_total_time": task_prefill + task_decode,
                    "avg_tokens": task_tokens,
                    "avg_peak_memory_gb": task_peak_memory
                }
        
        return {
            "overall_summary": summary,
            "task_summaries": task_summaries
        }
    
    def export_to_json(self, output_file):
        """导出统计数据到JSON文件"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    从新的DESED任务JSON文件加载数据
    
    Args:
        json_file: DESED任务JSON文件路径
        audio_base_dir: 音频文件基目录
    
    Returns:
        dataset: 包含任务数据的列表
    """
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"错误: JSON文件不存在: {json_file}")
        return []
    
    print(f"加载DESED任务JSON: {json_file}")
    print(f"音频基目录: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return []
    
    # 检查JSON格式
    if not isinstance(data, dict) or 'tasks' not in data:
        print(f"错误: JSON文件格式不正确，期望包含'tasks'字段的字典格式")
        return []
    
    tasks = data['tasks']
    print(f"从JSON加载了 {len(tasks)} 个任务")
    
    # 统计任务类型
    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        # 构建完整的音频路径
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"警告: 任务缺少音频路径: {task}")
            continue
        
        # 检查音频文件是否存在
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:  # 只显示前5个缺失文件
                print(f"警告: 音频文件不存在: {full_audio_path}")
            continue
        
        # 获取任务信息
        task_type = task.get("task_type", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        
        # 获取选择题选项 - 新格式使用choices字典
        choices = task.get("choices", {})
        choice_a = choices.get("A", "")
        choice_b = choices.get("B", "")
        choice_c = choices.get("C", "")
        choice_d = choices.get("D", "")
        
        # 获取音频信息
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
        except Exception as e:
            print(f"无法读取音频文件信息 {full_audio_path}: {e}")
            continue
        
        # 创建数据项
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_type": task_type,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "original_events": task.get("all_events", []),
            "all_events": task.get("all_events", []),
            "primary_event": task.get("primary_event", ""),
            "correct_event": task.get("correct_event", ""),
            "path_extracted_event": task.get("path_extracted_event", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"qa_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[task_type] += 1
    
    if missing_files > 5:
        print(f"警告: 共有 {missing_files} 个音频文件不存在")
    
    print(f"加载了 {len(dataset)} 个有效样本")
    print(f"任务类型统计: {dict(task_type_stats)}")
    return dataset

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """按照DESED_test.py的方式正确处理音频文件"""
    
    try:
        # 方法1: 直接使用soundfile
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"soundfile加载失败: {e}")
            
            # 方法2: 使用ffmpeg
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav = temp_file.name
                
                print(f"使用ffmpeg转换: {audio_path} -> {temp_wav}")
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
                print(f"ffmpeg转换成功: 形状={audio.shape}, 采样率={sample_rate}Hz")
                
            except Exception as e:
                print(f"ffmpeg转换失败: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)  # 3秒静音
                sample_rate = target_sr
                print("生成静音替代音频")
        
        # 直接转换为单声道 - 无论原始格式如何都先转为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.flatten()  # 确保是一维数组
        print(f"转换为单声道: 形状={audio.shape}")
        
        # 重采样到目标采样率
        if sample_rate != target_sr and sample_rate > 0:
            audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
            sample_rate = target_sr
            print(f"重采样到{target_sr}Hz: 新长度={len(audio)}")
            
        # 检查音频是否为空
        if len(audio) == 0:
            print("警告: 音频为空，创建3秒静音")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
        
        # 确保数据类型为float32并归一化
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        return [(audio, sample_rate)]
        
    except Exception as e:
        print(f"音频处理出错: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [(silence, target_sr)]

def create_qa_prompt(doc):
    """为声音事件检测任务生成提示词"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    
    # 构建选择题格式的提示词
    prompt_text = f"""{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."""
    
    # 定义提示词结构
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # 构建完整提示词
    return f"{user_prompt}<|audio_1|>{prompt_text}{prompt_suffix}{assistant_prompt}"

def extract_answer_choice(response):
    """从模型响应中提取答案选择（A、B、C、D），只提取输出的答案部分"""
    if not response:
        return ""
    
    # 移除可能的system prompt和格式化标记
    # 查找assistant标记后的内容
    if '<|assistant|>' in response:
        response = response.split('<|assistant|>')[-1]
    elif 'assistant' in response.lower() and ':' in response:
        parts = response.split(':')
        for i, part in enumerate(parts):
            if 'assistant' in part.lower():
                if i + 1 < len(parts):
                    response = ':'.join(parts[i+1:])
                break
    
    # 移除其他常见的格式标记
    response = re.sub(r'<\|.*?\|>', '', response)
    response = re.sub(r'<.*?>', '', response)
    
    # 清理响应并转换为大写
    response = response.strip().upper()
    
    # 如果响应太长，只取前50个字符查找答案
    search_text = response[:50] if len(response) > 50 else response
    
    # 直接匹配单个字母
    if search_text in ['A', 'B', 'C', 'D']:
        return search_text
    
    # 优先匹配行首的选择（更可能是答案）
    match = re.search(r'^\s*([ABCD])\s*[.):]?', search_text, re.MULTILINE)
    if match:
        return match.group(1)
    
    # 匹配包含字母的响应（使用单词边界）
    match = re.search(r'\b([ABCD])\b', search_text)
    if match:
        return match.group(1)
    
    # 匹配选项格式 (如 "A.", "A)", "(A)")
    match = re.search(r'[(\[]?([ABCD])[)\].]?', search_text)
    if match:
        return match.group(1)
    
    # 最后尝试在完整响应中匹配第一个出现的选择
    match = re.search(r'([ABCD])', response)
    if match:
        return match.group(1)
    
    # 如果没有找到明确的选择，返回空字符串
    return ""

def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """评估声音事件检测任务准确性"""
    try:
        # 确保都是大写字母
        pred = predicted_choice.strip().upper() if predicted_choice else ""
        gt = ground_truth_choice.strip().upper() if ground_truth_choice else ""
        
        # 简单的准确率计算
        accuracy = 1.0 if pred == gt else 0.0
        
        return {
            "accuracy": accuracy,
            "predicted_choice": pred,
            "ground_truth_choice": gt,
            "is_correct": pred == gt
        }
    except Exception as e:
        print(f"评估声音事件检测准确性时出错: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

def main():
    # 获取环境变量配置
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    print(f"使用 GPU ID: {gpu_id}")
    
    # 从环境变量获取KV Press参数
    compression_ratio = float(os.environ.get("COMPRESSION_RATIO", 0.5))
    press_type = os.environ.get("PRESS_TYPE", "knorm")
    
    # 样本限制 (如果提供)
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"样本限制设置为: {sample_limit}")

    # 数据路径配置
    qa_json_file = "/data/hepeize05/Audio_Longbench/Dataset/DESED/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json"
    audio_base_dir = "/data/hepeize05/Audio_Longbench/Dataset/DESED/DESED_dataset/concatenated_audio"
    
    print(f"QA JSON文件: {qa_json_file}")
    print(f"音频基目录: {audio_base_dir}")
    
    # 加载DESED QA数据集
    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)
    
    result_dir = os.environ.get("RESULTS_DIR", './DESED_Results')
    os.makedirs(result_dir, exist_ok=True)

    # 修改输出文件路径和命名 - 加入KV Press标识
    output_file = f'{result_dir}/desed_sound_event_detection_results_kvpress_{press_type}_{compression_ratio}.json'
    timing_output_file = f'{result_dir}/desed_sound_event_detection_timing_stats_kvpress_{press_type}_{compression_ratio}.json'
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")

    # 音频特殊token ID
    _AUDIO_SPECIAL_TOKEN_ID = 200011

    # 创建时间统计器
    timing_stats = DESEDTimingStats()

    print(f"\n=== DESED声音事件检测评测配置 (KV Press) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"任务JSON文件: {qa_json_file}")
    print(f"音频基目录: {audio_base_dir}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 40)

    # 加载模型
    print("加载Phi-4-multimodal-instruct模型...")
    model_path = "/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 使用KV Press配置加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  
        torch_dtype="auto",
        trust_remote_code=True,
        **kv_press_config["model_kwargs"]
    )
    generation_config = GenerationConfig.from_pretrained(model_path)
    model.eval()

    # 为 Phi4MM 模型添加补丁以便与 KV Press 兼容
    patch_success = patch_phi4mm_for_kvpress(model)
    if not patch_success:
        print("警告: KV Press补丁应用失败，某些Press类型可能无法正常工作")

    # 初始化 KV Press
    press = initialize_kv_press(model, compression_ratio)

    # 音频特殊token ID
    _AUDIO_SPECIAL_TOKEN_ID = 200011

    # 创建pipeline实例
    pipeline = KVPressAudioTranscriptionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        processor=processor,
        audio_special_token_id=_AUDIO_SPECIAL_TOKEN_ID
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 打印初始内存使用情况并记录初始内存
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 记录初始内存用于后续计算
    timing_stats.record_initial_memory()  # 记录初始GPU内存

    print(f"使用数据集: {len(samples)} 个样本")
    
    # 应用样本限制
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"应用样本限制，处理 {len(samples)} 个样本")

    # 统计任务类型数量
    task_type_stats = defaultdict(int)
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_stats[task_type] += 1
    
    print(f"任务类型统计: {dict(task_type_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0
    
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)

    # 检测是否在screen或非交互式环境中运行
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("检测到screen或非交互式环境，使用简化进度显示")
    
    # 设置tqdm参数
    tqdm_kwargs = {
        'ascii': True,        # 使用ASCII字符而非Unicode
        'dynamic_ncols': True, # 自适应终端宽度
        'file': sys.stdout    # 确保直接输出到标准输出
    }

    print(f"开始评估 {len(samples)} 个样本...")
    
    # 打印初始内存使用情况并记录初始内存
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 记录初始内存用于后续计算
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="DESED声音事件检测评估 (KV Press)", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
            # 在每个样本处理前重置GPU内存统计，确保准确的峰值测量
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
            # 直接从音频路径加载音频数据
            audio_path = sample["audio"]["path"]
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"跳过不存在的音频文件: {audio_path}")
                # 创建跳过记录
                result_entry = {
                    "audio_path": audio_path,
                    "question": sample.get("question", ""),
                    "choices": sample.get("choices", {}),
                    "answer_gt": sample.get("answer_gt", ""),
                    "ground_truth": sample.get("answer_gt", ""),
                    "model_output": "SKIPPED - File not found",
                    "extracted_answer": "skip",
                    "is_correct": False,
                    "task_type": sample.get("task_type", "unknown"),
                    "audio_duration": 0.0,
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
                
            audio = prepare_audio_for_processor(audio_path)
            
            # 获取参考答案
            ground_truth_choice = sample.get("answer_gt", "")
            task_type = sample.get("task_type", "unknown")
            
            # 创建QA提示词
            prompt = create_qa_prompt(sample)

            # 使用Pipeline进行推理，支持KV Press压缩和CUDA Events精确计时
            try:
                # 使用pipeline处理，应用压缩
                with press(model) if press is not None else contextlib.nullcontext():
                    result = pipeline(
                        prompt=prompt,
                        audios=audio,
                        press=press,
                        input_mode=2,
                        measure_time=True,
                        max_new_tokens=10,  # 声音事件检测任务只需要很少的token
                        do_sample=False,
                        return_legacy_cache=True,  # 处理缓存格式警告
                    )
                
                # 初始化默认值，避免变量引用错误
                resp = result['text']
                prefill_time = 0.0
                decode_time = 0.0
                output_tokens = 0
                audio_token_length = 0  # 添加audio_token_length初始化
                input_tokens = 0  # 添加input_tokens初始化
                current_peak_memory = 0  # 添加峰值内存初始化
                
                # 从pipeline返回结果中正确提取数据
                if 'metrics' in result and result['metrics']:
                    metrics = result['metrics']
                    prefill_time = metrics.get('prefill_time', 0.0)
                    decode_time = metrics.get('generation_time', 0.0)  # pipeline中叫generation_time
                else:
                    # 如果没有metrics，使用默认值
                    prefill_time = 0.0
                    decode_time = 0.0
                
                # 获取输出token数量
                output_tokens = result.get('output_tokens', 0)
                input_tokens = result.get('input_tokens', 0)  # 从pipeline结果获取输入token数量
                
                # 从pipeline返回的metrics中获取峰值内存
                current_peak_memory = 0
                if 'metrics' in result and result['metrics']:
                    current_peak_memory = result['metrics'].get('peak_memory_gb', 0.0) * (1024**3)  # 转换回字节
                
            except Exception as pipeline_error:
                print(f"Pipeline推理失败: {pipeline_error}")
                print("回退到标准推理方式")
                
                # 处理输入
                inputs = processor(
                    text=prompt,
                    audios=audio,
                    return_tensors="pt",
                ).to(device)
                inputs['input_mode'] = torch.tensor([2])

                # 标准推理
                start_time = time.time()
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        generation_config=generation_config,
                        do_sample=False,
                        return_dict_in_generate=True
                    )
                end_time = time.time()
                
                # 获取峰值内存（fallback模式）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    current_peak_memory = torch.cuda.max_memory_allocated()
                else:
                    current_peak_memory = 0
                
                tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                resp = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                prefill_time = 0.0
                decode_time = end_time - start_time
                audio_token_length = 0  # 添加audio_token_length初始化
                input_tokens = inputs['input_ids'].shape[1]  # 获取输入token数量

            # 解析输出并提取答案
            predicted_choice = extract_answer_choice(resp)

            # 计算声音事件检测准确性
            metrics = evaluate_qa_accuracy(predicted_choice, ground_truth_choice)
            
            accuracy = metrics["accuracy"]
            is_correct = metrics["is_correct"]
            
            total_accuracy += accuracy
            processed_samples += 1

            # 更新任务类型统计
            task_type_total[task_type] += 1
            if is_correct:
                task_type_correct[task_type] += 1

            current_avg_acc = total_accuracy / processed_samples
            
            # 在screen环境下每10个样本更新一次，在标准环境下每个样本都更新
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_type[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })
                
                if is_screen_env:
                    # 在screen环境额外打印一行进度
                    print(f"  进度: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"准确率: {current_avg_acc:.3f}")
            else:
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_type[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })

            # 保存详细结果
            results.append({
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "filename": sample.get("filename", ""),
                "task_type": task_type,
                "path": sample.get("path", ""),
                "duration": sample.get("duration", 0),
                "question": sample.get("question", ""),
                "choice_a": sample.get("choice_a", ""),
                "choice_b": sample.get("choice_b", ""),
                "choice_c": sample.get("choice_c", ""),
                "choice_d": sample.get("choice_d", ""),
                "ground_truth_choice": ground_truth_choice,
                "predicted_choice": predicted_choice,
                "accuracy": accuracy,
                "is_correct": is_correct,
                "response_text": resp,
                "original_events": sample.get("original_events", []),
                "metrics_detail": metrics,
                "audio_token_length": audio_token_length
            })

            # 5️⃣ 收集 timing 信息
            if idx > 0:  # 跳过第一个样本的时间统计
                timing_stats.add_record(
                    prefill_time, decode_time, 
                    output_tokens,
                    input_tokens,
                    sample.get("duration", 0),
                    task_type,
                    current_peak_memory / (1024**3) if current_peak_memory else 0.0  # 转换为GB
                )

            # 内存清理
            if 'inputs' in locals():
                del inputs
            if 'out_ids' in locals():
                del out_ids
            
            # 清理音频处理过程中的中间变量
            if 'audio' in locals():
                del audio
            
            torch.cuda.empty_cache()
            
            # 每10个样本进行一次深度清理
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
                # 每100个样本打印内存使用情况
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [样本 {idx+1}] GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"推理错误: {e}")
            traceback.print_exc()
            resp = "ERROR"
            predicted_choice = "error"
            accuracy = 0.0
            is_correct = False
            prefill_time = 0.0
            decode_time = 0.0
            output_tokens = 0
            audio_token_length = 0
            input_tokens = 0  # 添加input_tokens初始化
            
            # 清理可能的中间变量
            if 'audio' in locals():
                del audio
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'out_ids' in locals():
                del out_ids
            torch.cuda.empty_cache()
            continue

    # 计算最终统计
    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0

    # 计算F1 Score（对于多分类任务，计算宏平均F1）
    # 收集所有可能的选择（A, B, C, D）
    all_choices = ['A', 'B', 'C', 'D']
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for choice in all_choices:
        # 计算每个选择的 TP, FP, FN
        tp = sum(1 for r in results if r["predicted_choice"] == choice and r["ground_truth_choice"] == choice)
        fp = sum(1 for r in results if r["predicted_choice"] == choice and r["ground_truth_choice"] != choice)
        fn = sum(1 for r in results if r["predicted_choice"] != choice and r["ground_truth_choice"] == choice)
        
        # 计算精度、召回率和F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if tp > 0 or fp > 0 or fn > 0:  # 只计算有数据的选择
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    # 计算宏平均F1分数
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    macro_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    # 计算内存统计摘要
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    
    # 获取初始内存和峰值内存的GB值
    initial_memory_gb = initial_memory / (1024**3) if initial_memory else 0
    avg_peak_memory_gb = overall_summary.get("avg_peak_memory_gb", 0)

    # 按任务类型统计性能
    task_type_accuracies = {}
    for task_type in task_type_stats.keys():
        if task_type_total[task_type] > 0:
            task_type_accuracies[task_type] = task_type_correct[task_type] / task_type_total[task_type]

    # 创建结果摘要
    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "macro_f1_score": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "task_type_stats": dict(task_type_stats),
        "task_type_accuracies": task_type_accuracies,
        "task_type_correct": dict(task_type_correct),
        "task_type_total": dict(task_type_total),
        "memory_stats": {
            "initial_memory_gb": initial_memory_gb,
            "avg_peak_memory_gb": avg_peak_memory_gb
        },
        "config": {
            "gpu_id": gpu_id,
            "compression_ratio": compression_ratio,
            "press_type": press_type,
            "sample_limit": sample_limit,
            "task_json_file": qa_json_file,
            "audio_base_dir": audio_base_dir,
            "kv_press_config": kv_press_config
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
    print(f"\n=== DESED声音事件检测评测结果摘要 (KV Press) ===")
    print(f"总样本数: {len(results)}")
    print(f"处理样本数: {processed_samples}")
    print(f"总体准确率: {final_accuracy:.3f}")
    print(f"宏平均F1分数: {macro_f1:.3f}")
    print(f"宏平均精度: {macro_precision:.3f}")
    print(f"宏平均召回率: {macro_recall:.3f}")
    print(f"任务类型数量: {len(task_type_stats)}")
    
    print(f"\n各任务类型准确率:")
    for task_type, acc in task_type_accuracies.items():
        correct_num = task_type_correct[task_type]
        total_num = task_type_total[task_type]
        print(f"  {task_type}: {acc:.3f} ({correct_num}/{total_num})")
    
    print(f"\n=== 内存统计 ===")
    print(f"初始内存: {initial_memory_gb:.2f}GB")
    print(f"平均峰值内存: {avg_peak_memory_gb:.2f}GB")
    
    print(f"\n=== 推理时间统计 ===")
    print(f"平均推理时间: {overall_summary.get('avg_total_time', 0):.4f}秒 (排除第一个样本)")
    print(f"平均Prefill时间: {overall_summary.get('avg_prefill_time', 0):.4f}秒 (排除第一个样本)")
    print(f"平均Decode时间: {overall_summary.get('avg_decode_time', 0):.4f}秒 (排除第一个样本)")
    print(f"平均吞吐量: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/秒")
    
    print(f"\n=== KV Press配置摘要 ===")
    print(f"Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"模型补丁状态: {'成功' if patch_success else '失败'}")
    
    print(f"结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

if __name__ == "__main__":
    main()
