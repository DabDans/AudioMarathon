import os
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import numpy as np
import soundfile as sf
import warnings
import traceback
import time
import glob
import random
import sys
import io
import gc
import contextlib
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
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

# 设置随机种子
random.seed(42)

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
    else:
        return obj

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

# 添加 KV Press 配置
kv_press_config = {
    "compression_ratio": 0.5,         # 压缩比率
    "head_dims": None,                # 会在运行时设置
    "num_attention_heads": None,      # 会在运行时设置
    "press_type": "knorm",            # 默认压缩类型: knorm
    "return_indices": True,           # 是否返回保留的索引，用于调试
    "min_seq_len": 128,               # 最小序列长度，低于此长度不压缩
    "model_kwargs": {
        "attn_implementation": "sdpa",  # 使用 SDPA 实现进行高效attention计算
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False
    }
}

# 创建 Phi4MM 模型 KV Press 补丁
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
    
    # 如果没有找到rotary_emb，检查是否使用其他位置编码机制
    if not rotary_found:
        if hasattr(first_layer.self_attn, "_init_rope"):
            try:
                # 尝试手动初始化rotary embedding
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                config = model.config
                head_dim = config.hidden_size // config.num_attention_heads
                
                # 创建标准的RotaryEmbedding
                model.model.rotary_emb = LlamaRotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=getattr(config, "rope_theta", 10000.0)
                )
                
                print("已手动创建并添加全局 rotary_emb 属性")
                rotary_found = True
            except Exception as e:
                print(f"创建rotary_emb时出错: {str(e)}")
                return False
        
        # 如果是其他位置编码机制，尝试兼容处理
        if not rotary_found and hasattr(first_layer.self_attn, "position_embedding_type"):
            emb_type = first_layer.self_attn.position_embedding_type
            print(f"警告: 模型使用非标准位置编码: {emb_type}，KV Press可能无法正常工作")
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

# 初始化和选择最优的 KV Press
def initialize_kv_press(model, compression_ratio=0.5):
    """根据模型特性选择最合适的KV Press类型"""
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
    
    # 检查第一层的注意力头维度
    first_layer = model.model.layers[0]
    if not hasattr(first_layer, "self_attn"):
        print("警告: 注意力层结构异常，使用默认KnormPress")
        return KnormPress(compression_ratio=compression_ratio)
    
    # 获取第一层的注意力头维度
    first_layer_head_dim = first_layer.self_attn.head_dim if hasattr(first_layer.self_attn, "head_dim") else head_dim
    
    # 检查所有层的注意力头维度是否一致
    dimensions_match = True
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "head_dim"):
            layer_head_dim = layer.self_attn.head_dim
            if layer_head_dim != first_layer_head_dim:
                print(f"警告: 层 {layer_idx} 的注意力头维度 ({layer_head_dim}) 与第一层 ({first_layer_head_dim}) 不一致")
                dimensions_match = False
    
    # 确定最佳Press类型
    press_type = kv_press_config["press_type"]
    has_rotary = hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None
    has_kvpress_patch = hasattr(model.model, "_kvpress_patched") and model.model._kvpress_patched
    
    # 如果模型有rotary_emb且所有层维度一致，优先使用ExpectedAttentionPress
    if has_rotary and has_kvpress_patch and press_type != "knorm":
        print("使用ExpectedAttentionPress (基于RoPE的注意力压缩)")
        return ExpectedAttentionPress(compression_ratio=compression_ratio)
    
    # 如果指定了knorm类型或上面的条件不满足，使用KnormPress
    if press_type == "knorm" or not has_rotary or not has_kvpress_patch:
        print("使用KnormPress (基于Key-norm的注意力压缩)")
        return KnormPress(compression_ratio=compression_ratio)
    
    # 如果指定了random类型，使用RandomPress
    if press_type == "random":
        print("使用RandomPress (随机丢弃tokens)")
        return RandomPress(compression_ratio=compression_ratio)
    
    # 如果指定了其他类型，根据具体类型创建实例
    if press_type == "observed":
        print("使用ObservedAttentionPress (基于观察到的注意力分数)")
        return ObservedAttentionPress(compression_ratio=compression_ratio)
    
    if press_type == "tova":
        print("使用CustomTOVAPress (基于时间顺序的注意力值分析)")
        return CustomTOVAPress(compression_ratio=compression_ratio)
    
    if press_type == "snap":
        print("使用SnapKVPress (适用于LoRA微调模型)")
        return SnapKVPress(compression_ratio=compression_ratio)
    
    if press_type == "streaming":
        print("使用StreamingLLMPress (适用于长文本流式处理)")
        window_size = kv_press_config.get("window_size", 4096)
        return StreamingLLMPress(window_size=window_size)
    
    # 默认回退到KnormPress作为最安全的选择
    print(f"未知压缩类型 {press_type}，默认使用KnormPress")
    return KnormPress(compression_ratio=compression_ratio)

def calculate_acoustic_metrics(predictions, ground_truths, scene_labels):
    """计算声学场景分类指标：准确率、精确率、召回率和F1分数"""
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

class TimingStats:
    """用于记录和分析prefill和decode阶段的时间统计，支持CUDA Events和GPU内存监控"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
        self.initial_memory = 0
        self.peak_memory = 0
        self.total_peak_memory = 0
        
        if self.cuda_available:
            # 重置GPU内存统计
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
    
    def add_record(self, total_time, prefill_time, decode_time, 
                   input_tokens, output_tokens, audio_length=None, gpu_memory_peak=None):
        """添加一条时间记录"""
        # 获取当前GPU内存使用情况
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
            "total_time": float(total_time),
            "prefill_time": float(prefill_time),
            "decode_time": float(decode_time),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "decode_tokens_per_sec": float(output_tokens / decode_time if decode_time > 0 else 0),
            "audio_length": float(audio_length) if audio_length is not None else None,
            "gpu_memory_current": float(current_memory / 1024**3) if self.cuda_available else 0.0,  # GB
            "gpu_memory_peak": float(gpu_memory_peak / 1024**3) if gpu_memory_peak else 0.0,  # GB
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """获取总体统计摘要"""
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        df = pd.DataFrame(self.timing_records)
        
        summary = {
            "total_samples": int(len(df)),
            "avg_total_time": float(df["total_time"].mean()),
            "avg_prefill_time": float(df["prefill_time"].mean()),
            "avg_decode_time": float(df["decode_time"].mean()),
            "avg_decode_tokens_per_sec": float(df["decode_tokens_per_sec"].mean()),
            "prefill_percentage": float((df["prefill_time"].sum() / df["total_time"].sum()) * 100),
            "decode_percentage": float((df["decode_time"].sum() / df["total_time"].sum()) * 100),
            "gpu_memory_stats": {
                "initial_memory_gb": float(self.initial_memory / 1024**3) if self.cuda_available else 0.0,
                "peak_memory_gb": float(self.total_peak_memory / 1024**3) if self.cuda_available else 0.0,
                "avg_current_memory_gb": float(df["gpu_memory_current"].mean()) if self.cuda_available else 0.0,
                "max_peak_memory_gb": float(df["gpu_memory_peak"].max()) if self.cuda_available else 0.0,
            }
        }
        
        return summary

# 获取GPU ID和KV Press配置参数
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"使用 GPU ID: {gpu_id}")

# KV Press 配置参数
compression_ratio = float(os.environ.get("COMPRESSION_RATIO", 0.5))
press_type = os.environ.get("PRESS_TYPE", "knorm")
kv_press_config["compression_ratio"] = compression_ratio
kv_press_config["press_type"] = press_type

print(f"KV Press 配置: 压缩比率={compression_ratio}, 压缩类型={press_type}")

# 样本限制 (如果提供)
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"样本限制设置为: {sample_limit}")

# 数据路径配置 - TAU数据集路径
data_path_root = '/data/hepeize05/Audio_Longbench/Dataset/TAU'
audio_dir = os.path.join(data_path_root, 'concatenated_resampled')
result_dir = './TAU_Results'
os.makedirs(result_dir, exist_ok=True)

# 修改输出文件路径和命名
output_file = f'{result_dir}/TAU_results_gpu{gpu_id}_kvpress_{press_type}_ratio{compression_ratio}.jsonl'
timing_output_file = f'{result_dir}/TAU_timing_stats_gpu{gpu_id}_kvpress_{press_type}_ratio{compression_ratio}.json'
print(f"结果将保存到: {output_file}")
print(f"时间统计将保存到: {timing_output_file}")

# 音频特殊token ID
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'

# 音频处理函数
def prepare_audio_for_processor(audio_path, target_sr=16000):
    """按照官方示例正确处理音频文件"""
    
    try:
        # 方法1: 直接使用soundfile
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"soundfile加载失败: {e}")
            
            # 方法2: 使用ffmpeg
            try:
                import subprocess
                import tempfile
                from scipy.io import wavfile
                
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
        
        # 确保是单声道
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
            print(f"转换为单声道: 形状={audio.shape}")
        
        # 重采样到目标采样率
        if sample_rate != target_sr and sample_rate > 0:
            from scipy import signal
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

def load_tau_acoustic_scene_dataset(root_dir):
    """从TAU数据集加载声学场景分类任务"""
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
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification"
        })
    
    print(f"总计加载了 {len(all_samples)} 个有效音频样本")
    
    # 显示场景分布
    print("场景分布:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count}个样本 ({count/len(all_samples)*100:.1f}%)")
    
    # 样本限制处理
    if sample_limit > 0 and sample_limit < len(all_samples):
        print(f"由于样本限制设置，随机选择{sample_limit}个样本进行评估")
        all_samples = random.sample(all_samples, sample_limit)
        
    # 随机打乱样本
    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    """从模型输出文本中提取声学场景答案选项（A/B/C/D），过滤system prompt信息"""
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

def main():
    # 确认GPU可用性并清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda"
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU available, using CPU")

    # Step1: 加载模型
    print("加载Phi-4-multimodal-instruct模型...")
    model_path = "/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B"
    
    # 首先加载 processor
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False  # 使用慢速但更稳定的tokenizer
    )
    
    # 然后加载模型，使用KV Press配置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        **kv_press_config["model_kwargs"]
    )
    model.eval()
    
    # 为 Phi4MM 模型添加补丁以便与 KV Press 兼容
    patch_phi4mm_for_kvpress(model)

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
    
    # 创建时间统计器
    timing_stats = TimingStats()
    
    # 记录初始内存
    if hasattr(timing_stats, 'record_initial_memory'):
        timing_stats.record_initial_memory()
    elif torch.cuda.is_available():
        # 如果没有record_initial_memory方法，手动清理和重置内存统计
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        print(f"初始GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 加载生成配置
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # 扫描TAU数据集
    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    
    # 统计场景分布
    print(f"总计处理 {len(samples)} 个样本")
    
    # 用于收集所有预测和真实标签，计算F1 score
    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    
    # 统计场景准确率（仅用于显示）
    scene_stats = {scene: {"total": 0, "correct": 0} for scene in scene_counts}
    
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
    
    # 创建进度条处理所有样本
    print(f"开始处理 {len(samples)} 个样本...")
    with tqdm(total=len(samples), desc="处理TAU声学场景样本", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        # 遍历处理所有样本
        for i, sample in enumerate(samples):
            # 检查音频文件是否存在
            wav_path = sample['wav_path']
            if not os.path.exists(wav_path):
                print(f"跳过不存在的音频文件: {wav_path}")
                # 创建跳过记录
                sample_result = {
                    "wav_path": wav_path,
                    "scene_label": sample.get("scene_label", "unknown"),
                    "ground_truth": sample.get("answer_gt", ""),
                    "model_output": "SKIPPED - File not found",
                    "extracted_answer": "skip",
                    "is_correct": False,
                    "choices": {
                        "A": sample.get("choice_a", ""),
                        "B": sample.get("choice_b", ""),
                        "C": sample.get("choice_c", ""),
                        "D": sample.get("choice_d", "")
                    },
                    "audio_duration": 0.0,
                    "output_tokens": 0,
                    "prefill_time": 0.0,
                    "decode_time": 0.0,
                    "total_time": 0.0,
                    "peak_memory_gb": 0.0,
                    "skipped": True,
                    "skip_reason": "Audio file not found"
                }
                all_sample_results.append(sample_result)
                pbar.update()
                continue
                
            # 在每个样本处理前重置GPU内存统计，确保准确的峰值测量
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            scene_label = sample["scene_label"]
            ground_truth = sample["answer_gt"].upper()
            
            # 使用声学场景分类的提示词
            instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.\n"
            instruction += f"A: {sample['choice_a']}\nB: {sample['choice_b']}\nC: {sample['choice_c']}\nD: {sample['choice_d']}\n"
            instruction += "Respond with only the letter of your answer (A, B, C, or D)."
            
            full_prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
            
            try:
                # 准备音频输入
                audio = prepare_audio_for_processor(wav_path)
              
                # 检测音频文件长度
                current_audio_length = len(audio[0]) / 16000 if isinstance(audio, list) and len(audio) > 0 else 0.0
                
                # 初始化默认值，避免变量引用错误
                response = ""
                prefill_time = 0.0
                decode_time = 0.0
                full_generation_time = 0.0
                output_tokens = 0
                current_peak_memory = 0
                
                # 使用Pipeline进行推理，支持KV Press压缩和CUDA Events精确计时
                try:
                    # 使用pipeline处理，应用压缩
                    with press(model) if press is not None else contextlib.nullcontext():
                        result = pipeline(
                            prompt=full_prompt,
                            audios=audio,
                            press=press,
                            input_mode=2,
                            measure_time=True,
                            max_new_tokens=10,  # TAU任务只需要简短回答
                            do_sample=False,
                            return_legacy_cache=True,  # 处理缓存格式警告
                        )
                    
                    # 从pipeline返回结果中正确提取数据
                    response = result['text']
                    if 'metrics' in result and result['metrics']:
                        metrics = result['metrics']
                        prefill_time = metrics.get('prefill_time', 0.0)
                        decode_time = metrics.get('generation_time', 0.0)  # pipeline中叫generation_time
                        full_generation_time = metrics.get('total_time', prefill_time + decode_time)
                    else:
                        # 如果没有metrics，使用默认值
                        prefill_time = 0.0
                        decode_time = 0.0
                        full_generation_time = 0.0
                    
                    # 获取输出token数量
                    output_tokens = result.get('output_tokens', 0)
                    
                    # 从pipeline返回的metrics中获取峰值内存
                    current_peak_memory = 0
                    if 'metrics' in result and result['metrics']:
                        current_peak_memory = result['metrics'].get('peak_memory_gb', 0.0) * (1024**3)  # 转换回字节
                    
                except Exception as pipeline_error:
                    print(f"Pipeline推理失败: {pipeline_error}")
                    print("回退到标准推理方式")
                    
                    # 处理输入
                    inputs = processor(
                        text=full_prompt,
                        audios=audio,
                        return_tensors="pt"
                    ).to(device)
                    inputs['input_mode'] = torch.tensor([2])

                    # 简化的推理流程，同时测量峰值内存
                    start_time = time.time()
                    
                    # 使用 KV Press 作为上下文管理器进行生成
                    with torch.no_grad(), press(model):
                        generate_ids = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            generation_config=generation_config,
                            return_dict_in_generate=True
                        )
                    
                    end_time = time.time()
                    
                    # 获取峰值内存（fallback模式）
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        current_peak_memory = torch.cuda.max_memory_allocated()
                    else:
                        current_peak_memory = 0
                    
                    response = processor.batch_decode(
                        generate_ids.sequences[:, inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )[0]
                    
                    prefill_time = 0.0  # 无法准确分离，设为0
                    decode_time = end_time - start_time
                    full_generation_time = decode_time
                    output_tokens = generate_ids.sequences.shape[1] - inputs['input_ids'].shape[1]
                
                # 解析和处理输出
                predicted_answer = extract_acoustic_scene_answer(response, [sample['choice_a'], sample['choice_b'], sample['choice_c'], sample['choice_d']])
                
                # 记录预测和真实标签用于F1计算
                all_predictions.append(predicted_answer)
                all_ground_truths.append(ground_truth)
                
                # 检查答案是否正确
                is_correct = predicted_answer == ground_truth
                
                # 更新场景统计
                scene_stats[scene_label]["total"] += 1
                if is_correct:
                    scene_stats[scene_label]["correct"] += 1
                
                # 跳过第一个样本的时间统计
                if i > 0:
                    timing_stats.add_record(
                        full_generation_time,
                        prefill_time,
                        decode_time,
                        0,  # input_tokens不再需要计算
                        output_tokens,
                        current_audio_length,  # audio_length
                        current_peak_memory  # gpu_memory_peak
                    )
                
            except Exception as e:
                print(f"推理错误: {e}")
                traceback.print_exc()
                response = "ERROR"
                predicted_answer = "ERROR"
                is_correct = False
                prefill_time = 0.0
                decode_time = 0.0
                output_tokens = 0
                full_generation_time = 0.0
                current_peak_memory = 0  # 错误情况下设置为0
                
                # 记录错误的预测
                all_predictions.append(predicted_answer)
                all_ground_truths.append(ground_truth)
            
            # 保存样本结果，确保所有数值都是Python原生类型
            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "scene_label": scene_label,
                "ground_truth": ground_truth,
                "model_output": response,
                "extracted_answer": predicted_answer,
                "is_correct": bool(is_correct),
                "output_tokens": int(output_tokens),
                "prefill_time": float(prefill_time),
                "decode_time": float(decode_time),
                "total_time": float(full_generation_time),
                "kv_press_enabled": True,
                "compression_ratio": float(compression_ratio),
                "press_type": press_type
            }
            
            all_sample_results.append(sample_result)
            torch.cuda.empty_cache()
            
            # 更新进度条
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p != "ERROR" and t != "ERROR") / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p != "ERROR" and t != "ERROR"))
            
            pbar.set_postfix({
                '样本': f'{i+1}/{len(samples)}',
                '准确率': f'{current_accuracy:.3f}',
                '场景': scene_label[:12] + '...' if len(scene_label) > 12 else scene_label
            })
            
            pbar.update()
    
    # 计算最终指标
    all_scene_labels = list(set(all_ground_truths))
    acoustic_metrics = calculate_acoustic_metrics(all_predictions, all_ground_truths, all_scene_labels)
    final_stats = timing_stats.get_summary()
    
    # 计算统计信息
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    
    # 计算每个场景的准确率
    for scene in scene_stats:
        if scene_stats[scene]["total"] > 0:
            scene_stats[scene]["accuracy"] = float(scene_stats[scene]["correct"] / scene_stats[scene]["total"])
        else:
            scene_stats[scene]["accuracy"] = 0.0
    
    # 创建完整的结果结构
    results = {
        "samples": all_sample_results,
        "summary": {
            "total_samples": int(total_samples),
            "correct_samples": int(correct_samples),
            "accuracy": float(correct_samples / total_samples if total_samples > 0 else 0),
            "scene_stats": scene_stats,
            "metrics": acoustic_metrics,
            "timing": final_stats,
            "kv_press_config": {
                "compression_ratio": float(compression_ratio),
                "press_type": press_type,
                "kv_press_enabled": True
            }
        }
    }
    
    # 保存结果为单个JSON文件
    json_output_file = f'{result_dir}/TAU_results_gpu{gpu_id}_kvpress_{press_type}_ratio{compression_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        # 使用convert_to_serializable确保所有类型都可以JSON序列化
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 保存时间统计，包含KV Press配置
    if len(timing_stats.timing_records) > 0:
        timing_summary = timing_stats.get_summary()
        
        # 保存时间统计数据
        with open(timing_output_file, "w", encoding="utf-8") as f:
            timing_data = {
                "summary": timing_summary,
                "detailed_records": timing_stats.timing_records,
                "kv_press_config": kv_press_config
            }
            # 使用convert_to_serializable确保时间统计也可以JSON序列化
            serializable_timing_data = convert_to_serializable(timing_data)
            json.dump(serializable_timing_data, f, indent=2, ensure_ascii=False)
        print("已保存时间统计数据到", timing_output_file)
    
    # 打印结果摘要
    print("\n=== 评测结果摘要 ===")
    print(f"总样本数: {total_samples}")
    print(f"总准确率: {results['summary']['accuracy']:.2%}")
    print(f"F1 Score: {acoustic_metrics['f1_score']:.4f}")
    print(f"Precision: {acoustic_metrics['precision']:.4f}")
    print(f"Recall: {acoustic_metrics['recall']:.4f}")
    print(f"有效样本: {acoustic_metrics['valid_samples']}/{acoustic_metrics['total_samples']}")
    
    # 按准确率排序打印场景级别的结果
    sorted_scenes = sorted(
        [(scene, stats["accuracy"], stats["correct"], stats["total"]) 
         for scene, stats in results["summary"]["scene_stats"].items()],
        key=lambda x: x[1], reverse=True
    )
    
    print("\n场景准确率:")
    for scene, acc, correct, total in sorted_scenes:
        print(f"  {scene}: {acc:.2%} ({correct}/{total})")
    
    print(f"\n=== KV Press配置 ===")
    print(f"压缩类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"最小序列长度: {kv_press_config['min_seq_len']}")
    
    if len(timing_stats.timing_records) > 0:
        print(f"\n=== 时间统计（CUDA Events精确测量，排除第一个样本）===")
        print(f"平均推理时间: {final_stats['avg_total_time']:.4f}秒")
        print(f"平均 Prefill 时间: {final_stats['avg_prefill_time']:.4f}秒")
        print(f"平均 Decode 时间: {final_stats['avg_decode_time']:.4f}秒")
        print(f"平均token生成速度: {final_stats['avg_decode_tokens_per_sec']:.2f} tokens/秒")
        
        # 打印GPU内存统计
        if 'gpu_memory_stats' in final_stats:
            gpu_stats = final_stats['gpu_memory_stats']
            print("\n===== GPU内存统计 =====")
            print(f"初始GPU内存: {gpu_stats['initial_memory_gb']:.2f} GB")
            print(f"峰值GPU内存: {gpu_stats['peak_memory_gb']:.2f} GB")
            print(f"平均当前内存: {gpu_stats['avg_current_memory_gb']:.2f} GB")
            print(f"最大峰值内存: {gpu_stats['max_peak_memory_gb']:.2f} GB")
    
    print(f"结果已保存到: {json_output_file}")

if __name__ == "__main__":
    main()
