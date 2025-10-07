import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
import gc
import re
import traceback
import subprocess
import tempfile
import warnings
import contextlib
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
from scipy.io import wavfile
from scipy import signal
import torch.nn.functional as F

# 添加sklearn导入（如果可用）
try:
    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
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

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，确保JSON兼容性"""
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

# 环境配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 添加 KV Press 配置
kv_press_config = {
    "compression_ratio": 0.5,         # 压缩比率
    "head_dims": None,                # 会在运行时设置
    "num_attention_heads": None,      # 会在运行时设置
    "press_type": "knorm",            # 默认压缩类型: knorm
    "return_indices": True,           # 是否返回保留的索引，用于调试
    "min_seq_len": 128,               # 最小序列长度，低于此长度不压缩
    "model_kwargs": {
        "attn_implementation": "sdpa",  # 使用 sdpa 实现而不是 flash attention
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False
    }
}

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
    
    def forward_hook(self, module, input_args, kwargs, layer_idx):
        """修复版本的forward hook，处理position_embeddings参数"""
        try:
            # 从kwargs中提取参数
            hidden_states = kwargs.get('hidden_states', input_args[0] if input_args else None)
            attention_mask = kwargs.get('attention_mask', input_args[1] if len(input_args) > 1 else None)
            
            # 检查和修复position_ids参数
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                position_ids = kwargs['position_ids']
                if isinstance(position_ids, (list, tuple)):
                    position_ids = torch.tensor(position_ids, device=hidden_states.device)
                elif not isinstance(position_ids, torch.Tensor):
                    position_ids = torch.tensor([position_ids], device=hidden_states.device)
                kwargs['position_ids'] = position_ids
            
            # 移除不支持的position_embeddings参数
            if 'position_embeddings' in kwargs:
                del kwargs['position_embeddings']
            
            # 对input_args也做同样处理
            if len(input_args) > 2:
                position_ids = input_args[2]
                if position_ids is not None:
                    if isinstance(position_ids, (list, tuple)):
                        position_ids = torch.tensor(position_ids, device=hidden_states.device)
                    elif not isinstance(position_ids, torch.Tensor):
                        position_ids = torch.tensor([position_ids], device=hidden_states.device)
                    input_args = tuple(list(input_args[:2]) + [position_ids] + list(input_args[3:]))
            
            # 调用父类的forward_hook
            result = super().forward_hook(module, input_args, layer_idx, **kwargs)
            
            return result
            
        except Exception as e:
            print(f"CustomTOVAPress forward_hook出错: {e}")
            # 如果出错，直接调用原始模块
            try:
                if hasattr(module, '_original_forward'):
                    return module._original_forward(*input_args, **kwargs)
                else:
                    return module(*input_args, **kwargs)
            except:
                # 最后的安全措施
                return input_args[0] if input_args else kwargs.get('hidden_states')

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

def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class TimingStats:
    """用于记录和分析prefill和decode阶段的时间统计，支持CUDA Events和GPU内存监控"""
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
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
                   input_tokens, output_tokens, audio_length=None, gpu_memory_peak=None, task_type=None):
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
            "total_time": total_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_length": audio_length,
            "task_type": task_type,
            "gpu_memory_current": current_memory / 1024**3 if self.cuda_available else 0,  # GB
            "gpu_memory_peak": gpu_memory_peak / 1024**3 if gpu_memory_peak else 0,  # GB
        }
        self.timing_records.append(record)
        
        if task_type:
            self.task_type_stats[task_type].append(record)
    
    def get_summary(self):
        """获取总体统计摘要"""
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
            "decode_percentage": (df["decode_time"].sum() / df["total_time"].sum()) * 100,
            "gpu_memory_stats": {
                "initial_memory_gb": self.initial_memory / 1024**3 if self.cuda_available else 0,
                "peak_memory_gb": self.total_peak_memory / 1024**3 if self.cuda_available else 0,
                "avg_current_memory_gb": df["gpu_memory_current"].mean() if self.cuda_available else 0,
                "max_peak_memory_gb": df["gpu_memory_peak"].max() if self.cuda_available else 0,
            }
        }
        
        # 添加任务类型统计
        task_summaries = {}
        for task_type, records in self.task_type_stats.items():
            if len(records) > 0:
                task_df = pd.DataFrame(records)
                task_summaries[task_type] = {
                    "samples": len(records),
                    "avg_prefill_time": task_df["prefill_time"].mean(),
                    "avg_decode_time": task_df["decode_time"].mean(),
                    "avg_total_time": task_df["total_time"].mean(),
                    "avg_tokens_per_sec": task_df["decode_tokens_per_sec"].mean()
                }
        
        return {
            "overall_summary": summary,
            "task_summaries": task_summaries
        }

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

def load_slue_dataset(json_file, audio_base_dir):
    """
    从JSON文件加载SLUE任务数据
    
    Args:
        json_file: SLUE格式JSON任务文件路径
        audio_base_dir: 音频文件基目录
    
    Returns:
        dataset: 包含任务数据的列表
    """
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"错误: JSON文件不存在: {json_file}")
        return []
    
    print(f"加载SLUE JSON文件: {json_file}")
    print(f"音频基目录: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return []
    
    if not isinstance(data, list):
        print(f"错误: JSON文件格式不正确，期望列表格式")
        return []
    
    print(f"从JSON加载了 {len(data)} 个任务")
    
    # 统计任务类型
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(data):
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
        task_name = task.get("task_name", "unknown")
        dataset_name = task.get("dataset_name", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        
        # 获取选择题选项
        choice_a = task.get("choice_a", "")
        choice_b = task.get("choice_b", "")
        choice_c = task.get("choice_c", "")
        choice_d = task.get("choice_d", "")
        
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
            "task_name": task_name,
            "dataset_name": dataset_name,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "entity_count": task.get("entity_count", 0),
            "entity_types": task.get("entity_types", []),
            "source_count": task.get("source_count", 0),
            "audio_duration_info": task.get("audio_duration_info", ""),
            "source_folder": task.get("source_folder", ""),
            "source_file": task.get("source_file", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1
    
    if missing_files > 5:
        print(f"警告: 共有 {missing_files} 个音频文件不存在")
    
    print(f"加载了 {len(dataset)} 个有效样本")
    print(f"任务类型统计: {dict(task_type_stats)}")
    print(f"数据集统计: {dict(dataset_stats)}")
    return dataset

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """按照参考代码的方式正确处理音频文件"""
    
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

def create_slue_prompt(doc):
    """为SLUE格式任务生成提示词"""
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
    """从模型响应中提取答案选择（A、B、C、D），过滤system prompt信息"""
    if not response:
        return ""
    
    # 移除常见的system prompt模式
    response = re.sub(r'^.*?(?:system|assistant|user).*?:\s*', '', response, flags=re.IGNORECASE | re.MULTILINE)
    response = re.sub(r'^.*?(?:Answer|Response|Output).*?:\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'^\s*<?/?s?>\s*', '', response)
    
    # 清理响应
    response = response.strip().upper()
    
    # 优先匹配明确的选项格式
    option_patterns = [
        r'(?:选择|答案|answer|choice|option)?\s*[：:]\s*([ABCD])',
        r'([ABCD])[).]',
        r'([ABCD])\s*[：:]',
        r'(?:选项|option|choice)\s*([ABCD])',
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 直接匹配单个字母
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    # 匹配包含字母的响应
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    
    # 匹配选项格式 (如 "A.", "A)", "(A)")
    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)
    
    # 如果没有找到明确的选择，返回空字符串
    return ""

def evaluate_slue_accuracy(predicted_choice, ground_truth_choice):
    """评估SLUE任务准确性"""
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
        print(f"评估SLUE准确性时出错: {e}")
        return {"accuracy": 0.0, "predicted_choice": "", "ground_truth_choice": gt, "is_correct": False}

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

    # 数据路径配置
    slue_json_file = "/data/hepeize05/Audio_Longbench/Dataset/SLUE/merged_audio_data.json"
    audio_base_dir = "/data/hepeize05/Audio_Longbench/Dataset/SLUE"
    
    print(f"SLUE JSON文件: {slue_json_file}")
    print(f"音频基目录: {audio_base_dir}")
    
    # 加载SLUE数据集
    samples = load_slue_dataset(slue_json_file, audio_base_dir)
    
    result_dir = os.environ.get("RESULTS_DIR", './SLUE_Results')
    os.makedirs(result_dir, exist_ok=True)

    # 修改输出文件路径和命名
    output_file = f'{result_dir}/slue_results_gpu{gpu_id}_kvpress_{press_type}_ratio{compression_ratio}.json'
    timing_output_file = f'{result_dir}/slue_timing_stats_gpu{gpu_id}_kvpress_{press_type}_ratio{compression_ratio}.json'
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")

    # 音频特殊token ID
    _AUDIO_SPECIAL_TOKEN_ID = 200011

    print(f"\n=== SLUE NER任务评测配置 ===")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press 压缩比率: {compression_ratio}")
    print(f"KV Press 压缩类型: {press_type}")
    print(f"SLUE JSON文件: {slue_json_file}")
    print(f"音频基目录: {audio_base_dir}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 40)

    # 加载模型
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
    generation_config = GenerationConfig.from_pretrained(model_path)
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
    
    # 确保模型完全加载到GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"使用数据集: {len(samples)} 个样本")
    
    # 应用样本限制
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"应用样本限制，处理 {len(samples)} 个样本")

    # 统计任务类型数量
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    for sample in samples:
        task_name = sample.get("task_name", "unknown")
        dataset_name = sample.get("dataset_name", "unknown")
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1
    
    print(f"任务类型统计: {dict(task_type_stats)}")
    print(f"数据集统计: {dict(dataset_stats)}")

    results = []
    total_accuracy = 0
    processed_samples = 0
    
    task_type_correct = defaultdict(int)
    task_type_total = defaultdict(int)
    dataset_correct = defaultdict(int)
    dataset_total = defaultdict(int)

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
    
    # 打印初始内存使用情况
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 记录初始内存用于后续计算
    timing_stats.record_initial_memory()  # 记录初始GPU内存
    
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="SLUE评估", **tqdm_kwargs)

    for idx, sample in progress_bar:
        try:
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
                    "task_name": sample.get("task_name", "unknown"),
                    "dataset_name": sample.get("dataset_name", "unknown"),
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
            
            # 获取参考答案和任务信息
            ground_truth_choice = sample.get("answer_gt", "")
            task_name = sample.get("task_name", "unknown")
            dataset_name = sample.get("dataset_name", "unknown")
            
            # 创建SLUE提示词
            prompt = create_slue_prompt(sample)

            # 重置峰值内存统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # 使用Pipeline进行推理，支持KV Press压缩和CUDA Events精确计时
            try:
                result = pipeline(
                    prompt=prompt,
                    audios=audio,
                    press=press,
                    max_new_tokens=20,  # SLUE任务需要更多tokens处理NER
                    do_sample=False,
                    measure_time=True
                )
                
                # 初始化默认值，避免变量引用错误
                resp = result['text']
                prefill_time = 0.0
                decode_time = 0.0
                output_tokens = 0
                
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
                
                # 从pipeline结果中获取峰值内存
                final_peak_memory = result.get('metrics', {}).get('peak_memory_gb', 0) * (1024**3)  # 转换为字节
                
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
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        generation_config=generation_config,
                        do_sample=False,
                        return_dict_in_generate=True
                    )
                end_time = time.time()
                
                tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                resp = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                prefill_time = 0.0
                decode_time = end_time - start_time
                final_peak_memory = 0

            # 解析输出
            predicted_choice = extract_answer_choice(resp)

            # 计算准确性
            metrics = evaluate_slue_accuracy(predicted_choice, ground_truth_choice)
            
            accuracy = metrics["accuracy"]
            is_correct = metrics["is_correct"]
            
            total_accuracy += accuracy
            processed_samples += 1

            # 更新任务类型和数据集统计
            task_type_total[task_name] += 1
            dataset_total[dataset_name] += 1
            if is_correct:
                task_type_correct[task_name] += 1
                dataset_correct[dataset_name] += 1

            current_avg_acc = total_accuracy / processed_samples
            
            # 在screen环境下每10个样本更新一次，在标准环境下每个样本都更新
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                progress_bar.set_postfix({
                    'Acc': f'{current_avg_acc:.3f}',
                    'Task': task_name[:10],
                    'Dataset': dataset_name[:10],
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
                    'Task': task_name[:10],
                    'Dataset': dataset_name[:10],
                    'Pred': predicted_choice,
                    'GT': ground_truth_choice
                })

            # 保存详细结果
            results.append({
                "idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "filename": sample.get("filename", ""),
                "task_name": task_name,
                "dataset_name": dataset_name,
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
                "entity_count": sample.get("entity_count", 0),
                "entity_types": sample.get("entity_types", []),
                "source_count": sample.get("source_count", 0),
                "metrics_detail": metrics,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "kv_press_enabled": True,
                "compression_ratio": compression_ratio,
                "press_type": press_type
            })

            # 收集 timing 信息
            timing_stats.add_record(
                prefill_time + decode_time,
                prefill_time, 
                decode_time, 
                inputs["input_ids"].shape[1] if 'inputs' in locals() and inputs is not None else 0,
                output_tokens,
                sample.get("duration", 0),
                final_peak_memory,
                task_name
            )

            # 内存清理 - 使用torch.cuda.synchronize()确保同步
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            
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
            full_generation_time = 0.0
            final_peak_memory = 0
            
            # 清理可能的中间变量
            if 'audio' in locals():
                del audio
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            continue

    # 计算最终统计
    final_accuracy = total_accuracy / processed_samples if processed_samples > 0 else 0.0

    # 计算F1 Score (多分类)
    if processed_samples > 0 and results:
        # 获取预测值和真实值
        y_true = []
        y_pred = []
        
        for result in results:
            if 'predicted_choice' in result and 'ground_truth_choice' in result:
                y_true.append(result['ground_truth_choice'])
                y_pred.append(result['predicted_choice'])
        
        # 计算macro和weighted F1 Score
        try:
            if SKLEARN_AVAILABLE and len(y_true) > 0:
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                
                # 计算每个类别的F1 Score
                labels = ['A', 'B', 'C', 'D']
                individual_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                individual_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                individual_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                
                class_metrics = {}
                for i, label in enumerate(labels):
                    class_metrics[f'class_{label}'] = {
                        'f1_score': float(individual_f1[i]) if i < len(individual_f1) else 0.0,
                        'precision': float(individual_precision[i]) if i < len(individual_precision) else 0.0,
                        'recall': float(individual_recall[i]) if i < len(individual_recall) else 0.0
                    }
            else:
                # Manual F1 score calculation if sklearn is not available
                from collections import Counter
                
                labels = ['A', 'B', 'C', 'D']
                class_metrics = {}
                f1_scores = []
                precisions = []
                recalls = []
                
                for label in labels:
                    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
                    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
                    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_metrics[f'class_{label}'] = {
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall
                    }
                    
                    f1_scores.append(f1)
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Macro averages
                macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
                macro_precision = sum(precisions) / len(precisions) if precisions else 0
                macro_recall = sum(recalls) / len(recalls) if recalls else 0
                
                # Weighted F1 (approximation using accuracy for simplicity)
                weighted_f1 = final_accuracy
            
        except Exception as e:
            print(f"F1 Score计算失败: {e}")
            macro_f1 = weighted_f1 = macro_precision = macro_recall = 0
            class_metrics = {}
    else:
        macro_f1 = weighted_f1 = macro_precision = macro_recall = 0
        class_metrics = {}

    # 按任务类型统计性能
    task_type_accuracies = {}
    for task_name in task_type_stats.keys():
        if task_type_total[task_name] > 0:
            task_type_accuracies[task_name] = task_type_correct[task_name] / task_type_total[task_name]

    # 按数据集统计性能
    dataset_accuracies = {}
    for dataset_name in dataset_stats.keys():
        if dataset_total[dataset_name] > 0:
            dataset_accuracies[dataset_name] = dataset_correct[dataset_name] / dataset_total[dataset_name]

    # 创建结果摘要
    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_accuracy": final_accuracy,
        "macro_f1_score": macro_f1,
        "weighted_f1_score": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "class_metrics": class_metrics,
        "task_type_stats": dict(task_type_stats),
        "dataset_stats": dict(dataset_stats),
        "task_type_accuracies": task_type_accuracies,
        "dataset_accuracies": dataset_accuracies,
        "task_type_correct": dict(task_type_correct),
        "task_type_total": dict(task_type_total),
        "dataset_correct": dict(dataset_correct),
        "dataset_total": dict(dataset_total),
        "config": {
            "gpu_id": gpu_id,
            "compression_ratio": compression_ratio,
            "press_type": press_type,
            "sample_limit": sample_limit,
            "slue_json_file": slue_json_file,
            "audio_base_dir": audio_base_dir,
            "kv_press_config": kv_press_config
        },
        "timing": timing_stats.get_summary()
    }

    # 保存结果 - 转换numpy类型
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    # 转换numpy类型为Python原生类型
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # 保存时间统计，包含KV Press配置
    timing_summary = timing_stats.get_summary()
    
    # 转换numpy类型为Python原生类型
    timing_data = {
        "summary": timing_summary,
        "detailed_records": timing_stats.timing_records,
        "kv_press_config": kv_press_config
    }
    timing_data = convert_numpy_types(timing_data)
    
    with open(timing_output_file, "w", encoding="utf-8") as f:
        json.dump(timing_data, f, indent=2, ensure_ascii=False)

    # 输出结果摘要
    print(f"\n=== SLUE NER任务评测结果摘要 ===")
    print(f"总样本数: {len(results)}")
    print(f"处理样本数: {processed_samples}")
    print(f"总体准确率: {final_accuracy:.3f}")
    print(f"任务类型数量: {len(task_type_stats)}")
    print(f"数据集数量: {len(dataset_stats)}")
    
    # 显示F1 Score结果
    print(f"\n=== F1 Score 指标 ===")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    
    if class_metrics:
        print(f"\n=== 各类别指标 ===")
        for class_name, metrics in class_metrics.items():
            print(f"{class_name.upper()}: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    
    print(f"\n各任务类型准确率:")
    for task_name, acc in task_type_accuracies.items():
        correct_num = task_type_correct[task_name]
        total_num = task_type_total[task_name]
        print(f"  {task_name}: {acc:.3f} ({correct_num}/{total_num})")
    
    print(f"\n各数据集准确率:")
    for dataset_name, acc in dataset_accuracies.items():
        correct_num = dataset_correct[dataset_name]
        total_num = dataset_total[dataset_name]
        print(f"  {dataset_name}: {acc:.3f} ({correct_num}/{total_num})")
    
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== KV Press配置 ===")
    print(f"压缩类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"最小序列长度: {kv_press_config['min_seq_len']}")
    
    print(f"\n=== 时间统计（CUDA Events精确测量）===")
    print(f"平均推理时间: {overall_summary.get('avg_total_time', 0):.4f}秒")
    print(f"平均Prefill时间: {overall_summary.get('avg_prefill_time', 0):.4f}秒")
    print(f"平均Decode时间: {overall_summary.get('avg_decode_time', 0):.4f}秒")
    print(f"平均吞吐量: {overall_summary.get('avg_decode_tokens_per_sec', 0):.2f} tokens/秒")
    
    # 打印GPU内存统计
    if 'gpu_memory_stats' in overall_summary:
        gpu_stats = overall_summary['gpu_memory_stats']
        print("\n===== GPU内存统计 =====")
        print(f"初始GPU内存: {gpu_stats['initial_memory_gb']:.2f} GB")
        print(f"峰值GPU内存: {gpu_stats['peak_memory_gb']:.2f} GB")
        print(f"平均当前内存: {gpu_stats['avg_current_memory_gb']:.2f} GB")
        print(f"最大峰值内存: {gpu_stats['max_peak_memory_gb']:.2f} GB")
    
    print(f"结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

if __name__ == "__main__":
    main()
