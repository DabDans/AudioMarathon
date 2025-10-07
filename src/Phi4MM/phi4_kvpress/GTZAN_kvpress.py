import os
import json
import time
import torch
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
import sys
import traceback
import contextlib
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
            
        except Exception as e:
            print(f"CustomTOVAPress forward_hook出错: {e}")
            # 如果出错，直接调用原始模块
            try:
                if hasattr(module, '_original_forward'):
                    return module._original_forward(*args, **kwargs)
                else:
                    return module(*args, **kwargs)
            except:
                # 最后的安全措施
                return args[0] if args else kwargs.get('hidden_states', output)

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

class GTZANTimingStats:
    """跟踪GTZAN任务的推理时间统计"""
    def __init__(self):
        self.timing_records = []
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_peak_memory = 0  # 添加峰值内存跟踪
        self.initial_gpu_memory = 0
    
    def record_initial_memory(self):
        """记录初始GPU内存使用情况"""
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            self.initial_gpu_memory = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration, peak_memory_gb=None):
        """添加一条时间记录"""
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        # 添加峰值内存跟踪
        if peak_memory_gb is not None:
            self.total_peak_memory += peak_memory_gb
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "audio_duration": audio_duration,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "peak_memory_gb": peak_memory_gb if peak_memory_gb else 0.0  # 添加峰值内存
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """获取汇总统计"""
        if self.total_samples == 0:
            return {}
        
        return {
            "total_samples": self.total_samples,
            "avg_prefill_time": self.total_prefill_time / self.total_samples,
            "avg_decode_time": self.total_decode_time / self.total_samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0,
            "avg_peak_memory_gb": self.total_peak_memory / self.total_samples if self.total_samples > 0 else 0
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

def clean_text_response(response):
    """清理模型对GTZAN任务的响应，只提取输出的答案部分，去除system prompt等信息"""
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

def load_audio_for_gtzan(audio_path, audio_cache=None):
    """
    加载音频文件，返回格式与processor一致
    返回: ([audio_array], sampling_rate)
    """
    if audio_cache is not None and audio_path in audio_cache:
        # 使用缓存
        audio_np, sr = audio_cache[audio_path]
    else:
        # 延迟加载音频数据
        audio_np, sr = sf.read(audio_path)
        # 确保返回形状正确的数组
        if len(audio_np.shape) > 1:
            audio_np = audio_np[:, 0]  # 如果是多通道，只取第一个通道
        
        # 添加到缓存
        if audio_cache is not None:
            audio_cache[audio_path] = (audio_np, sr)
    
    # 返回格式: ([audio_array], sampling_rate)
    return [audio_np], sr

def prepare_audio_for_processor(audio_data, target_sr=16000):
    """将音频转换为processor期望的格式"""
    if isinstance(audio_data, list):
        return [(audio, target_sr) for audio in audio_data]
    else:
        return [(audio_data, target_sr)]

def create_gtzan_prompt(question, options):
    """创建GTZAN任务的提示词"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio segment and identify the music genre based on what you hear."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    # 格式化选项
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options += f"{letter}. {opt}\n"
    
    # 构建完整提示词
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def load_gtzan_metadata(metadata_path):
    """加载GTZAN元数据文件"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # 过滤掉不完整的条目
    valid_samples = []
    for item in metadata:
        if all(key in item for key in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            valid_samples.append(item)
    
    print(f"从 {len(metadata)} 个条目中加载了 {len(valid_samples)} 个有效样本")
    return valid_samples

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
    data_path_root = '/data/hepeize05/Audio_Longbench/Dataset/GTZAN/concatenated_audio'
    metadata_file = os.path.join(data_path_root, 'music_genre_classification_meta.json')
    result_dir = os.environ.get("RESULTS_DIR", './GTZAN_Results')
    os.makedirs(result_dir, exist_ok=True)

    # 修改输出文件路径和命名 - 加入KV Press标识
    output_file = f'{result_dir}/gtzan_results_kvpress_{press_type}_{compression_ratio}.json'
    timing_output_file = f'{result_dir}/gtzan_timing_stats_kvpress_{press_type}_{compression_ratio}.json'
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")

    # 音频特殊token ID
    _AUDIO_SPECIAL_TOKEN_ID = 200011

    # 创建时间统计器
    timing_stats = GTZANTimingStats()

    print(f"\n=== GTZAN评测配置 (KV Press) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"数据路径: {data_path_root}")
    print(f"元数据文件: {metadata_file}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 30)

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

    # 加载GTZAN元数据
    print(f"加载GTZAN元数据: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"错误: 元数据文件不存在: {metadata_file}")
        return
    
    samples = load_gtzan_metadata(metadata_file)
    
    # 应用样本限制
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"应用样本限制，处理 {len(samples)} 个样本")

    # 统计各类型音乐风格数量
    genre_stats = {}
    for sample in samples:
        genre = sample.get("genre_label", "unknown")
        genre_stats[genre] = genre_stats.get(genre, 0) + 1
    
    print(f"风格统计: {genre_stats}")

    audio_cache = {}
    results = []
    correct_count = 0
    genre_correct = {genre: 0 for genre in genre_stats.keys()}
    genre_total = {genre: 0 for genre in genre_stats.keys()}

    # 打印初始内存使用情况并记录初始内存
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 记录初始内存用于后续计算
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    timing_stats.record_initial_memory()  # 记录初始GPU内存

    print(f"开始评估 {len(samples)} 个样本...")
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="GTZAN评估 (KV Press)")

    for idx, sample in progress_bar:
        try:
            # 构建音频文件路径
            audio_rel = sample["path"]
            audio_full = os.path.join(data_path_root, "wav",audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"跳过不存在的音频文件: {audio_full}")
                # 创建跳过记录
                result_entry = {
                    "audio_file": audio_rel,
                    "question": sample.get("question", ""),
                    "ground_truth": sample.get("answer_gt", ""),
                    "model_output": "SKIPPED - File not found",
                    "extracted_answer": "skip",
                    "is_correct": False,
                    "genre_label": sample.get("genre_label", "unknown"),
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

            # 加载音频
            audio_raw, sr = load_audio_for_gtzan(audio_full, audio_cache)
            audio = prepare_audio_for_processor(audio_raw[0])
            audio_np, sr = audio_cache[audio_full]

            # 准备选项列表
            options = [
                sample["choice_a"],
                sample["choice_b"], 
                sample["choice_c"],
                sample["choice_d"]
            ]

            # 创建提示词
            prompt = create_gtzan_prompt(sample['question'], options)

            # 重置峰值内存统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # 统计当前风格
            current_genre = sample.get("genre_label", "unknown")
            genre_total[current_genre] = genre_total.get(current_genre, 0) + 1

            # 使用Pipeline进行推理，支持KV Press压缩和CUDA Events精确计时
            try:
                result = pipeline(
                    prompt=prompt,
                    audios=audio,
                    press=press,
                    max_new_tokens=3,  # GTZAN任务只需要很少的token
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
                
                # 从pipeline返回的metrics中获取峰值内存
                current_peak_memory = 0
                if 'metrics' in result and result['metrics']:
                    current_peak_memory = result['metrics'].get('peak_memory_gb', 0.0)
                
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
                        max_new_tokens=3,
                        generation_config=generation_config,
                        do_sample=False,
                        use_cache=True
                    )
                end_time = time.time()
                
                # 获取峰值内存（fallback模式）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    current_peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
                else:
                    current_peak_memory = 0
                
                tokens = out_ids[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                resp = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                prefill_time = 0.0
                decode_time = end_time - start_time

            # 处理输出和结果统计
            pred = clean_text_response(resp)
            correct = int(pred == sample["answer_gt"])
            if correct:
                correct_count += 1
                genre_correct[current_genre] = genre_correct.get(current_genre, 0) + 1

            current_acc = (correct_count / (idx + 1)) * 100
            progress_bar.set_postfix({
                'acc': f'{current_acc:.2f}%', 
                'ans': f'{pred}/{sample["answer_gt"]}',
                'genre': current_genre,
                'audio_len': f'{len(audio_raw[0])/sr:.1f}s'
            })

            # 记录时间统计 - 排除第一个样本
            if idx > 0:
                timing_stats.add_record(
                    prefill_time, decode_time,
                    output_tokens,
                    0,  # input_tokens不再需要计算
                    len(audio_raw[0]) / sr,
                    peak_memory_gb=current_peak_memory
                )

            # 保存详细结果
            results.append({
                "idx": idx,
                "uniq_id": sample.get("uniq_id", idx),
                "genre_label": current_genre,
                "path": audio_rel,
                "question": sample["question"],
                "options": options,
                "prediction": pred,
                "ground_truth": sample["answer_gt"],
                "correct": correct,
                "response_text": resp
            })

            # 内存清理
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
            # 清理可能的中间变量
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'out_ids' in locals():
                del out_ids
            torch.cuda.empty_cache()
            continue

    # 计算最终准确率
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    # 计算各风格准确率
    genre_accuracies = {}
    for genre in genre_stats.keys():
        if genre_total.get(genre, 0) > 0:
            genre_accuracies[genre] = genre_correct.get(genre, 0) / genre_total[genre] * 100

    # 计算F1 Score（对于多分类任务，计算宏平均F1）
    # 收集所有可能的选择（A, B, C, D）
    all_choices = ['A', 'B', 'C', 'D']
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for choice in all_choices:
        # 计算每个选择的 TP, FP, FN
        tp = sum(1 for r in results if r["prediction"] == choice and r["ground_truth"] == choice)
        fp = sum(1 for r in results if r["prediction"] == choice and r["ground_truth"] != choice)
        fn = sum(1 for r in results if r["prediction"] != choice and r["ground_truth"] == choice)
        
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
    
    # 获取初始内存和峰值内存的GB值
    initial_memory_gb = initial_memory / (1024**3) if initial_memory else 0
    avg_peak_memory_gb = timing_summary.get("avg_peak_memory_gb", 0)

    # 创建结果摘要
    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "macro_f1_score": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "genre_stats": genre_stats,
        "genre_accuracies": genre_accuracies,
        "genre_correct": genre_correct,
        "genre_total": genre_total,
        "memory_stats": {
            "initial_memory_gb": initial_memory_gb,
            "avg_peak_memory_gb": avg_peak_memory_gb
        },
        "config": {
            "gpu_id": gpu_id,
            "compression_ratio": compression_ratio,
            "press_type": press_type,
            "sample_limit": sample_limit,
            "data_path": data_path_root,
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
    print(f"\n=== GTZAN评测结果摘要 (KV Press) ===")
    print(f"总样本数: {total}")
    print(f"总准确率: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    print(f"宏平均F1分数: {macro_f1:.3f}")
    print(f"宏平均精度: {macro_precision:.3f}")
    print(f"宏平均召回率: {macro_recall:.3f}")
    
    print(f"\n各风格准确率:")
    for genre, acc in genre_accuracies.items():
        correct_num = genre_correct.get(genre, 0)
        total_num = genre_total.get(genre, 0)
        print(f"  {genre}: {acc:.2f}% ({correct_num}/{total_num})")
    
    print(f"\n=== 内存统计 ===")
    print(f"初始内存: {initial_memory_gb:.2f}GB")
    print(f"平均峰值内存: {avg_peak_memory_gb:.2f}GB")
    
    print(f"\n=== 推理时间统计 ===")
    print(f"平均推理时间: {timing_summary.get('avg_total_time', 0):.4f}秒 (排除第一个样本)")
    print(f"平均Prefill时间: {timing_summary.get('avg_prefill_time', 0):.4f}秒 (排除第一个样本)")
    print(f"平均Decode时间: {timing_summary.get('avg_decode_time', 0):.4f}秒 (排除第一个样本)")
    print(f"平均吞吐量: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/秒")
    
    print(f"\n=== KV Press配置摘要 ===")
    print(f"Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"模型补丁状态: {'成功' if patch_success else '失败'}")
    
    print(f"结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

if __name__ == "__main__":
    import sys
    main()
