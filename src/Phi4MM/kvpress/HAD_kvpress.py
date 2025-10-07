import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
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
import re
import contextlib
import subprocess
import tempfile
from collections import defaultdict

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

# 导入管道
from kvpress.pipeline import KVPressAudioTranscriptionPipeline

# 禁用警告
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

def initialize_kv_press_simplified(model, config):
    """初始化 KV Press 压缩组件，基于配置选择合适的压缩类型"""
    # 获取压缩参数
    press_type = config.get("press_type", "knorm").lower()
    compression_ratio = config.get("compression_ratio", 0.5)
    window_size = config.get("window_size", 1024)

    print(f"正在初始化 KV Press 类型: {press_type}, 压缩比: {compression_ratio}")

    # 检查模型是否成功应用了 KV Press 补丁
    if not getattr(model.model, "_kvpress_patched", False):
        print("警告: KV Press 补丁未成功应用或未运行，KV Press 可能无法正常工作")

    # 根据配置的类型选择 Press 实现
    if press_type == "knorm":
        print("使用 KnormPress (基于 Key-norm 的压缩方法，最稳定)")
        return KnormPress(compression_ratio=compression_ratio)
    elif press_type == "expected":
        print("使用 ExpectedAttentionPress (基于 RoPE 的注意力压缩)")
        return ExpectedAttentionPress(compression_ratio=compression_ratio)
    elif press_type == "random":
        print("使用 RandomPress (随机丢弃tokens)")
        return RandomPress(compression_ratio=compression_ratio)
    elif press_type == "observed":
        print("使用 ObservedAttentionPress (基于观察到的注意力分数)")
        return ObservedAttentionPress(compression_ratio=compression_ratio)
    elif press_type == "tova":
        print("使用 CustomTOVAPress (基于时间顺序的注意力值分析)")
        return CustomTOVAPress(compression_ratio=compression_ratio)
    elif press_type == "snap":
        print("使用 SnapKVPress (适用于LoRA微调模型)")
        return SnapKVPress(compression_ratio=compression_ratio)
    elif press_type == "streaming":
        print("使用 StreamingLLMPress (适用于长文本流式处理)")
        return StreamingLLMPress(window_size=window_size)
    else:
        print(f"未知压缩类型 {press_type}，默认使用 KnormPress")
        return KnormPress(compression_ratio=compression_ratio)

# 获取GPU ID
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"使用 GPU ID: {gpu_id}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

# 从环境变量获取KV Press参数
compression_ratio = float(os.environ.get("COMPRESSION_RATIO", 0.5))
press_type = os.environ.get("PRESS_TYPE", "knorm")

# 样本限制 (如果提供)
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"样本限制设置为: {sample_limit}")

# 数据路径配置
data_path_root = '/data/hepeize05/Audio_Longbench/Dataset/HAD/concatenated_audio'  # 包含real和fake文件夹的目录
result_dir = os.environ.get("RESULTS_DIR", './HAD_Results')
os.makedirs(result_dir, exist_ok=True)

# 修改输出文件路径和命名 - 加入KV Press标识
output_file = f'{result_dir}/HAD_results_gpu{gpu_id}_kvpress_{press_type}_{compression_ratio}.json'
timing_output_file = f'{result_dir}/HAD_timing_stats_gpu{gpu_id}_kvpress_{press_type}_{compression_ratio}.json'
print(f"结果将保存到: {output_file}")
print(f"时间统计将保存到: {timing_output_file}")

# 音频特殊token ID
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'

# 时间统计类
class FolderTimingStats:
    """跟踪每个文件夹的推理时间统计"""
    def __init__(self):
        self.folder_stats = {}
        self.current_folder = None
        self.initial_gpu_memory = 0
    
    def record_initial_memory(self):
        """记录初始GPU内存使用情况"""
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            self.initial_gpu_memory = 0
    
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
    
    def add_record(self, prefill_time, decode_time, output_tokens, peak_memory_gb=None):
        if self.current_folder is None:
            return
        
        folder_data = self.folder_stats[self.current_folder]
        folder_data["samples"] += 1
        folder_data["total_prefill_time"] += prefill_time
        folder_data["total_decode_time"] += decode_time
        folder_data["total_tokens"] += output_tokens
        
        # 添加峰值内存跟踪
        if peak_memory_gb is not None:
            if "total_peak_memory" not in folder_data:
                folder_data["total_peak_memory"] = 0
            folder_data["total_peak_memory"] += peak_memory_gb
        
        # 添加详细记录
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        }
        if peak_memory_gb is not None:
            record["peak_memory_gb"] = peak_memory_gb
        folder_data["timing_records"].append(record)
    
    def export_to_json(self, output_file):
        """导出统计数据到JSON文件"""
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
                    "avg_tokens_per_sec": stats["total_tokens"] / stats["total_decode_time"] if stats["total_decode_time"] > 0 else 0,
                    "avg_peak_memory_gb": stats.get("total_peak_memory", 0) / stats["samples"] if stats["samples"] > 0 else 0
                }
                for folder, stats in self.folder_stats.items() if stats["samples"] > 0
            },
            "detailed_records": self.folder_stats
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

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

def load_had_dataset(root_dir):
    """加载HAD数据集，平衡真假样本数量"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    
    all_samples = []
    
    # 加载真实音频样本
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "real",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "real",
                "task": "Audio_Authenticity_Detection"
            })
    
    # 加载伪造音频样本
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for wav_path in fake_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "fake",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "fake",
                "task": "Audio_Authenticity_Detection"
            })
    
    print(f"总计加载了 {len(all_samples)} 个音频样本")
    
    # 按标签分组样本
    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"原始样本数量: 真实={len(real_samples)}, 伪造={len(fake_samples)}")
    
    # 计算每种类别较少一方的样本数量
    min_samples_per_category = min(len(real_samples), len(fake_samples))
    
    # 从样本较多的一方随机抽取等量样本
    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)
    
    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)
    
    # 合并平衡后的真假样本
    balanced_samples = real_samples + fake_samples
    
    # 随机打乱整个数据集
    random.shuffle(balanced_samples)
    
    print(f"平衡后样本数量: 真实={len(real_samples)}, 伪造={len(fake_samples)}, 总计={len(balanced_samples)}")
    
    return balanced_samples

def extract_authenticity_answer(text, choice_a="real", choice_b="fake"):
    """从模型输出文本中提取音频真伪答案"""
    text_lower = text.lower().strip()
    
    # 标准化选项值
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    
    # 直接检测a/b回答
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a_lower
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b_lower
        
    # 检查是否含有明确的a/b选项指示
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a_lower
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b_lower
    
    # 检查是否直接包含选项文本
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a_lower
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b_lower
    
    # 如果仍无法确定，尝试更精确的模式匹配
    if choice_a_lower == "real" and choice_b_lower == "fake":
        # 使用单词边界确保精确匹配
        real_match = re.search(r'\breal\b|\bauthentic\b|\bgenuine\b', text_lower) is not None
        fake_match = re.search(r'\bfake\b|\bartificial\b|\bsynthetic\b|\bsynthesized\b', text_lower) is not None
        
        if real_match and not fake_match:
            return "real"
        if fake_match and not real_match:
            return "fake"
    
    # 如果仍无法确定，返回空字符串
    return ""

def main():
    print(f"\n=== HAD音频真伪检测评测配置 (KV Press) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"数据路径: {data_path_root}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 40)

    # Step1: 加载模型
    print("加载Phi-4-multimodal-instruct模型...")
    model_path = "/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B"
    # 固定模型版本以避免下载新版本的代码文件
    model_revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        revision=model_revision,
        trust_remote_code=True
    )
    
    # 使用KV Press配置加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=model_revision,
        device_map="auto",  # 修复：使用auto而不是balanced_low_0
        torch_dtype="auto",
        trust_remote_code=True,
        **kv_press_config["model_kwargs"]
    )
    model.eval()

    # 为 Phi4MM 模型添加补丁以便与 KV Press 兼容
    patch_success = patch_phi4mm_for_kvpress(model)
    if not patch_success:
        print("警告: KV Press补丁应用失败，某些Press类型可能无法正常工作")

    # 初始化 KV Press - 使用简化版本
    press = initialize_kv_press_simplified(model, kv_press_config)
    
    # 修复：使用正确的pipeline初始化方式
    pipeline = KVPressAudioTranscriptionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        processor=processor,
        audio_special_token_id=_AUDIO_SPECIAL_TOKEN_ID
    )
    
    # 创建时间统计器
    timing_stats = FolderTimingStats()
    
    
    # 扫描HAD数据集
    samples = load_had_dataset(data_path_root)
    
    # 如果设置了样本限制，则截取指定数量的样本
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"应用样本限制，将处理 {len(samples)} 个样本")
    
    # 按类别分组样本用于统计
    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)
    
    # 统计真假样本数量
    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"分类统计: 真实样本={real_count}, 伪造样本={fake_count}")
    
    # 创建结果数据结构
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
            "timing": {
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_prefill_time": 0,
                "total_decode_time": 0,
                "total_total_time": 0,
            },
            "kv_press_config": kv_press_config
        }
    }
    
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
    
    # 打印初始内存使用情况并记录初始内存
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 记录初始内存用于后续计算
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    timing_stats.record_initial_memory()  # 记录初始GPU内存
    
    # 创建进度条处理所有样本
    with tqdm(total=len(samples), desc="处理HAD音频样本 (KV Press)", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        # 设置当前处理的文件夹用于时间统计
        timing_stats.set_current_folder("HAD_Audio_Detection")
        
        # 遍历处理所有样本
        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            label = item['label']
            task = item.get('task', 'Audio_Authenticity_Detection')
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"跳过不存在的音频文件: {audio_path}")
                # 创建跳过记录
                sample_result = {
                    "audio_file": os.path.basename(audio_path),
                    "audio_label": label,
                    "ground_truth": item.get("answer_gt", "unknown"),
                    "model_output": "SKIPPED - File not found",
                    "extracted_answer": "skip",
                    "is_correct": False,
                    "audio_tokens": 0,
                    "output_tokens": 0,
                    "prefill_time": 0.0,
                    "decode_time": 0.0,
                    "total_time": 0.0,
                    "peak_memory_gb": 0.0,
                    "skipped": True,
                    "skip_reason": "Audio file not found"
                }
                results["samples"].append(sample_result)
                pbar.update()
                continue
                
            # 在每个样本处理前重置GPU内存统计，确保准确的峰值测量
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # 使用专门的音频真伪检测提示词
            instruction = "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'."
            full_prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
            
            # 准备音频输入
            audio = prepare_audio_for_processor(audio_path)
            
            # 使用管道进行推理 - 修复调用方式，使用context manager
            start_time = time.time()
            with press(model) if press is not None else contextlib.nullcontext():
                result = pipeline(
                    prompt=full_prompt,
                    audios=audio,
                    press=press,
                    input_mode=2,
                    measure_time=True,
                    max_new_tokens=5,  # 音频问答任务通常需要较短回答
                    do_sample=False,
                    return_legacy_cache=True
                )
            end_time = time.time()
            
            # 手动获取峰值内存使用情况
            peak_memory_bytes = 0
            peak_memory_gb = 0.0
            if torch.cuda.is_available():
                peak_memory_bytes = torch.cuda.max_memory_allocated()
                peak_memory_gb = peak_memory_bytes / (1024**3)  # 转换为GB
            
            # 从结果中提取信息
            output = result.get("text", "")
            timing_info = result.get("metrics", {})
            prefill_time = timing_info.get("prefill_time", 0.0)
            decode_time = timing_info.get("generation_time", 0.0)
            output_tokens = result.get("output_tokens", 0)
            
            # 提取答案 - 更可靠的方式
            predicted_label = extract_authenticity_answer(output)
            
            # 检查答案是否正确
            ground_truth = item["answer_gt"].lower().strip()
            
            is_correct = predicted_label == ground_truth
            
            # 更新统计信息
            results["summary"]["total_samples"] += 1
            if ground_truth in ["real", "fake"]:
                results["summary"][f"{ground_truth}_total"] += 1
                if is_correct:
                    results["summary"][f"{ground_truth}_correct"] += 1
                    results["summary"]["correct_samples"] += 1
            
            # 更新时间统计 - 排除第一个样本
            if i > 0:  # 排除第一个样本 (index 0)
                results["summary"]["timing"]["total_prefill_time"] += prefill_time
                results["summary"]["timing"]["total_decode_time"] += decode_time
                results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
            
            # 记录时间统计
            audio_data = audio[0] if isinstance(audio, list) and len(audio) > 0 else audio
            timing_stats.add_record(
                prefill_time=prefill_time,
                decode_time=decode_time,
                output_tokens=output_tokens,
                peak_memory_gb=peak_memory_gb
            )
            
            # 保存样本结果
            audio_data = audio[0] if isinstance(audio, list) and len(audio) > 0 else audio
            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "audio_label": label,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_label,
                "is_correct": is_correct,
                "audio_tokens": len(audio_data[0]) // 320 if isinstance(audio_data, tuple) else 0,  # 估算audio tokens
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time,
                "peak_memory_gb": peak_memory_gb
            }
            # 添加到结果列表
            results["samples"].append(sample_result)
            
            # 内存清理
            del result
            torch.cuda.empty_cache()
            
            # 每10个样本进行一次深度清理
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
                # 每100个样本打印内存使用情况
                if (i + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [样本 {i+1}] GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
            
            # 在screen环境下每10个样本更新一次，在标准环境下每个样本都更新
            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                # 计算准确率
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                
                # 更新进度条的后缀
                pbar.set_postfix_str(
                    f"准确率:{current_accuracy:.2%}"
                )
                
                if is_screen_env:
                    # 在screen环境额外打印一行进度
                    print(f"  进度: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"准确率: {current_accuracy:.2%}")
            
            # 更新进度条
            pbar.update()
    
    # 计算平均时间 - 排除第一个样本
    total_samples = results["summary"]["total_samples"]
    timing_sample_count = max(0, total_samples - 1)  # 排除第一个样本
    if timing_sample_count > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / timing_sample_count
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / timing_sample_count
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / timing_sample_count
    else:
        results["summary"]["timing"]["avg_prefill_time"] = 0
        results["summary"]["timing"]["avg_decode_time"] = 0
        results["summary"]["timing"]["avg_total_time"] = 0
    
    # 计算准确率
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    results["summary"]["real_accuracy"] = results["summary"]["real_correct"] / results["summary"]["real_total"] if results["summary"]["real_total"] > 0 else 0
    results["summary"]["fake_accuracy"] = results["summary"]["fake_correct"] / results["summary"]["fake_total"] if results["summary"]["fake_total"] > 0 else 0
    
    # 计算精度、召回率和F1分数（以fake为正类）
    tp = results["summary"]["fake_correct"]  # 真正例：正确识别的fake
    fp = results["summary"]["real_total"] - results["summary"]["real_correct"]  # 假正例：错误识别为fake的real
    fn = results["summary"]["fake_total"] - results["summary"]["fake_correct"]  # 假负例：错误识别为real的fake
    tn = results["summary"]["real_correct"]  # 真负例：正确识别的real
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results["summary"]["precision"] = precision
    results["summary"]["recall"] = recall
    results["summary"]["f1_score"] = f1_score
    
    # 保存结果为单个JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存时间统计
    timing_stats.export_to_json(timing_output_file)
    
    # 打印结果摘要
    print("\n=== HAD 音频真伪检测评测结果摘要 (KV Press) ===")
    print(f"总样本数: {total_samples}")
    print(f"总准确率: {results['summary']['accuracy']:.2%}")
    print(f"真实音频准确率: {results['summary']['real_accuracy']:.2%} ({results['summary']['real_correct']}/{results['summary']['real_total']})")
    print(f"伪造音频准确率: {results['summary']['fake_accuracy']:.2%} ({results['summary']['fake_correct']}/{results['summary']['fake_total']})")
    print(f"精度 (Precision): {precision:.2%}")
    print(f"召回率 (Recall): {recall:.2%}")
    print(f"F1分数: {f1_score:.2%}")
    print(f"平均推理时间: {results['summary']['timing']['avg_total_time']:.4f}秒 (排除第一个样本)")
    print(f"平均 Prefill 时间: {results['summary']['timing']['avg_prefill_time']:.4f}秒 (排除第一个样本)")
    print(f"平均 Decode 时间: {results['summary']['timing']['avg_decode_time']:.4f}秒 (排除第一个样本)")
    
    print(f"\n=== KV Press配置摘要 ===")
    print(f"Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"模型补丁状态: {'成功' if patch_success else '失败'}")
    
    print(f"结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

if __name__ == "__main__":
    main()
