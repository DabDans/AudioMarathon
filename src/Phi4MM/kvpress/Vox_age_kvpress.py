import os
import json
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
import re
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

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

random.seed(42)

def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

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
        print("使用TOVAPress (基于时间顺序的注意力值分析)")
        return TOVAPress(compression_ratio=compression_ratio)
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

# 数据路径配置 - 修改为你的目录
data_path_root = '/data/hepeize05/Audio_Longbench/Dataset/VoxCeleb/concatenated_audio'  # 包含wav文件夹和元数据的目录
result_dir = os.environ.get("RESULTS_DIR", './Vox_Results')
os.makedirs(result_dir, exist_ok=True)

# 修改输出文件路径和命名 - 加入KV Press标识
output_file = f'{result_dir}/VoxCeleb_age_results_gpu{gpu_id}_kvpress_{press_type}_{compression_ratio}.json'
timing_output_file = f'{result_dir}/VoxCeleb_age_timing_stats_gpu{gpu_id}_kvpress_{press_type}_{compression_ratio}.json'
print(f"结果将保存到: {output_file}")
print(f"时间统计将保存到: {timing_output_file}")

# 音频特殊token ID
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'

# 时间统计类
class VoxTimingStats:
    """跟踪VoxCeleb年龄分类任务的推理时间统计，支持CUDA Events和GPU内存监控"""
    def __init__(self):
        self.timing_records = []
        self.age_group_stats = defaultdict(list)
        self.speaker_stats = defaultdict(list)
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
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens=0, 
                   age_group=None, speaker_id=None, gpu_memory_peak=None):
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
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "current_memory_gb": current_memory / 1024**3 if self.cuda_available else 0,
            "peak_memory_gb": peak_memory / 1024**3 if self.cuda_available else 0,
            "age_group": age_group,
            "speaker_id": speaker_id
        }
        
        self.timing_records.append(record)
        
        # 按年龄组分类记录
        if age_group:
            self.age_group_stats[age_group].append(record)
        
        # 按说话人分类记录
        if speaker_id:
            self.speaker_stats[speaker_id].append(record)
    
    def get_summary(self):
        """计算并返回统计摘要，排除前100个样本"""
        if len(self.timing_records) <= 100:
            return {"message": "样本数量不足100，无法计算统计"}
        
        # 排除前100个样本进行统计
        valid_records = self.timing_records[100:]
        
        if not valid_records:
            return {"message": "没有有效的时间记录"}
        
        # 基本统计
        total_samples = len(valid_records)
        total_prefill = sum(r["prefill_time"] for r in valid_records)
        total_decode = sum(r["decode_time"] for r in valid_records)
        total_time = sum(r["total_time"] for r in valid_records)
        total_tokens = sum(r["output_tokens"] for r in valid_records)
        
        avg_prefill = total_prefill / total_samples
        avg_decode = total_decode / total_samples
        avg_total = total_time / total_samples
        avg_tokens = total_tokens / total_samples
        avg_tokens_per_sec = total_tokens / total_decode if total_decode > 0 else 0
        
        # GPU内存统计
        gpu_memory_stats = {}
        if self.cuda_available and valid_records:
            current_memories = [r["current_memory_gb"] for r in valid_records]
            peak_memories = [r["peak_memory_gb"] for r in valid_records]
            
            gpu_memory_stats = {
                "initial_memory_gb": self.initial_memory / 1024**3,
                "peak_memory_gb": self.peak_memory / 1024**3,
                "total_peak_memory_gb": self.total_peak_memory / 1024**3,
                "avg_current_memory_gb": sum(current_memories) / len(current_memories),
                "max_peak_memory_gb": max(peak_memories) if peak_memories else 0,
                "min_peak_memory_gb": min(peak_memories) if peak_memories else 0
            }
        
        # 按年龄组统计
        age_group_summaries = {}
        for age_group, records in self.age_group_stats.items():
            if len(records) > 0:
                age_group_summaries[age_group] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": {
                "total_samples": total_samples,
                "avg_prefill_time": avg_prefill,
                "avg_decode_time": avg_decode,
                "avg_total_time": avg_total,
                "avg_tokens": avg_tokens,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "prefill_percentage": (avg_prefill / avg_total * 100) if avg_total > 0 else 0,
                "decode_percentage": (avg_decode / avg_total * 100) if avg_total > 0 else 0,
                "gpu_memory_stats": gpu_memory_stats
            },
            "age_group_summaries": age_group_summaries,
            "detailed_records": valid_records
        }
    
    def export_to_json(self, output_file):
        """导出统计数据到JSON文件"""
        summary = self.get_summary()
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return output_file

    # 老接口兼容性方法
    def set_current_folder(self, folder_name):
        """兼容性方法：设置当前文件夹(说话人)"""
        pass  # 新版本不需要设置当前文件夹

# 修复的音频处理函数
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

def read_text_file(txt_path):
    """读取对应的文本文件内容"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"读取文本文件失败: {e}")
        return ""

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    """从concatenated_audio目录加载数据集，基于age_classification_task_meta.json"""
    # 加载元数据JSON文件
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"从{meta_file}加载了{len(metadata)}个样本元数据")
    
    # 遍历元数据中的每个条目
    for item in metadata:
        # 获取音频文件路径
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        # 检查文件是否存在
        if not os.path.exists(wav_path):
            print(f"警告: 文件不存在 {wav_path}")
            continue
        
        # 提取说话人ID和年龄组信息
        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()  # 年龄组标签
        speaker_age = item.get("speaker_age", 0)  # 具体年龄
        
        # 构建样本信息
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
    
    print(f"总计加载了 {len(all_samples)} 个有效音频样本")
    
    # 应用样本限制
    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"应用样本限制: 从{len(all_samples)}个样本中随机选择{sample_limit}个")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"限制后样本数量: {len(all_samples)}")
    
    # 按年龄组分组统计
    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("年龄组分布:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count}个样本")
    
    # 随机打乱数据集
    random.shuffle(all_samples)
    
    return all_samples

def extract_age_answer(text, choices):
    """从模型输出文本中提取年龄组答案，过滤system prompt信息"""
    if not text:
        return ""
    
    # 移除常见的system prompt模式
    text = re.sub(r'^.*?(?:system|assistant|user).*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^.*?(?:Answer|Response|Output).*?:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*<?/?s?>\s*', '', text)
    
    text_lower = text.lower().strip()
    
    # 优先匹配明确的选项格式
    option_patterns = [
        r'(?:选择|答案|answer|choice|option)?\s*[：:]\s*([ABCDE])',
        r'([ABCDE])[).]',
        r'([ABCDE])\s*[：:]',
        r'(?:选项|option|choice)\s*([ABCDE])',
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            option_letter = match.group(1).upper()
            choice_key = f"choice_{option_letter.lower()}"
            if choice_key in choices:
                return choices[choice_key]
    
    # 直接检测a/b/c/d/e回答
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return choices["choice_d"]
    if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)'):
        return choices["choice_e"]
        
    # 检查是否含有明确的选项指示
    for option, choice_text in choices.items():
        option_letter = option[-1].lower()  # 提取选项字母 (a, b, c, d, e)
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"{option_letter})" in text_lower:
            return choice_text
    
    # 检查是否直接包含选项文本
    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    
    # 如果只匹配到一个选项，返回该选项
    if len(choice_matches) == 1:
        return choice_matches[0]
    
    # 如果仍无法确定，返回空字符串
    return ""

def calculate_age_classification_metrics(results_samples):
    """计算年龄分类的F1 Score和其他指标"""
    y_true = []
    y_pred = []
    valid_samples = 0
    total_samples = len(results_samples)
    
    # 提取有效的预测和真实标签
    for sample in results_samples:
        ground_truth = sample.get("ground_truth", "").strip()
        extracted_answer = sample.get("extracted_answer", "").strip()
        
        if ground_truth and extracted_answer:
            y_true.append(ground_truth)
            y_pred.append(extracted_answer)
            valid_samples += 1
    
    if valid_samples == 0:
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "valid_samples": 0,
            "total_samples": total_samples,
            "classification_report": "没有有效样本进行指标计算"
        }
    
    # 计算F1 Score等指标
    try:
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 生成分类报告
        class_report = classification_report(y_true, y_pred, zero_division=0)
        
        return {
            "f1_score": float(f1_weighted),
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "valid_samples": valid_samples,
            "total_samples": total_samples,
            "classification_report": class_report
        }
    
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "valid_samples": valid_samples,
            "total_samples": total_samples,
            "classification_report": f"计算出错: {str(e)}"
        }

def main():
    print(f"\n=== VoxCeleb说话人年龄分类评测配置 (KV Press) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"数据路径: {data_path_root}")
    if sample_limit > 0:
        print(f"样本限制: {sample_limit}")
    print("=" * 50)

    # Step1: 加载模型
    print("加载Phi-4-multimodal-instruct模型...")
    model_path = "/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 使用KV Press配置加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="balanced_low_0",
        torch_dtype="auto",
        trust_remote_code=True,
        **kv_press_config["model_kwargs"]
    )


    # 为 Phi4MM 模型添加补丁以便与 KV Press 兼容
    patch_success = patch_phi4mm_for_kvpress(model)
    if not patch_success:
        print("警告: KV Press补丁应用失败，某些Press类型可能无法正常工作")

    # 初始化 KV Press
    press = initialize_kv_press(model, compression_ratio)
        # 加载生成配置
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # 初始化管道
    pipeline = KVPressAudioTranscriptionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        processor=processor,
        audio_special_token_id=_AUDIO_SPECIAL_TOKEN_ID
    )
    
    # 创建时间统计器
    timing_stats = VoxTimingStats()
    timing_stats.record_initial_memory()

    # 扫描VoxCeleb数据集
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    
    # 按说话人ID分组
    grouped_samples = {}
    for sample in samples:
        speaker_id = sample["speaker_id"]
        if speaker_id not in grouped_samples:
            grouped_samples[speaker_id] = []
        grouped_samples[speaker_id].append(sample)
    
    # 统计年龄组数量
    age_group_counts = {}
    for s in samples:
        group = s["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("年龄组统计:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count}个样本")
    
    # 创建结果数据结构
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
    
    # 打印初始内存使用情况
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 创建外层循环的总进度条
    with tqdm(total=len(grouped_samples), desc="处理说话人组 (KV Press)", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        # 处理说话人组的循环
        for speaker_id, items in grouped_samples.items():
            folder_count += 1
            pbar_folders.set_description(f"处理说话人[{folder_count}/{total_folders}]: {speaker_id}")
            
            # 设置当前处理的文件夹
            timing_stats.set_current_folder(speaker_id)
            # 移除内层tqdm进度条，使用简单计数替代
            sample_count = 0
            total_samples = len(items)            
            # 处理当前说话人的所有样本
            for i, item in enumerate(items):
                try:
                    sample_count = i + 1
                    wav_path = item['wav_path']
                    
                    # 检查音频文件是否存在
                    if not os.path.exists(wav_path):
                        print(f"跳过不存在的音频文件: {wav_path}")
                        # 创建跳过记录
                        sample_result = {
                            "wav_path": wav_path,
                            "speaker_id": speaker_id,
                            "age_group": item.get("age_group", "unknown"),
                            "ground_truth": item.get("age_group", "unknown"),
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
                        results["samples"].append(sample_result)
                        continue
                        
                    task = item.get('task', 'Speaker_Age_Classification')
                    
                    # 使用专门的年龄识别提示词
                    instruction = "Listen to this audio and identify the speaker's age group. Choose the most appropriate option: (a) Young Adult (18-30), (b) Early Career (31-40), (c) Mid Career (41-50), (d) Senior (51-70), (e) Elderly (71+). Answer with only the letter (a, b, c, d, or e)."
                    full_prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
                    
                    # 准备音频输入
                    audio = prepare_audio_for_processor(wav_path)
                    
                    # 计算音频token长度（估算）
                    audio_token_length = len(audio[0][0]) // 320 if audio and len(audio) > 0 else 0
                    
                    # 重置峰值内存统计
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    
                    # 使用Pipeline进行推理，支持KV Press压缩和CUDA Events精确计时
                    try:
                        result = pipeline(
                            prompt=full_prompt,
                            audios=audio,
                            press=press,
                            max_new_tokens=3,  # 年龄分类任务通常是单字母答案
                            do_sample=False,
                            measure_time=True
                        )
                        
                        # 初始化默认值，避免变量引用错误
                        output = result['text']
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
                        peak_memory_gb = result.get('metrics', {}).get('peak_memory_gb', 0)
                        
                    except Exception as pipeline_error:
                        print(f"Pipeline推理失败: {pipeline_error}")
                        print("回退到标准推理方式")
                        
                        # 处理输入
                        inputs = processor(
                            text=full_prompt,
                            audios=audio,
                            return_tensors="pt",
                        ).to("cuda")
                        inputs['input_mode'] = torch.tensor([2])

                        # 标准推理
                        start_time = time.time()
                        with torch.no_grad():
                            out_ids = model.generate(
                                **inputs,
                                max_new_tokens=3,
                                generation_config=generation_config,
                                do_sample=False,
                                return_dict_in_generate=True
                            )
                        end_time = time.time()
                        
                        tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
                        output_tokens = len(tokens[0])
                        output = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                        
                        # 简单的时间估算（因为没有CUDA events）
                        total_time = end_time - start_time
                        prefill_time = total_time * 0.3  # 估算prefill占30%
                        decode_time = total_time * 0.7   # 估算decode占70%
                        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        
                        prefill_time = 0.0
                        decode_time = end_time - start_time
                        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                    
                    # 提取答案 - 年龄组分类
                    choices = {
                        "choice_a": item["choice_a"],
                        "choice_b": item["choice_b"],
                        "choice_c": item["choice_c"],
                        "choice_d": item["choice_d"],
                        "choice_e": item["choice_e"]
                    }
                    predicted_age_group = extract_age_answer(output, choices)
                    
                    # 检查答案是否正确
                    ground_truth = item["age_group"].strip()
                    is_correct = predicted_age_group == ground_truth
                    
                    # 更新统计信息
                    results["summary"]["total_samples"] += 1
                    if ground_truth not in results["summary"]["age_group_stats"]:
                        results["summary"]["age_group_stats"][ground_truth] = {
                            "total": 0,
                            "correct": 0
                        }
                    
                    results["summary"]["age_group_stats"][ground_truth]["total"] += 1
                    if is_correct:
                        results["summary"]["age_group_stats"][ground_truth]["correct"] += 1
                        results["summary"]["correct_samples"] += 1
                    
                    # 跳过第一个样本的时间统计
                    if i > 0:
                        # 更新时间统计
                        results["summary"]["timing"]["total_prefill_time"] += prefill_time
                        results["summary"]["timing"]["total_decode_time"] += decode_time
                        results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                        
                        # 获取GPU峰值内存
                        gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                        
                        timing_stats.add_record(
                            prefill_time=prefill_time, 
                            decode_time=decode_time, 
                            output_tokens=output_tokens,
                            input_tokens=audio_token_length,
                            age_group=ground_truth,
                            speaker_id=item["speaker_id"],
                            gpu_memory_peak=gpu_peak
                        )
                    
                except Exception as e:
                    print(f"推理错误: {e}")
                    traceback.print_exc()
                    output = "ERROR"
                    predicted_age_group = "error"
                    is_correct = False
                    prefill_time = 0
                    decode_time = 0
                    output_tokens = 0
                    audio_token_length = 0
                    ground_truth = item.get("age_group", "unknown")
                
                # 保存样本结果
                sample_result = {
                    "audio_file": os.path.basename(wav_path),
                    "speaker_id": item["speaker_id"],
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "extracted_answer": predicted_age_group,
                    "is_correct": is_correct,
                    "audio_tokens": audio_token_length,
                    "output_tokens": output_tokens,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
                results["samples"].append(sample_result)
                
                # 安全地清理变量
                if 'inputs' in locals():
                    del inputs
                if 'outputs' in locals():
                    del outputs
                if 'generate_ids' in locals():
                    del generate_ids
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
                
                if sample_count % update_interval == 0 or sample_count == total_samples:
                    # 计算准确率
                    current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                    
                    # 更新外层进度条的后缀
                    pbar_folders.set_postfix_str(
                        f"样本:{sample_count}/{total_samples}, 准确率:{current_accuracy:.2%}"
                    )
                    
                    if is_screen_env:
                        # 在screen环境额外打印一行进度
                        print(f"  进度: {sample_count}/{total_samples} ({sample_count/total_samples*100:.1f}%), "
                              f"准确率: {current_accuracy:.2%}")
            
            # 处理完一个说话人，更新外层进度条
            pbar_folders.update()

    # 计算平均时间（排除第一个样本）
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
    
    # 计算各年龄组准确率
    for age_group, stats in results["summary"]["age_group_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # 计算F1 Score指标
    age_metrics = calculate_age_classification_metrics(results["samples"])
    results["summary"]["age_classification_metrics"] = age_metrics
    
    # 保存结果为单个JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存时间统计
    timing_stats.export_to_json(timing_output_file)
    
    # 打印结果摘要
    print("\n=== VoxCeleb 说话人年龄分类评测结果摘要 (KV Press) ===")
    print(f"总样本数: {total_samples}")
    print(f"总准确率: {results['summary']['accuracy']:.2%}")
    
    # 显示F1 Score结果
    print(f"\n=== F1 Score 指标 ===")
    print(f"Weighted F1 Score: {age_metrics['f1_score']:.4f}")
    print(f"Weighted Precision: {age_metrics['precision']:.4f}")
    print(f"Weighted Recall: {age_metrics['recall']:.4f}")
    print(f"有效样本: {age_metrics['valid_samples']}/{age_metrics['total_samples']}")
    
    print("\n各年龄组准确率:")
    for age_group, stats in results["summary"]["age_group_stats"].items():
        print(f"  {age_group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== 时间统计（CUDA Events精确测量，排除前100个样本）===")
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
    
    print(f"\n=== KV Press配置摘要 ===")
    print(f"Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"模型补丁状态: {'成功' if patch_success else '失败'}")
    
    print(f"结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")

if __name__ == "__main__":
    main()
