import os
import json
import glob
import time
import torch
import soundfile as sf
import numpy as np
import pandas as pd
import gc
import re
import sys
import warnings
import traceback
import contextlib
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
from collections import defaultdict
from transformers import logging
import torch.nn.functional as F
# 添加sklearn导入（如果可用）
try:
    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型，用于JSON序列化"""
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
    else:
        return obj

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

def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

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

class RaceTimingStats:
    """用于记录和分析RACE任务的prefill和decode阶段时间统计，支持CUDA Events和GPU内存监控"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
        self.initial_memory = 0
        self.peak_memory = 0
        
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
            print(f"初始GPU内存使用: {self.initial_memory / 1024**3:.2f} GB")
    
    def record_initial_memory(self):
        """记录初始GPU内存使用情况"""
        if self.cuda_available:
            self.initial_memory = torch.cuda.memory_allocated()
        else:
            self.initial_memory = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration, audio_tokens=0, gpu_memory_peak=None):
        """添加一条时间记录"""
        current_memory = 0
        if self.cuda_available:
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            self.peak_memory = max(self.peak_memory, peak_memory)
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "audio_tokens": audio_tokens,
            "audio_duration": audio_duration,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "gpu_memory_current": current_memory / 1024**3 if self.cuda_available else 0,
            "gpu_memory_peak": gpu_memory_peak / 1024**3 if gpu_memory_peak else 0,
            "kv_press_enabled": True,
            "compression_ratio": kv_press_config["compression_ratio"]
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """获取汇总统计"""
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        df = pd.DataFrame(self.timing_records)
        
        summary = {
            "total_samples": len(df),
            "avg_prefill_time": df["prefill_time"].mean(),
            "avg_decode_time": df["decode_time"].mean(),
            "avg_total_time": df["total_time"].mean(),
            "avg_tokens_per_sec": df["tokens_per_sec"].mean(),
            "total_tokens": df["output_tokens"].sum(),
            "prefill_percentage": (df["prefill_time"].sum() / df["total_time"].sum()) * 100,
            "decode_percentage": (df["decode_time"].sum() / df["total_time"].sum()) * 100,
            "gpu_memory_stats": {
                "initial_memory_gb": self.initial_memory / 1024**3 if self.cuda_available else 0,
                "peak_memory_gb": self.peak_memory / 1024**3 if self.cuda_available else 0,
                "avg_current_memory_gb": df["gpu_memory_current"].mean() if self.cuda_available else 0,
                "max_peak_memory_gb": df["gpu_memory_peak"].max() if self.cuda_available else 0,
            },
            "kv_press_stats": {
                "compression_ratio": kv_press_config["compression_ratio"],
                "press_type": kv_press_config["press_type"],
                "samples_with_compression": len(df)
            }
        }
        
        return summary
    
    def export_to_json(self, output_file):
        """导出统计数据到JSON文件"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records,
            "kv_press_config": kv_press_config
        }
        
        # 转换NumPy类型为JSON可序列化类型
        result = convert_numpy_types(result)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

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

def initialize_kv_press(model, compression_ratio=None):
    """根据模型特性和环境变量选择最合适的KV Press类型"""
    if compression_ratio is None:
        compression_ratio = kv_press_config["compression_ratio"]
    
    if not hasattr(model, "config"):
        print("警告: 模型没有config属性，使用默认KnormPress")
        return KnormPress(compression_ratio=compression_ratio)
    
    kv_press_config["compression_ratio"] = compression_ratio
    
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    kv_press_config["head_dims"] = head_dim
    kv_press_config["num_attention_heads"] = config.num_attention_heads
    
    print(f"模型配置: hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, head_dim={head_dim}")
    
    press_type = kv_press_config["press_type"]
    has_rotary = hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None
    has_kvpress_patch = hasattr(model.model, "_kvpress_patched") and model.model._kvpress_patched
    
    print(f"使用KV Press类型: {press_type}, 压缩比率: {compression_ratio}")
    
    if press_type == "expected" and has_rotary and has_kvpress_patch:
        print("使用ExpectedAttentionPress (基于RoPE的注意力压缩)")
        return ExpectedAttentionPress(compression_ratio=compression_ratio)
    elif press_type == "observed":
        print("使用ObservedAttentionPress (基于观察到的注意力分数)")
        return ObservedAttentionPress(compression_ratio=compression_ratio)
    elif press_type == "random":
        print("使用RandomPress (随机丢弃tokens)")
        return RandomPress(compression_ratio=compression_ratio)
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
        print("使用KnormPress (基于Key-norm的注意力压缩)")
        return KnormPress(compression_ratio=compression_ratio)

def clean_text_response(response):
    """清理模型对RACE任务的响应，仅保留首个字符作为选项标签，过滤system prompt"""
    if not response:
        return ""
    
    # 移除常见的system prompt模式
    response = re.sub(r'^.*?(?:system|assistant|user).*?:\s*', '', response, flags=re.IGNORECASE | re.MULTILINE)
    response = re.sub(r'^.*?(?:Answer|Response|Output).*?:\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'^\s*<?/?s?>\s*', '', response)
    
    resp = response.strip().upper()
    
    # 优先匹配明确的选项格式
    option_patterns = [
        r'(?:选择|答案|answer|choice|option)?\s*[：:]\s*([ABCD])',
        r'([ABCD])[).]',
        r'([ABCD])\s*[：:]',
        r'(?:选项|option|choice)\s*([ABCD])',
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, resp, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果没有找到明确格式，查找首个ABCD字符
    for ch in resp:
        if ch in ["A","B","C","D"]:
            return ch
    
    # 最后fallback
    return resp.split()[0] if resp.split() else ""

def load_audio_for_race(audio_path, audio_cache=None):
    """
    加载音频文件，返回格式与processor一致
    返回: ([audio_array], sampling_rate)
    """
    if audio_cache is not None and audio_path in audio_cache:
        audio_np, sr = audio_cache[audio_path]
    else:
        audio_np, sr = sf.read(audio_path)
        if len(audio_np.shape) > 1:
            audio_np = audio_np[:, 0]
        
        if audio_cache is not None:
            audio_cache[audio_path] = (audio_np, sr)
    
    return [audio_np], sr

def prepare_audio_for_processor(audio_data, target_sr=16000):
    """将音频转换为processor期望的格式"""
    if isinstance(audio_data, list):
        return [(audio, target_sr) for audio in audio_data]
    else:
        return [(audio_data, target_sr)]

def create_race_prompt(question, options):
    """创建RACE任务的提示词"""
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted_options += f"{letter}. {opt}\n"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

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
    
    # 获取环境变量配置
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    
    print(f"使用 GPU ID: {gpu_id}")
    if sample_limit > 0:
        print(f"样本限制设置为: {sample_limit}")
    
    # 数据路径配置
    data_path_root = "/data/hepeize05/Audio_Longbench/Dataset/race_audio"
    results_dir_name = os.environ.get("RESULTS_DIR", "Race_KVPress_Results")
    
    if not os.path.isabs(results_dir_name):
        result_dir = os.path.abspath(results_dir_name)
    else:
        result_dir = results_dir_name
    
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"数据目录: {data_path_root}")
    print(f"结果目录: {result_dir}")
    
    # 输出文件路径
    press_type = kv_press_config["press_type"]
    compression_ratio = kv_press_config["compression_ratio"]
    output_file = os.path.join(result_dir, f'race_kvpress_results_{press_type}_{compression_ratio}.json')
    timing_output_file = os.path.join(result_dir, f'race_timing_stats_{press_type}_{compression_ratio}.json')
    
    print(f"结果将保存到: {output_file}")
    print(f"时间统计将保存到: {timing_output_file}")
    
    # 创建时间统计器
    timing_stats = RaceTimingStats()
    
    # 记录初始内存
    if hasattr(timing_stats, 'record_initial_memory'):
        timing_stats.record_initial_memory()
    elif torch.cuda.is_available():
        # 如果没有record_initial_memory方法，手动清理和重置内存统计
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        print(f"初始GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print(f"\n=== RACE KV Press 评测配置 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比例: {compression_ratio}")
    print(f"数据目录: {data_path_root}")
    print(f"结果目录: {result_dir}")
    print("=" * 50)
    
    # 加载模型和处理器
    print("加载模型和处理器...")
    model_path = "/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B"
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        **kv_press_config["model_kwargs"]
    )
    model.eval()

    # 打印初始内存使用情况
    allocated, reserved = get_gpu_memory_usage()
    print(f"模型加载完成后GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 记录初始内存用于后续计算
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # 为 Phi4MM 模型添加KV Press补丁
    patch_phi4mm_for_kvpress(model)
    
    # 初始化 KV Press
    press = initialize_kv_press(model)

    # 创建pipeline实例
    pipeline = KVPressAudioTranscriptionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        processor=processor,
        audio_special_token_id=_AUDIO_SPECIAL_TOKEN_ID
    )
    
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # 加载benchmark数据
    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"错误: 找不到benchmark文件: {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    # 应用样本限制
    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f"样本数量限制为: {sample_limit}")
    
    audio_cache = {}
    results = []
    
    # 统计变量
    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0
    
    print(f"开始评估 {len(benchmark)} 个样本...")
    
    # 检测环境配置
    is_screen_env = not os.sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': os.sys.stdout
    }
    
    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc="RACE KV Press评估", **tqdm_kwargs)
    
    for idx, sample in progress_bar:
        try:
            audio_rel = sample["audio_path"]
            audio_full = os.path.join(data_path_root, audio_rel)
            
            if not os.path.exists(audio_full):
                print(f"跳过不存在的音频文件: {audio_full}")
                # 创建跳过记录
                result_entry = {
                    "audio_path": audio_rel,
                    "question": sample.get("question", ""),
                    "options": sample.get("options", []),
                    "answer": sample.get("answer", ""),
                    "ground_truth": sample.get("answer", ""),
                    "model_output": "SKIPPED - File not found",
                    "extracted_answer": "skip",
                    "is_correct": False,
                    "difficulty_level": "high" if "high" in audio_rel else ("middle" if "middle" in audio_rel else "unknown"),
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
                
            audio_raw, sr = load_audio_for_race(audio_full, audio_cache)
            audio = prepare_audio_for_processor(audio_raw[0])
            audio_np, sr = audio_cache[audio_full]
            prompt = create_race_prompt(sample['question'], sample['options'])
            
            # 判断属于 high 还是 middle
            if "high" in audio_rel:
                total_high += 1
            elif "middle" in audio_rel:
                total_middle += 1
            
            # 在每个样本处理前重置GPU内存统计，确保准确的峰值测量
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # 使用Pipeline进行推理，支持KV Press压缩和CUDA Events精确计时
            try:
                result = pipeline(
                    prompt=prompt,
                    audios=audio,
                    press=press,
                    max_new_tokens=5,  # RACE任务只需要很少tokens（A/B/C/D）
                    do_sample=False,
                    measure_time=True
                )
                
                # 初始化默认值，避免变量引用错误
                response = result['text']
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
                if peak_memory_gb == 0 and torch.cuda.is_available():
                    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                
            except Exception as pipeline_error:
                print(f"Pipeline推理失败: {pipeline_error}")
                print("回退到标准推理方式")
                
                # 处理输入
                inputs = processor(
                    text=prompt,
                    audios=audio,
                    return_tensors="pt",
                ).to(device)
                inputs['input_mode'] = torch.tensor([2]).to(device)

                # 标准推理
                start_time = time.time()
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        generation_config=generation_config,
                        do_sample=False,
                        return_dict_in_generate=True
                    )
                end_time = time.time()
                
                tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                response = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                prefill_time = 0.0
                decode_time = end_time - start_time
                peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                
                # 清理临时变量
                del inputs, out_ids, tokens

            # 解析输出
            pred = clean_text_response(response)
            
            # 统计结果
            correct = int(pred == sample["answer"])
            if correct:
                correct_count += 1
                if "high" in audio_rel:
                    correct_high += 1
                elif "middle" in audio_rel:
                    correct_middle += 1
            
            current_acc = (correct_count / (idx + 1)) * 100
            progress_bar.set_postfix({
                'acc': f'{current_acc:.2f}%', 
                'ans': f'{pred}/{sample["answer"]}',
                'audio_len': f'{len(audio_np)/sr:.1f}s',
                'press': press_type[:4]
            })
            
            # 记录结果
            results.append({
                "idx": idx,
                "article_id": sample.get("article_id", ""),
                "question_idx": sample.get("question_idx", idx),
                "pred": pred, 
                "gt": sample["answer"],
                "correct": correct,
                "audio_path": audio_rel,
                "subset": "high" if "high" in audio_rel else "middle",
                "kv_press_type": press_type,
                "compression_ratio": compression_ratio
            })
            
            # 添加时间统计
            timing_stats.add_record(
                prefill_time=prefill_time,
                decode_time=decode_time,
                output_tokens=output_tokens,
                input_tokens=inputs["input_ids"].shape[1] if 'inputs' in locals() and inputs is not None else 0,
                audio_duration=len(audio_np) / sr,
                audio_tokens=len(audio_np) // 320,  # 估算audio tokens
                gpu_memory_peak=peak_memory_gb * (1024**3) if peak_memory_gb else 0  # 转换为字节
            )
            
            # 清理变量并释放GPU内存，防止内存泄漏
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 清理局部变量
            if 'result' in locals():
                del result
            if 'response' in locals() and 'response' in str(type(response)):
                del response
            
            # 每10个样本进行一次深度清理
            if (idx + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 定期显示进度
            if (idx + 1) % 50 == 0:
                current_stats = timing_stats.get_summary()
                print(f"\n样本 {idx+1}/{len(benchmark)} - 当前准确率: {current_acc:.2f}%")
                print(f"平均推理时间: {current_stats.get('avg_total_time', 0):.3f}s")
                print(f"平均tokens/sec: {current_stats.get('avg_tokens_per_sec', 0):.2f}")
                if torch.cuda.is_available():
                    print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except KeyboardInterrupt:
            print("用户中断评估")
            break
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            # 清理内存后再继续
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
    
    # 计算最终准确率
    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0
    
    # 计算F1 Score (多分类)
    if total > 0:
        # 获取预测值和真实值
        y_true = [r["gt"] for r in results]
        y_pred = [r["pred"] for r in results]
        
        # 计算macro和weighted F1 Score
        try:
            if SKLEARN_AVAILABLE:
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
                        'f1_score': float(individual_f1[i]),
                        'precision': float(individual_precision[i]),
                        'recall': float(individual_recall[i])
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
                weighted_f1 = overall_acc / 100
            
        except Exception as e:
            print(f"F1 Score计算失败: {e}")
            macro_f1 = weighted_f1 = macro_precision = macro_recall = 0
            class_metrics = {}
    else:
        macro_f1 = weighted_f1 = macro_precision = macro_recall = 0
        class_metrics = {}

    # 创建结果摘要
    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "high_accuracy": correct_high / total_high * 100 if total_high > 0 else 0,
        "middle_accuracy": correct_middle / total_middle * 100 if total_middle > 0 else 0,
        "high_correct": correct_high,
        "high_total": total_high,
        "middle_correct": correct_middle,
        "middle_total": total_middle,
        # F1 Score相关指标
        "macro_f1_score": macro_f1,
        "weighted_f1_score": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "class_metrics": class_metrics,
        "kv_press_config": kv_press_config,
        "timing": timing_stats.get_summary()
    }    # 保存结果
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    # 转换NumPy类型为JSON可序列化的类型  
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 保存时间统计
    timing_stats.export_to_json(timing_output_file)
    
    # 输出结果摘要
    print(f"\n=== RACE KV Press 评测结果摘要 ===")
    print(f"总样本数: {total}")
    print(f"总准确率: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    if total_high > 0:
        print(f"HIGH集准确率: {correct_high/total_high*100:.2f}% ({correct_high}/{total_high})")
    if total_middle > 0:
        print(f"MIDDLE集准确率: {correct_middle/total_middle*100:.2f}% ({correct_middle}/{total_middle})")
    
    # 显示F1 Score结果
    print(f"\n=== F1 Score 指标 ===")
    print(f"Macro F1 Score: {summary['macro_f1_score']:.4f}")
    print(f"Weighted F1 Score: {summary['weighted_f1_score']:.4f}")
    print(f"Macro Precision: {summary['macro_precision']:.4f}")
    print(f"Macro Recall: {summary['macro_recall']:.4f}")
    
    if summary['class_metrics']:
        print(f"\n=== 各类别指标 ===")
        for class_name, metrics in summary['class_metrics'].items():
            print(f"{class_name.upper()}: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    
    timing_summary = timing_stats.get_summary()
    print(f"\n=== KV Press 性能统计 ===")
    print(f"KV Press类型: {press_type}")
    print(f"压缩比率: {compression_ratio}")
    print(f"平均推理时间: {timing_summary.get('avg_total_time', 0):.4f}秒")
    print(f"平均Prefill时间: {timing_summary.get('avg_prefill_time', 0):.4f}秒 ({timing_summary.get('prefill_percentage', 0):.1f}%)")
    print(f"平均Decode时间: {timing_summary.get('avg_decode_time', 0):.4f}秒 ({timing_summary.get('decode_percentage', 0):.1f}%)")
    print(f"平均吞吐量: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/秒")
    
    if 'gpu_memory_stats' in timing_summary:
        gpu_stats = timing_summary['gpu_memory_stats']
        print(f"\n=== GPU内存统计 ===")
        print(f"初始GPU内存: {gpu_stats['initial_memory_gb']:.2f} GB")
        print(f"峰值GPU内存: {gpu_stats['peak_memory_gb']:.2f} GB")
        print(f"平均当前内存: {gpu_stats['avg_current_memory_gb']:.2f} GB")
        print(f"最大峰值内存: {gpu_stats['max_peak_memory_gb']:.2f} GB")
    
    print(f"\n结果已保存到: {output_file}")
    print(f"时间统计已保存到: {timing_output_file}")
    
    # 最终内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
