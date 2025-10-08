#!/usr/bin/env python3
"""
SLUE 命名实体识别评测脚本 - Qwen2.5-Omni + KV Press 版本

基于 GTZAN_qwen_kvpress.py 模板，采用相同的 KV 压缩适配器结构
"""

import os
import sys
import json
import argparse
import warnings
import random
import traceback
import contextlib
import re
import time
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import transformers
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

random.seed(42)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"使用 GPU ID: {gpu_id}")

ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "SLUE_QwenKVPress_Results")
print(f"KV Press 配置: 压缩比率={ENV_COMPRESSION_RATIO}, 压缩类型={ENV_PRESS_TYPE}")

sys.path.append("/data/hepeize05/Audio_Longbench/Code/Qwen_2.5")
# 添加修改后的 KV Press 库路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../kvpress_origin"))

try:
    from modeling_qwen2_5_omni_origin import Qwen2_5OmniForConditionalGeneration
    from processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    QWEN_AVAILABLE = True
except Exception as e:
    print(f"[警告] Qwen2.5-Omni 导入失败: {e}")
    QWEN_AVAILABLE = False

try:
    from kvpress import (
        ExpectedAttentionPress, KnormPress, ObservedAttentionPress,
        RandomPress, SnapKVPress, StreamingLLMPress, TOVAPress,
    )
    KV_PRESS_AVAILABLE = True
    print("[信息] KV Press 库加载成功")
except Exception as e:
    print(f"[警告] KV Press 导入失败: {e}")
    KV_PRESS_AVAILABLE = False

# 导入 Qwen-Omni KV Press Pipeline
try:
    from qwen_omni_kvpress_pipeline import QwenOmniKVPressAudioPipeline
    QWEN_KVPRESS_PIPELINE_AVAILABLE = True
    print("[信息] Qwen-Omni KV Press Pipeline 加载成功")
except Exception as e:
    print(f"[警告] Qwen-Omni KV Press Pipeline 导入失败: {e}")
    QWEN_KVPRESS_PIPELINE_AVAILABLE = False

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648


def parse_args():
    p = argparse.ArgumentParser(description="SLUE with Qwen2.5-Omni + KV Press")
    p.add_argument("--model-path", type=str, default="/data/hepeize05/Audio_Longbench/Code/Model/Qwen2.5-Omni-3B")
    p.add_argument("--json-file", type=str, default="/data/hepeize05/Audio_Longbench/Dataset/SLUE/merged_audio_data.json")
    p.add_argument("--audio-base-dir", type=str, default="/data/hepeize05/Audio_Longbench/Dataset/SLUE")
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument("--min-seq-len", type=int, default=128)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--use-pipeline", action="store_true", help="使用 Qwen-Omni KV Press Pipeline")
    return p.parse_args()


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    return obj


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0.0, 0.0


def verify_multimodal_compatibility(model):
    """验证模型与多模态压缩方法的兼容性"""
    try:
        print("[KVPress] 验证 Qwen2.5-Omni 兼容性...")
        
        # 检查 Qwen2.5-Omni 的 thinker.model 结构
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            print(f"[KVPress] 检测到 Qwen2.5-Omni thinker.model 结构")
            
            # 检查基本的层结构
            if hasattr(base_model, 'layers') and len(base_model.layers) > 0:
                print(f"[KVPress] 模型有 {len(base_model.layers)} 层")
                
                # 检查第一层的注意力结构
                first_layer = base_model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    print("[KVPress] 多模态压缩方法与 Qwen2.5-Omni 兼容性验证通过")
                    return True
                    
        print("[KVPress] 兼容性验证失败：模型结构不符合要求")
        return False
        
    except Exception as e:
        print(f"[KVPress] 兼容性验证出错: {e}")
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
            if verify_multimodal_compatibility(model):
                press_obj = TOVAPress(compression_ratio=compression_ratio)
                print("[KVPress] 使用修复版本的 TOVAPress（支持 Qwen2.5-Omni）")
            else:
                print("[KVPress] TOVA 兼容性检查失败，回退为 KnormPress")
                press_obj = KnormPress(compression_ratio=compression_ratio)
        elif press_type == 'snap':
            if verify_multimodal_compatibility(model):
                press_obj = SnapKVPress(compression_ratio=compression_ratio)
                print("[KVPress] 使用修复版本的 SnapKVPress（支持 Qwen2.5-Omni）")
            else:
                print("[KVPress] SnapKV 兼容性检查失败，回退为 KnormPress")
                press_obj = KnormPress(compression_ratio=compression_ratio)
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


def apply_kv_press_to_model(model, press_obj):
    """将KV Press应用到模型"""
    if press_obj is None or not KV_PRESS_AVAILABLE:
        return model
        
    try:
        # 对于 Qwen2.5-Omni，需要应用到 thinker.model
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            # 使用正确的模型访问路径
            target_model = model.thinker.model
            print(f"[KVPress] 应用压缩到 thinker.model: {type(press_obj).__name__}")
        else:
            target_model = model
            print(f"[KVPress] 应用压缩到基础模型: {type(press_obj).__name__}")
            
        # 应用 KV Press
        press_obj.apply(target_model)
        print("[KVPress] 压缩应用成功")
        return model
        
    except Exception as e:
        print(f"[KVPress] 应用压缩失败: {e}")
        traceback.print_exc()
        return model


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    return obj


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0.0, 0.0


def patch_qwen_for_kvpress(model):
    """与 GTZAN 相同：确保 base_model 和每层 self_attn 拥有兼容的 rotary_emb"""
    try:
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            return False

        if not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
            return False

        existing_rotary = getattr(base_model, 'rotary_emb', None)
        if existing_rotary is not None:
            for i, layer in enumerate(base_model.layers):
                if hasattr(layer, 'self_attn') and not hasattr(layer.self_attn, 'rotary_emb'):
                    layer.self_attn.rotary_emb = existing_rotary
            return True

        first_layer = base_model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
            src = first_layer.self_attn.rotary_emb
            if src is not None:
                base_model.rotary_emb = src
                for i, layer in enumerate(base_model.layers):
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn.rotary_emb = src
                return True

        # 创建兼容 rotary_emb
        config = None
        for cand in [model, getattr(model, 'thinker', None), base_model]:
            if cand and hasattr(cand, 'config'):
                config = cand.config
                break
        if config is None:
            return False

        class KVPressCompatibleRotaryEmbedding:
            def __init__(self, cfg):
                self.config = cfg
                self.max_position_embeddings = getattr(cfg, 'max_position_embeddings', 32768)
                self.rope_theta = getattr(cfg, 'rope_theta', 10000.0)
                self.hidden_size = getattr(cfg, 'hidden_size', 3584)
                self.num_attention_heads = getattr(cfg, 'num_attention_heads', 28)
                self.head_dim = self.hidden_size // self.num_attention_heads
                inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
                self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
                self.register_buffer("inv_freq", inv_freq)
            def forward(self, x, position_ids=None):
                seq_len = x.shape[-2] if position_ids is None else position_ids.shape[-1]
                device, dtype = x.device, x.dtype
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
                freqs = torch.outer(position_ids.float(), self.inv_freq.to(device))
                emb = torch.cat((freqs, freqs), dim=-1)
                return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

        comp = KVPressCompatibleRotaryEmbedding(config)
        base_model.rotary_emb = comp
        for i, layer in enumerate(base_model.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.rotary_emb = comp
        return True
    except Exception:
        traceback.print_exc()
        return False



def verify_kvpress_compatibility(model, press_type):
    try:
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            return False
        if not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
            return False
        if press_type in ['snap', 'tova']:
            global_rotary = getattr(base_model, 'rotary_emb', None)
            if global_rotary is None:
                return False
            for i in range(min(3, len(base_model.layers))):
                layer = base_model.layers[i]
                if not hasattr(layer, 'self_attn'):
                    return False
                if getattr(layer.self_attn, 'rotary_emb', None) is None:
                    return False
        attn = base_model.layers[0].self_attn if hasattr(base_model.layers[0], 'self_attn') else None
        if attn is None:
            return False
        for attr in ['q_proj', 'k_proj', 'v_proj']:
            if not hasattr(attn, attr):
                return False
        return True
    except Exception:
        return False


def verify_tova_multimodal_compatibility(model):
    try:
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        if base_model is None:
            return False
        rotary = getattr(base_model, 'rotary_emb', None)
        if rotary is None:
            return False
        if hasattr(rotary, 'head_dim'):
            hd = rotary.head_dim
            if hd <= 0 or hd % 2 != 0:
                return False
        return True
    except Exception:
        return False


def verify_snapkv_multimodal_compatibility(model):
    try:
        base_model = None
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
        elif hasattr(model, 'model'):
            base_model = model.model
        if base_model is None:
            return False
        rotary = getattr(base_model, 'rotary_emb', None)
        if rotary is None or not hasattr(rotary, 'forward'):
            return False
        inv = getattr(rotary, 'inv_freq', None)
        if inv is not None and hasattr(inv, 'shape'):
            if len(inv.shape) != 1 or inv.shape[0] == 0:
                return False
        return True
    except Exception:
        return False


def create_qwen_kvpress_pipeline(model, processor, press_obj):
    """创建 Qwen-Omni KV Press Pipeline"""
    if not QWEN_KVPRESS_PIPELINE_AVAILABLE:
        print("[警告] Qwen-Omni KV Press Pipeline 不可用，使用传统方法")
        return None
    
    try:
        pipeline = QwenOmniKVPressAudioPipeline(model=model, processor=processor)
        print("[信息] Qwen-Omni KV Press Pipeline 创建成功")
        return pipeline
    except Exception as e:
        print(f"[警告] Pipeline 创建失败: {e}")
        traceback.print_exc()
        return None


def process_sample_with_pipeline(pipeline, sample, press_obj, args):
    """使用 Pipeline 处理单个样本 - 使用新的消息格式"""
    try:
        audio_path = sample['path']
        q = sample['question']
        a = sample['choice_a']; b = sample['choice_b']
        c = sample['choice_c']; d = sample['choice_d']
        gt = sample['answer_gt']
        
        # 构建消息格式（而不是简单的提示）
        sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are a helpful assistant."
        
        instruction = "Listen to this audio and select the correct answer about named entities."
        format_hint = "Respond with only the letter of the correct option (A, B, C, or D)."
        options = f"A) {a}\nB) {b}\nC) {c}\nD) {d}"
        user_text = f"{instruction}\n\nQuestion: {q}\n\nOptions:\n{options}\n\n{format_hint}"
        
        # 构建完整的消息列表
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": user_text}
            ]},
        ]
        
        # 使用 pipeline 处理（使用新的消息格式）
        start_time = time.time()
        result = pipeline(
            messages=messages,  # 使用预构建的消息而不是简单的prompt
            press=press_obj,
            max_new_tokens=args.max_new_tokens,
            compression_ratio=ENV_COMPRESSION_RATIO if press_obj else None,
            measure_time=True,
            do_sample=False
        )
        end_time = time.time()
        
        # 提取结果
        output_text = result["generated_text"]
        pred_choice = extract_answer_choice(output_text)
        is_correct = pred_choice == gt
        
        return {
            'id': sample['id'],
            'task_name': sample.get('task_name', 'unknown'),
            'dataset_name': sample.get('dataset_name', 'unknown'),
            'question': q,
            'choices': [a, b, c, d],
            'ground_truth': gt,
            'predicted_choice': pred_choice,
            'is_correct': is_correct,
            'raw_response': output_text,
            'timing': {
                'prefill_time': result["metrics"].get("prefill_time", 0.0),
                'decode_time': result["metrics"].get("generation_time", 0.0),
                'total_time': result["metrics"].get("total_time", end_time - start_time),
                'output_tokens': result.get("output_tokens", 0),
            },
            'compression_info': result.get("compression_info", {}),
            'pipeline_used': True,
            'message_format': 'structured'  # 标记使用了结构化消息格式
        }
        
    except Exception as e:
        print(f"[Pipeline] 处理样本失败: {e}")
        traceback.print_exc()
        return None


def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
    """初始化 KV Press 对象，基于 GTZAN 实现 - 添加智能压缩比调整"""
    if not KV_PRESS_AVAILABLE:
        print("[KVPress] 库不可用")
        return None
    
    # 移除强制压缩比限制，允许用户自定义压缩比
    original_ratio = compression_ratio

        
    print(f"[KVPress] 初始化 {press_type} 方法 (压缩比: {compression_ratio})")
    
    patch_success = patch_qwen_for_kvpress(model)
    if not patch_success:
        if press_type in ['snap', 'tova']:
            print(f"[KVPress] 错误: {press_type} 方法需要正确的 rotary_emb 配置，patch 失败")
            return None
        else:
            print("[KVPress] 警告: patch 失败，但继续尝试其他方法")
    
    try:
        press_obj = None
        
        if press_type == 'expected':
            press_obj = ExpectedAttentionPress(compression_ratio=compression_ratio)
            print("[KVPress] ExpectedAttentionPress 初始化完成")
            
        elif press_type == 'observed':
            press_obj = ObservedAttentionPress(compression_ratio=compression_ratio)
            print("[KVPress] ObservedAttentionPress 初始化完成")
            
        elif press_type == 'random':
            press_obj = RandomPress(compression_ratio=compression_ratio)
            print("[KVPress] RandomPress 初始化完成")
            
        elif press_type == 'tova':
            try:
                press_obj = TOVAPress(compression_ratio=compression_ratio)
                print("[KVPress] TOVAPress 初始化完成")
                
                if not verify_kvpress_compatibility(model, 'tova'):
                    print("[KVPress] 警告: TOVA 兼容性验证失败")
                    return None
                    
                if not verify_tova_multimodal_compatibility(model):
                    print("[KVPress] 警告: TOVA 多模态兼容性验证失败")
                    return None
                    
            except Exception as e:
                print(f"[KVPress] TOVAPress 初始化失败: {e}")
                return None
                
        elif press_type == 'snap':
            try:
                if not verify_kvpress_compatibility(model, 'snap'):
                    print("[KVPress] 错误: SnapKV 兼容性验证失败")
                    return None
                
                if not verify_snapkv_multimodal_compatibility(model):
                    print("[KVPress] 错误: SnapKV 多模态兼容性验证失败")
                    return None
                    
                press_obj = SnapKVPress(compression_ratio=compression_ratio)
                print("[KVPress] SnapKVPress 初始化完成")
                
            except Exception as e:
                print(f"[KVPress] SnapKVPress 初始化失败: {e}")
                return None
            
        elif press_type == 'streaming':
            press_obj = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
            print("[KVPress] StreamingLLMPress 初始化完成")
            
        else:
            press_obj = KnormPress(compression_ratio=compression_ratio)
            print("[KVPress] KnormPress (默认) 初始化完成")
            
        if press_obj is not None:
            print(f"[KVPress] 成功创建 {type(press_obj).__name__} 对象")
            
        return press_obj
        
    except Exception as e:
        print(f"[KVPress] 创建 {press_type} 对象失败: {e}")
        traceback.print_exc()
        return None
    if not KV_PRESS_AVAILABLE:
        print("[KVPress] 库不可用")
        return None
        
    print(f"[KVPress] 初始化 {press_type} 方法 (压缩比: {compression_ratio})")
    
    patch_success = patch_qwen_for_kvpress(model)
    if not patch_success:
        if press_type in ['snap', 'tova']:
            print(f"[KVPress] 错误: {press_type} 方法需要正确的 rotary_emb 配置，patch 失败")
            return None
        else:
            print("[KVPress] 警告: patch 失败，但继续尝试其他方法")
    
    try:
        press_obj = None
        
        if press_type == 'expected':
            press_obj = ExpectedAttentionPress(compression_ratio=compression_ratio)
            print("[KVPress] ExpectedAttentionPress 初始化完成")
            
        elif press_type == 'observed':
            press_obj = ObservedAttentionPress(compression_ratio=compression_ratio)
            print("[KVPress] ObservedAttentionPress 初始化完成")
            
        elif press_type == 'random':
            press_obj = RandomPress(compression_ratio=compression_ratio)
            print("[KVPress] RandomPress 初始化完成")
            
        elif press_type == 'tova':
            try:
                press_obj = TOVAPress(compression_ratio=compression_ratio)
                print("[KVPress] TOVAPress 初始化完成")
                
                if not verify_kvpress_compatibility(model, 'tova'):
                    print("[KVPress] 警告: TOVA 兼容性验证失败")
                    return None
                    
                if not verify_tova_multimodal_compatibility(model):
                    print("[KVPress] 警告: TOVA 多模态兼容性验证失败")
                    return None
                    
            except Exception as e:
                print(f"[KVPress] TOVAPress 初始化失败: {e}")
                return None
                
        elif press_type == 'snap':
            try:
                if not verify_kvpress_compatibility(model, 'snap'):
                    print("[KVPress] 错误: SnapKV 兼容性验证失败")
                    return None
                
                if not verify_snapkv_multimodal_compatibility(model):
                    print("[KVPress] 错误: SnapKV 多模态兼容性验证失败")
                    return None
                    
                press_obj = SnapKVPress(compression_ratio=compression_ratio)
                print("[KVPress] SnapKVPress 初始化完成")
                
            except Exception as e:
                print(f"[KVPress] SnapKVPress 初始化失败: {e}")
                return None
            
        elif press_type == 'streaming':
            press_obj = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
            print("[KVPress] StreamingLLMPress 初始化完成")
            
        else:
            press_obj = KnormPress(compression_ratio=compression_ratio)
            print("[KVPress] KnormPress (默认) 初始化完成")
            
        if press_obj is not None:
            print(f"[KVPress] 成功创建 {type(press_obj).__name__} 对象")
            
        return press_obj
        
    except Exception as e:
        print(f"[KVPress] 创建 {press_type} 对象失败: {e}")
        traceback.print_exc()
        return None


def load_slue_dataset(json_file: str, audio_base_dir: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_file):
        print(f"错误: JSON文件不存在: {json_file}")
        return []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("错误: JSON文件格式应为列表")
        return []
    dataset = []
    missing = 0
    for i, item in enumerate(data):
        rel = item.get('path', '')
        if not rel:
            continue
        full = os.path.join(audio_base_dir, rel)
        if not os.path.exists(full):
            missing += 1
            if missing <= 5:
                print(f"缺失音频: {full}")
            continue
        dataset.append({
            'path': full,
            'question': item.get('question', ''),
            'choice_a': item.get('choice_a', ''),
            'choice_b': item.get('choice_b', ''),
            'choice_c': item.get('choice_c', ''),
            'choice_d': item.get('choice_d', ''),
            'answer_gt': str(item.get('answer_gt', '')).upper(),
            'task_name': item.get('task_name', 'unknown'),
            'dataset_name': item.get('dataset_name', 'unknown'),
            'id': item.get('uniq_id', f'slue_{i}'),
        })
    if missing > 5:
        print(f"警告: 共缺失 {missing} 个音频文件")
    print(f"加载了 {len(dataset)} 个有效SLUE样本")
    return dataset


def build_slue_prompt(question: str, a: str, b: str, c: str, d: str) -> str:
    instr = "Listen to this audio and select the correct answer about named entities."
    fmt = "Respond with only the letter of the correct option (A, B, C, or D)."
    options = f"A) {a}\nB) {b}\nC) {c}\nD) {d}"
    return f"{instr}\n\nQuestion: {question}\n\nOptions:\n{options}\n\n{fmt}"


def extract_answer_choice(response: str) -> str:
    if not response:
        return ""
    s = response.strip().upper()
    if s in ['A', 'B', 'C', 'D']:
        return s
    if s.startswith('A') and len(s) <= 3:
        return 'A'
    if s.startswith('B') and len(s) <= 3:
        return 'B'
    if s.startswith('C') and len(s) <= 3:
        return 'C'
    if s.startswith('D') and len(s) <= 3:
        return 'D'
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"[([\[]?([ABCD])[)\].]?", s)
    if m:
        return m.group(1)
    m = re.search(r"(?:option|choice)\s+([ABCD])", s)
    if m:
        return m.group(1)
    return ""


def calculate_slue_metrics(predictions: List[str], ground_truths: List[str]):
    try:
        valid = [(p, g) for p, g in zip(predictions, ground_truths) if p and g]
        if not valid:
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "valid_samples": 0}
        vp, vg = zip(*valid)
        precision, recall, f1, _ = precision_recall_fscore_support(vg, vp, average='weighted', zero_division=0)
        mp, mr, mf1, _ = precision_recall_fscore_support(vg, vp, average='macro', zero_division=0)
        return {"f1_score": float(f1), "precision": float(precision), "recall": float(recall), "macro_f1": float(mf1), "valid_samples": len(vp)}
    except Exception:
        traceback.print_exc()
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "valid_samples": 0}


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR_ENV, exist_ok=True)

    method_name = "no_compress" if args.no_compress else ENV_PRESS_TYPE
    ratio_str = f"{ENV_COMPRESSION_RATIO:.3f}"
    output_file = f"{RESULTS_DIR_ENV}/slue_results_kvpress_{method_name}_{ratio_str}.json"
    timing_file = f"{RESULTS_DIR_ENV}/slue_timing_kvpress_{method_name}_{ratio_str}.json"
    print(f"结果将保存到: {output_file}")

    samples = load_slue_dataset(args.json_file, args.audio_base_dir)
    if SAMPLE_LIMIT > 0 and len(samples) > SAMPLE_LIMIT:
        samples = samples[:SAMPLE_LIMIT]
        print(f"样本数量限制为: {len(samples)}")

    if not QWEN_AVAILABLE:
        print("[错误] Qwen2.5-Omni 模块不可用")
        return

    print("加载Qwen2.5-Omni模型...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.disable_talker()

    # 音频配置注入 (GTZAN 风格)
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
        cfg = model.thinker.config
        if not hasattr(cfg, 'audio_token_id'): cfg.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(cfg, 'image_token_id'): cfg.image_token_id = 151655
        if not hasattr(cfg, 'video_token_id'): cfg.video_token_id = 151656
        if not hasattr(cfg, 'audio_bos_token_id'): cfg.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(cfg, 'audio_eos_token_id'): cfg.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
        if not hasattr(cfg, 'image_layer_idx'): cfg.image_layer_idx = False
        if not hasattr(cfg, 'audio_layer_idx'): cfg.audio_layer_idx = None
        if not hasattr(cfg, 'audio_token_num'): cfg.audio_token_num = None
        if not hasattr(cfg, 'audio_token_start'): cfg.audio_token_start = None
        if not hasattr(cfg, 'audio_prune_ratio'): cfg.audio_prune_ratio = 0
        if not hasattr(cfg, 'random'): cfg.random = False
        if not hasattr(cfg, 'frame'): cfg.frame = False
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        cfg2 = model.thinker.model.config
        if not hasattr(cfg2, 'audio_token_id'): cfg2.audio_token_id = _AUDIO_TOKEN_ID
        if not hasattr(cfg2, 'image_token_id'): cfg2.image_token_id = 151655
        if not hasattr(cfg2, 'video_token_id'): cfg2.video_token_id = 151656
        if not hasattr(cfg2, 'audio_bos_token_id'): cfg2.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
        if not hasattr(cfg2, 'audio_eos_token_id'): cfg2.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
        if not hasattr(cfg2, 'image_layer_idx'): cfg2.image_layer_idx = False
        if not hasattr(cfg2, 'audio_layer_idx'): cfg2.audio_layer_idx = None
        if not hasattr(cfg2, 'audio_token_num'): cfg2.audio_token_num = None
        if not hasattr(cfg2, 'audio_token_start'): cfg2.audio_token_start = None
        if not hasattr(cfg2, 'audio_prune_ratio'): cfg2.audio_prune_ratio = 0
        if not hasattr(cfg2, 'random'): cfg2.random = False
        if not hasattr(cfg2, 'frame'): cfg2.frame = False

    # 初始化 KVPress
    if args.no_compress:
        press_obj = None
        print("[KVPress] 已禁用压缩")
    else:
        press_obj = initialize_kv_press(model, ENV_PRESS_TYPE, ENV_COMPRESSION_RATIO, args.min_seq_len)
        if press_obj is None:
            print("[KVPress] 未启用（创建失败或不可用）")
        else:
            print(f"[KVPress] 启用 {ENV_PRESS_TYPE}, 压缩比: {ENV_COMPRESSION_RATIO}")

    # 创建 Pipeline（如果启用）
    pipeline = None
    if args.use_pipeline:
        pipeline = create_qwen_kvpress_pipeline(model, processor, press_obj)
        if pipeline:
            print("[Pipeline] Qwen-Omni KV Press Pipeline 已启用")
        else:
            print("[Pipeline] Pipeline 创建失败，使用传统方法")
            args.use_pipeline = False

    # 统计容器
    results = []
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    correct = 0

    for s in samples:
        task_type_stats[s.get('task_name', 'unknown')] += 1
        dataset_stats[s.get('dataset_name', 'unknown')] += 1

    tqdm_kwargs = {'ascii': True, 'dynamic_ncols': True, 'file': sys.stdout}
    with tqdm(total=len(samples), desc="SLUE评估 (Qwen2.5 KVPress)", **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                # 使用 Pipeline 处理样本
                if args.use_pipeline and pipeline:
                    result = process_sample_with_pipeline(pipeline, sample, press_obj, args)
                    if result:
                        results.append(result)
                        correct += int(result['is_correct'])
                        curr_acc = correct / (len(results))
                        pbar.set_postfix({
                            'acc': f'{curr_acc:.3f}', 
                            'pipeline': '✓',
                            'compress': '✓' if press_obj else '-'
                        })
                    else:
                        print(f"[Pipeline] 样本 {idx} 处理失败，跳过")
                    pbar.update(1)
                    continue

                # 传统处理方法
                audio_path = sample['path']
                q = sample['question']
                a = sample['choice_a']; b = sample['choice_b']
                c = sample['choice_c']; d = sample['choice_d']
                gt = sample['answer_gt']
                prompt = build_slue_prompt(q, a, b, c, d)
                sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are a helpful assistant that analyzes audio and answers named-entity-related questions."
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": prompt},
                    ]},
                ]

                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): text = text[0]
                inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                inputs = inputs.to(model.device)
                inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}

                # Prefill phase - 直接执行，不处理错误
                prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if prefill_start: prefill_start.record()
                
                # 直接使用压缩或不使用压缩
                use_compression = press_obj is not None
                input_ids = inputs.get('input_ids', None)
                
                # 检测音频token用于日志
                if input_ids is not None and use_compression:
                    audio_detected = (_AUDIO_BOS_TOKEN_ID in input_ids[0] and _AUDIO_EOS_TOKEN_ID in input_ids[0]) if input_ids.numel() > 0 else False
                    if audio_detected:
                        print(f"[KVPress] 检测到音频token，启用 {ENV_PRESS_TYPE} 压缩")
                    else:
                        print(f"[KVPress] 未检测到音频token，仍然启用 {ENV_PRESS_TYPE} 压缩")
                
                with torch.no_grad(), (press_obj(model) if use_compression else contextlib.nullcontext()):
                    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                        
                if prefill_end: prefill_end.record()

                # Generate phase - 直接执行，不处理错误
                gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if gen_start: gen_start.record()
                
                # 生成阶段不再注册 KVPress hooks（只在 prefill 时压缩）
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                        
                if gen_end: gen_end.record()
                if torch.cuda.is_available(): torch.cuda.synchronize()
                
                # 记录压缩状态
                compression_used = use_compression
                
                # 计算时间
                prefill_time = prefill_start.elapsed_time(prefill_end)/1000 if prefill_start else 0.0
                total_time = gen_start.elapsed_time(gen_end)/1000 if gen_start else 0.0
                decode_time = max(total_time - prefill_time, 0.0)

                output_text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if "assistant\n" in output_text:
                    output_text = output_text.split("assistant\n")[-1].strip()
                pred_choice = extract_answer_choice(output_text)
                is_correct = pred_choice == gt
                correct += int(is_correct)
                output_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                input_tokens = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0

                results.append({
                    'id': sample['id'],
                    'task_name': sample.get('task_name', 'unknown'),
                    'dataset_name': sample.get('dataset_name', 'unknown'),
                    'question': q,
                    'choices': [a, b, c, d],
                    'ground_truth': gt,
                    'predicted_choice': pred_choice,
                    'is_correct': is_correct,
                    'raw_response': output_text,
                    'timing': {
                        'prefill_time': prefill_time,
                        'decode_time': decode_time,
                        'total_time': prefill_time + decode_time,
                        'output_tokens': output_tokens,
                    },
                    'compression_info': {
                        'enabled': press_obj is not None,
                        'used': use_compression,
                        'press_type': ENV_PRESS_TYPE if press_obj else None,
                        'compression_ratio': ENV_COMPRESSION_RATIO if press_obj else None,
                        'min_seq_len': args.min_seq_len,
                        'input_tokens': input_tokens,
                    },
                    'pipeline_used': False
                })

                curr_acc = correct / (idx + 1)
                pbar.set_postfix({
                    'acc': f'{curr_acc:.3f}', 
                    'pipeline': '-',
                    'compress': '✓' if use_compression else '-'
                })
                pbar.update(1)

                del inputs, out_ids
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                traceback.print_exc()
                pbar.update(1)
                continue

    preds = [r['predicted_choice'] for r in results]
    gts = [r['ground_truth'] for r in results]
    metrics = calculate_slue_metrics(preds, gts)

    if results:
        avg_prefill = float(np.mean([r['timing']['prefill_time'] for r in results]))
        avg_decode = float(np.mean([r['timing']['decode_time'] for r in results]))
        avg_total = float(np.mean([r['timing']['total_time'] for r in results]))
    else:
        avg_prefill = avg_decode = avg_total = 0.0

    final = {
        'summary': {
            'total_samples': len(results),
            'overall_accuracy': float(np.mean([r['is_correct'] for r in results])) if results else 0.0,
            'f1_weighted': metrics.get('f1_score', 0.0),
            'f1_macro': metrics.get('macro_f1', 0.0),
            'precision_weighted': metrics.get('precision', 0.0),
            'recall_weighted': metrics.get('recall', 0.0),
            'timing_summary': {
                'avg_prefill_time': avg_prefill,
                'avg_decode_time': avg_decode,
                'avg_total_time': avg_total,
            },
            'kv_press': {
                'enabled': not args.no_compress and (press_obj is not None),
                'press_type': ENV_PRESS_TYPE if (not args.no_compress and press_obj) else None,
                'compression_ratio': ENV_COMPRESSION_RATIO if (not args.no_compress and press_obj) else None,
                'min_seq_len': args.min_seq_len,
            },
            'task_type_stats': dict(task_type_stats),
            'dataset_stats': dict(dataset_stats),
        },
        'detailed_results': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(final), f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file}")

    timing_only = {
        'total_samples': len(results),
        'timing_summary': final['summary']['timing_summary'],
        'kv_press': final['summary']['kv_press'],
    }
    with open(timing_file, 'w', encoding='utf-8') as tf:
        json.dump(convert_to_serializable(timing_only), tf, ensure_ascii=False, indent=2)
    print(f"时间统计已保存到: {timing_file}")


if __name__ == '__main__':
    main()
