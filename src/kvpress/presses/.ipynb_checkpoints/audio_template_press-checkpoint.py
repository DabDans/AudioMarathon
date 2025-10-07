import torch
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Union

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

@dataclass
class ProtectAudioPromptTemplatePress(BasePress):
    """
    音频转录模板保护压缩器
    
    这个压缩器包装任何基础Press，并在压缩过程中保护提示词模板的关键部分。
    它识别助手和用户标记之间的区域，并确保这些重要的模板部分不会被压缩。
    
    适用于音频转录任务，可与任何 KV Press 压缩方式结合使用。
    """
    
    press: BasePress = None
    protect_user_tokens: bool = True     # 保护用户提示词部分
    protect_assistant_tokens: bool = True  # 保护助手提示词部分
    protect_system_tokens: bool = True    # 保护系统提示词部分
    
    def __post_init__(self):
        assert self.press is not None, "必须提供一个基础Press实例"
        # 保存原始压缩函数
        self._original_compress = self.press.compress
        # 替换为我们的包装函数
        self.press.compress = types.MethodType(self._protected_compress, self.press)
        # 初始化标记缓存
        self._token_markers = None
        
    @property
    def compression_ratio(self):
        return self.press.compression_ratio
    
    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value
        
    def _identify_template_markers(self, model, keys):
        """
        识别模板中的关键标记位置
        """
        # 如果已经识别过，直接返回
        if self._token_markers is not None:
            return self._token_markers
            
        # 初始化标记位置字典
        markers = {
            "user_start": [],
            "user_end": [],
            "assistant_start": [],
            "assistant_end": [],
            "system_start": [],
            "system_end": [],
            "audio_start": [],
            "audio_end": []
        }
        
        # 获取模型的tokenizer
        try:
            # 尝试从模型中获取特殊token id
            user_token_ids = []
            assistant_token_ids = []
            system_token_ids = []
            audio_token_id = getattr(model, "_AUDIO_SPECIAL_TOKEN_ID", 200011)  # 音频特殊token_id
            
            # 检查是否有指定的标记ID
            if hasattr(model, "config"):
                if hasattr(model.config, "user_token_id"):
                    user_token_ids.append(model.config.user_token_id)
                if hasattr(model.config, "assistant_token_id"):
                    assistant_token_ids.append(model.config.assistant_token_id)
                if hasattr(model.config, "system_token_id"):
                    system_token_ids.append(model.config.system_token_id)
                    
            # 查找常见的特殊token模式
            if not user_token_ids:
                user_token_ids = [199991, 199992]  # 常见的用户token ID
            if not assistant_token_ids:
                assistant_token_ids = [199993, 199994]  # 常见的助手token ID
            if not system_token_ids:
                system_token_ids = [199995, 199996]  # 常见的系统token ID
            
            # 在key缓存中查找这些标记
            seq_len = keys.shape[2]
            
            # 分析每个位置
            for pos in range(seq_len):
                # 检测用户标记
                if any(self._check_token_at_position(model, pos, token_id) for token_id in user_token_ids):
                    markers["user_start"].append(pos)
                    if markers["user_end"] and markers["user_end"][-1] < pos - 1:
                        markers["user_end"].append(pos - 1)
                        
                # 检测助手标记
                if any(self._check_token_at_position(model, pos, token_id) for token_id in assistant_token_ids):
                    markers["assistant_start"].append(pos)
                    if markers["assistant_end"] and markers["assistant_end"][-1] < pos - 1:
                        markers["assistant_end"].append(pos - 1)
                        
                # 检测系统标记
                if any(self._check_token_at_position(model, pos, token_id) for token_id in system_token_ids):
                    markers["system_start"].append(pos)
                    if markers["system_end"] and markers["system_end"][-1] < pos - 1:
                        markers["system_end"].append(pos - 1)
                        
                # 检测音频标记
                if self._check_token_at_position(model, pos, audio_token_id):
                    if not markers["audio_start"]:
                        markers["audio_start"].append(pos)
                    else:
                        markers["audio_end"].append(pos)
            
            # 如果没有找到结束位置，设置为序列末尾
            if markers["user_start"] and not markers["user_end"]:
                markers["user_end"].append(seq_len - 1)
            if markers["assistant_start"] and not markers["assistant_end"]:
                markers["assistant_end"].append(seq_len - 1)
            if markers["system_start"] and not markers["system_end"]:
                markers["system_end"].append(seq_len - 1)
            if markers["audio_start"] and not markers["audio_end"]:
                markers["audio_end"].append(seq_len - 1)
                
            # 如果检测到音频特殊token
            if markers["audio_start"] and markers["audio_end"]:
                print(f"检测到音频区域: {markers['audio_start'][0]} - {markers['audio_end'][0]}")
                
            self._token_markers = markers
            return markers
            
        except Exception as e:
            print(f"识别模板标记时出错: {str(e)}")
            # 返回空标记
            self._token_markers = markers
            return markers
    
    def _check_token_at_position(self, model, position, token_id):
        """检查指定位置是否为目标token_id"""
        # 这是一个简化实现，实际使用时可能需要更复杂的检测逻辑
        # 通常需要访问模型的输入token，这里假设已经有了一种方法
        return False
    
    def _protected_compress(self, press, module, hidden_states, keys, values, attentions, kwargs):
        """
        包装compress函数，在压缩前保护模板关键部分
        """
        seq_len = keys.shape[2]
        if seq_len <= 1:
            # 序列太短，不需要压缩
            return keys, values
            
        # 识别模板标记
        markers = self._identify_template_markers(module.model, keys)
        
        # 获取音频特殊token的位置
        audio_token_indices = getattr(self, "audio_token_indices", None)
        if audio_token_indices:
            audio_start = audio_token_indices.get("start", -1)
            audio_end = audio_token_indices.get("end", -1)
            
            if audio_start >= 0 and audio_end >= 0 and audio_start <= audio_end:
                # 使用检测到的音频位置更新标记
                markers["audio_start"] = [audio_start]
                markers["audio_end"] = [audio_end]
                print(f"使用已知音频区域: {audio_start} - {audio_end}")
        
        # 是否进行压缩的判断函数
        def should_compress_position(pos):
            # 默认压缩所有位置
            should_compress = True
            
            # 检查是否在音频区域
            if markers["audio_start"] and markers["audio_end"]:
                if markers["audio_start"][0] <= pos <= markers["audio_end"][0]:
                    # 在音频区域内，使用音频压缩率
                    return True
            
            # 检查是否在用户提示词区域
            if self.protect_user_tokens:
                for start, end in zip(markers["user_start"], markers["user_end"]):
                    if start <= pos <= end:
                        should_compress = False
                        break
                        
            # 检查是否在助手提示词区域
            if should_compress and self.protect_assistant_tokens:
                for start, end in zip(markers["assistant_start"], markers["assistant_end"]):
                    if start <= pos <= end:
                        should_compress = False
                        break
                        
            # 检查是否在系统提示词区域
            if should_compress and self.protect_system_tokens:
                for start, end in zip(markers["system_start"], markers["system_end"]):
                    if start <= pos <= end:
                        should_compress = False
                        break
                        
            return should_compress
        
        # 创建位置掩码
        positions_to_compress = torch.tensor(
            [should_compress_position(i) for i in range(seq_len)],
            device=keys.device, dtype=torch.bool
        )
        
        # 如果没有需要压缩的位置，直接返回原始键值
        if not positions_to_compress.any():
            return keys, values
            
        # 确保音频部分被压缩
        if hasattr(press, "audio_compression_ratio") and hasattr(press, "audio_token_indices"):
            # 获取原始压缩率
            orig_ratio = getattr(press, "_orig_compression_ratio", None)
            
            # 调用原始压缩函数
            compressed_keys, compressed_values = self._original_compress(module, hidden_states, keys, values, attentions, kwargs)
            
            # 恢复原始压缩率
            if orig_ratio is not None:
                press.compression_ratio = orig_ratio
                
            return compressed_keys, compressed_values
        else:
            # 如果不是音频增强型press，直接调用原始压缩
            return self._original_compress(module, hidden_states, keys, values, attentions, kwargs)
    
    @contextmanager
    def __call__(self, model):
        """
        应用模板保护压缩器
        """
        with self.press(model):
            yield