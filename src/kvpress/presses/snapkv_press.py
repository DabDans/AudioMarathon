# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV (https://arxiv.org/abs/2404.14469) use the attention of the latest window_size tokens to estimate the
    importance of the previous KV pairs. We use the default settings from:
    https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py#L24
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        计算最后window_size个查询和前q_len-window_size个键的注意力权重
        支持两种输入类型以及部分旋转位置编码
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # 获取最后window_size个查询
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV未实现{module.__class__}支持")

        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # 检测是否使用部分旋转位置编码
        rotary_ndims = head_dim
        if hasattr(module, "rotary_ndims"):
            rotary_ndims = module.rotary_ndims
        elif hasattr(module.config, "partial_rotary_factor") and module.config.partial_rotary_factor < 1.0:
            rotary_ndims = int(head_dim * module.config.partial_rotary_factor)

        # 处理位置编码 - 支持两种形式
        if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
            # 如果已经是(cos,sin)元组形式
            cos, sin = position_embeddings
            # 确保只取最后window_size个位置
            if cos.shape[1] > window_size:
                cos = cos[:, -window_size:]
                sin = sin[:, -window_size:]
        else:
            # 如果是position_ids，则使用模型的rotary_emb计算
            if position_embeddings is None:
                # 创建默认位置ID
                position_ids = torch.arange(window_size, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            else:
                position_ids = position_embeddings
                # 确保仅取最后window_size个位置ID
                if position_ids.shape[1] > window_size:
                    position_ids = position_ids[:, -window_size:]

            # 计算旋转位置编码
            if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
                cos, sin = module.rotary_emb(query_states, position_ids)
            elif hasattr(module, "cache_kwargs") and "sin" in module.cache_kwargs and "cos" in module.cache_kwargs:
                cos, sin = module.cache_kwargs["cos"], module.cache_kwargs["sin"]
                if cos.shape[1] > window_size:
                    cos = cos[:, -window_size:]
                    sin = sin[:, -window_size:]
            else:
                raise ValueError("无法获取旋转位置编码，需要rotary_emb或cache_kwargs中的cos/sin")

        # 分别处理需要旋转和不需要旋转的部分
        query_rot = query_states[..., :rotary_ndims]
        query_pass = query_states[..., rotary_ndims:]

        # 仅对部分维度应用RoPE
        query_rot = (query_rot * cos.unsqueeze(1)[..., :rotary_ndims]) + (
            rotate_half(query_rot) * sin.unsqueeze(1)[..., :rotary_ndims]
        )

        # 重组完整查询状态
        if rotary_ndims < head_dim:
            query_states = torch.cat([query_rot, query_pass], dim=-1)
        else:
            query_states = query_rot

        # 计算对前q_len-window_size个令牌的注意力
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights
    def score(self, module: nn.Module, hidden_states: torch.Tensor, 
            keys: torch.Tensor, values: torch.Tensor, 
            attentions: torch.Tensor, kwargs) -> torch.Tensor:
        
        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        
        if q_len <= self.window_size:
            # 如果序列太短，返回均匀分布
            return torch.ones(bsz, num_key_value_heads, q_len, device=keys.device) / q_len
        
        # 获取位置信息，优先使用sin/cos，其次使用position_ids
        position_embeddings = None
        if "sin" in kwargs and "cos" in kwargs:
            position_embeddings = (kwargs["cos"], kwargs["sin"])
        elif "position_ids" in kwargs:
            position_embeddings = kwargs["position_ids"]
        else:
            # 尝试从模块或缓存中找到位置信息
            position_embeddings = getattr(module, "position_ids", None)
        
        if position_embeddings is None:
            raise ValueError("需要position_ids或(cos,sin)来计算SnapKV评分")
        
        if attentions is not None:
            attn_weights = attentions[..., -self.window_size:, :-self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, position_embeddings
            )
        
        # 后续处理保持不变...
        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, 
                            padding=self.kernel_size // 2, stride=1)
        
        # 按组平均
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.mean(2)
        
        # 添加回观察窗口，确保窗口不被裁剪
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())
        
        return scores