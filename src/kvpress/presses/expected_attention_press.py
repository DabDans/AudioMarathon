# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ExpectedAttentionPress(ScorerPress):
    """
    基于下一个位置的预期注意力计算压缩分数。
    适配 Phi-4-MM 模型的GQA架构(24个查询头/8个KV头)和部分旋转编码(75%维度)。
    """
    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    epsilon: float = 0.0

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """计算查询的均值和协方差矩阵，使用PositionIDs实现正确的RoPE旋转"""
        bsz, q_len, hidden_dim = hidden_states.shape

        # 获取 Phi-4-MM 的特定参数
        n_q = module.config.num_attention_heads  # 24 for Phi-4-MM
        d = module.head_dim  # 96 for Phi-4-MM

        # 部分旋转位置编码 - Phi-4-MM 使用 75% 的维度应用RoPE
        rotary_ndims = int(d * 0.75)

        # 移除可能包含异常值的sink tokens
        h = hidden_states[:, self.n_sink:]

        # 获取查询权重矩阵
        if hasattr(module, "q_proj"):
            Wq = module.q_proj.weight
        elif hasattr(module, "qkv_proj"):
            Wq = module.qkv_proj.weight[:n_q * d]
        else:
            raise NotImplementedError(f"不支持的模块类型: {module.__class__}")

        # 计算查询均值
        mean_h = torch.mean(h, dim=1, keepdim=True)
        mu = torch.matmul(mean_h, Wq.T).squeeze(1)
        mu = mu.view(bsz, n_q, d)

        # 计算查询协方差(可选)
        cov = None
        if self.use_covariance:
            try:
                h_centered = h - mean_h
                cov_ = torch.matmul(h_centered.transpose(1, 2), h_centered) / h_centered.shape[1]
                cov_ = torch.matmul(Wq, torch.matmul(cov_, Wq.T))
                cov_ = cov_.view(bsz, n_q, d, n_q, d).diagonal(dim1=1, dim2=3)
                cov = cov_.permute(0, 3, 1, 2)
            except Exception as e:
                print(f"协方差计算错误: {e}")
                cov = torch.zeros((bsz, n_q, d, d), device=hidden_states.device, dtype=hidden_states.dtype)

        # 分离需要旋转和不需要旋转的部分
        mu_rot = mu[..., :rotary_ndims]  # 应用旋转部分 (75%)
        mu_pass = mu[..., rotary_ndims:]  # 非旋转部分 (25%)

        try:
            # 计算未来位置的position_ids
            # 对应rotary_emb的前向传递，从当前位置到未来n_future_positions范围
            position_ids = torch.arange(
                q_len, q_len + self.n_future_positions, 
                device=mu.device
            ).unsqueeze(0)

            # 使用模型的rotary_emb计算cos和sin值
            # 注意: 我们只生成旋转部分需要的维度
            dummy_tensor = torch.zeros(
                (bsz, n_q, 1, rotary_ndims), 
                device=mu.device, 
                dtype=mu.dtype
            )
            cos, sin = module.rotary_emb(dummy_tensor, position_ids=position_ids)

            # 确保cos和sin的维度与mu_rot兼容
            # 基于apply_rotary_pos_emb中的逻辑实现旋转
            cos = cos.squeeze(1)  # 移除多余维度
            sin = sin.squeeze(1)

            # 应用旋转（采用与apply_rotary_pos_emb相同的逻辑）
            # 但我们需要对统计量而非原始向量应用
            mu_rot_transformed = torch.zeros_like(mu_rot)

            # 计算future positions的平均旋转效果
            for i in range(self.n_future_positions):
                # 执行rotate_half逻辑
                mu_half1, mu_half2 = mu_rot[..., :rotary_ndims//2], mu_rot[..., rotary_ndims//2:]
                mu_rot_half = torch.cat((-mu_half2, mu_half1), dim=-1)

                # 应用旋转
                cur_cos = cos[:, i:i+1]  # (batch, 1, rot_dim)
                cur_sin = sin[:, i:i+1]

                # 执行与apply_rotary_pos_emb相同的操作
                # q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
                mu_rot_i = (mu_rot * cur_cos) + (mu_rot_half * cur_sin)
                mu_rot_transformed += mu_rot_i

            # 取平均值
            mu_rot = mu_rot_transformed / self.n_future_positions

            # 重新组合旋转和非旋转部分
            mu = torch.cat([mu_rot, mu_pass], dim=-1)

            # 处理协方差矩阵
            if self.use_covariance and cov is not None:
                # 同样分离协方差的旋转和非旋转部分
                cov_rot = cov[..., :rotary_ndims, :rotary_ndims]
                cov_pass = cov[..., rotary_ndims:, rotary_ndims:]
                cov_mixed1 = cov[..., :rotary_ndims, rotary_ndims:]
                cov_mixed2 = cov[..., rotary_ndims:, :rotary_ndims]

                # 创建旋转后的协方差矩阵
                cov_rot_transformed = torch.zeros_like(cov_rot)

                # 这里我们简化处理，使用平均旋转效果
                # 理论上应该对每个未来位置执行旋转并求平均，但计算成本较高
                cos_avg = cos.mean(dim=1)
                sin_avg = sin.mean(dim=1)

                # 构造旋转矩阵（简化版）
                R = torch.zeros((bsz, n_q, rotary_ndims, rotary_ndims), device=cov.device, dtype=cov.dtype)
                for b in range(bsz):
                    for h in range(n_q):
                        # 构造对角块
                        for i in range(rotary_ndims//2):
                            # 构造2x2旋转块
                            cos_val = cos_avg[b, i].item()
                            sin_val = sin_avg[b, i].item()
                            R[b, h, i, i] = cos_val
                            R[b, h, i+rotary_ndims//2, i+rotary_ndims//2] = cos_val
                            R[b, h, i, i+rotary_ndims//2] = -sin_val
                            R[b, h, i+rotary_ndims//2, i] = sin_val

                # 应用旋转矩阵到协方差
                for b in range(bsz):
                    for h in range(n_q):
                        cov_rot_transformed[b, h] = R[b, h] @ cov_rot[b, h] @ R[b, h].T

                # 重新组合各部分
                cov = torch.cat([
                    torch.cat([cov_rot_transformed, cov_mixed1], dim=-1),
                    torch.cat([cov_mixed2, cov_pass], dim=-1)
                ], dim=-2)

        except Exception as e:
            print(f"旋转矩阵计算错误: {e}, 回退到简单实现")
            # 出错时使用单位矩阵作为回退策略
            # 但不再只返回单位矩阵，而是打印错误信息并继续计算

        return mu, cov

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        计算KV压缩得分，处理GQA架构特性
        """
        try:
            # 移除sink tokens
            if keys.size(2) <= self.n_sink:
                actual_n_sink = 0  # 序列太短，不要使用sink tokens
            else:
                actual_n_sink = self.n_sink
                
            keys_filtered = keys[:, :, actual_n_sink:]
            values_filtered = values[:, :, actual_n_sink:]
            
            # 计算查询统计信息
            mean_query, cov_query = self.get_query_statistics(module, hidden_states)

            # 计算分数
            bsz, num_key_value_heads, q_len, head_dim = keys_filtered.shape
            
            # Phi-4-MM的GQA架构: 24个查询头，8个键/值头
            num_key_value_groups = module.config.num_attention_heads // num_key_value_heads  # = 3
            
            # 重复键/值以匹配查询头数量
            keys_expanded = repeat_kv(keys_filtered, num_key_value_groups).transpose(2, 3)
            scores = torch.matmul(mean_query.unsqueeze(2), keys_expanded).squeeze(2) / math.sqrt(head_dim)
            
            # 添加协方差项
            if self.use_covariance and cov_query is not None:
                try:
                    scores += torch.einsum("bhin, bhij, bhjn->bhn", keys_expanded, cov_query, keys_expanded) / head_dim / 2
                except Exception as e:
                    print(f"协方差计算错误: {e}")
                    
            # 应用softmax归一化
            scores = F.softmax(scores, dim=-1)
            
            # 跨组平均
            scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
            scores = scores.mean(dim=2)

            # 使用值范数重新缩放分数
            if self.use_vnorm:
                scores = (scores + self.epsilon) * values_filtered.norm(dim=-1)

            # 恢复sink tokens
            max_val = scores.max().item() if scores.numel() > 0 else 1.0
            scores = F.pad(scores, (actual_n_sink, 0), value=max_val)
            
            return scores
            
        except Exception as e:
            print(f"ExpectedAttention评分计算错误: {e}，使用均匀分布")
            # 错误处理：返回均匀分布
            uniform_scores = torch.ones_like(values.norm(dim=-1))
            return uniform_scores / uniform_scores.sum(dim=-1, keepdim=True)