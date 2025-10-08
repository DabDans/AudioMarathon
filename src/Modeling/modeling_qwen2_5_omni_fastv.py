import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.hub import cached_file
from configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniVisionEncoderConfig,
)


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


Qwen2_5Omni_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2_5OmniConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2_5Omni Model outputting raw hidden-states without any specific head on top.",
    Qwen2_5Omni_START_DOCSTRING,
)
class Qwen2_5OmniPreTrainedModel(PreTrainedModel):
    config_class = Qwen2_5OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[0][i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, token_indices.shape[1])

        return list(_iter())

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        audio_token_id = self.config.audio_token_index
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_video and not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    elif min_ed == ed_video and use_audio_in_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas



@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2_5OmniAudioAttention(nn.Module):

    def __init__(
        self,
        config: Qwen2_5OmniAudioEncoderConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)

        attention_mask = torch.full(
            [1, seq_length, key_states.shape[1]],
            torch.finfo(query_states.dtype).min,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states).transpose(0, 1).reshape(seq_length, self.embed_dim)


        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen2_5OmniAudioFlashAttention2(Qwen2_5OmniAudioAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        seq_length, all_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(seq_length, self.num_heads, -1)

        key_states = self.k_proj(hidden_states)
        key_states = key_states.reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states)
        value_states = value_states.reshape(seq_length, self.num_heads, -1)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
        )
        attn_output = attn_output.reshape(seq_length, all_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen2_5OmniAudioSdpaAttention(Qwen2_5OmniAudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        attention_mask = torch.zeros(
            [1, seq_length, key_states.shape[0]], device=query_states.device, dtype=torch.bool
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAudioAttention,
    "flash_attention_2": Qwen2_5OmniAudioFlashAttention2,
    "sdpa": Qwen2_5OmniAudioSdpaAttention,
}


class Qwen2_5OmniAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES[config._attn_implementation](config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


QWEN_OMNI_AUDIO_INPUTS = r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

            feature_lens: [B], torch.LongTensor , mel length

            aftercnn_lens : [B], torch.LongTensor , mel length after cnn
        """


class Qwen2_5OmniAudioEncoder(Qwen2_5OmniPreTrainedModel):

    config_class = Qwen2_5OmniAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    @add_start_docstrings_to_model_forward(QWEN_OMNI_AUDIO_INPUTS)
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        for idx, encoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    cu_seqlens,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    cu_seqlens,
                )

            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return BaseModelOutput(last_hidden_state=token_audio)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2_5OmniVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniVisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _apply_rotary_pos_emb_flashatt(self, tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        tensor_ = tensor.float()
        cos = freqs.cos()
        sin = freqs.sin()
        output = apply_rotary_emb(tensor_, cos, sin).type_as(tensor)
        return output

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = self._apply_rotary_pos_emb_flashatt(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_flashatt(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


QWEN2_5_OMNI_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniVisionAttention,
    "flash_attention_2": Qwen2_5OmniVisionFlashAttention2,
    "sdpa": Qwen2_5OmniVisionSdpaAttention,
}


class Qwen2_5OmniVisionBlock(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = QWEN2_5_OMNI_VISION_ATTENTION_CLASSES[config._attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5OmniMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5OmniVisionEncoder(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniVisionEncoderConfig
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]

    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    rotary_pos_emb=rotary_pos_emb,
                )
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5OmniRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5OmniAttention(nn.Module):

    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, query_states, key_states, value_states


class Qwen2MLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniFlashAttention2(Qwen2_5OmniAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_weights = None
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, query_states.permute(0, 2, 1, 3), key_states.permute(0, 2, 1, 3), value_states.permute(0, 2, 1, 3)


class Qwen2_5OmniSdpaAttention(Qwen2_5OmniAttention):


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "Qwen2_5OmniModel is using Qwen2_5OmniSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()


        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, query_states, key_states, value_states


QWEN2_5_OMNI_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAttention,
    "flash_attention_2": Qwen2_5OmniFlashAttention2,
    "sdpa": Qwen2_5OmniSdpaAttention,
}


class Qwen2_5OmniDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_OMNI_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, query_states, key_states, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        outputs += (query_states, key_states, value_states)
        return outputs


QWEN2_5OMNI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`{config_class}`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2.5OmniThinker Model outputting raw hidden-states without any specific head on top.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniTextConfig"),
)
class Qwen2_5OmniThinkerTextModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniTextConfig
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        K = self.config.image_layer_idx if self.config.image_layer_idx else False
        M = self.config.audio_layer_idx if self.config.audio_layer_idx else None
        output_attentions=False
        random=self.config.random
        frame=self.config.frame
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if K and (random==False):
                    ratio = self.config.image_prune_ratio
                    retain_token_num = None

                    FASTV_image_token_length = self.config.image_token_num if self.config.image_token_num else None
                    image_start = self.config.image_token_start if self.config.image_token_start else None
                    image_end = image_start + FASTV_image_token_length
                    if idx<K:
                        casual_mask=attention_mask
                    elif idx == K and seq_length > 1:
                        device = hidden_states.device
                        last_attention=layer_outputs[1]
                        last_layer_attention_avg = torch.mean(last_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        last_layer_attention_avg_last_tok_image=last_layer_attention_avg_last_tok[image_start:image_end]

                        if ratio is not None:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(int(FASTV_image_token_length * (1 - ratio))).indices + image_start
                        else:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(retain_token_num).indices + image_start

                        keep_indexs = torch.cat((torch.arange(image_start,device=device), top_attention_rank_index, torch.arange(image_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:] 
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==K-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                 )
                elif M and (random==False) and (frame==False):
                    ratio = self.config.audio_prune_ratio
                    retain_token_num = None

                    FASTV_audio_token_length = self.config.audio_token_num if self.config.audio_token_num else None
                    audio_start = self.config.audio_token_start if self.config.audio_token_start else None
                    audio_end = audio_start + FASTV_audio_token_length
                    if idx<M:
                        casual_mask=attention_mask
                    elif idx == M and seq_length > 1:
                        device = hidden_states.device
                        last_attention=layer_outputs[1]
                        last_layer_attention_avg = torch.mean(last_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        last_layer_attention_avg_last_tok_audio=last_layer_attention_avg_last_tok[audio_start:audio_end]

                        if ratio is not None:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_audio.topk(int(FASTV_audio_token_length * (1 - ratio))).indices + audio_start
                        else:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_audio.topk(retain_token_num).indices + audio_start

                        keep_indexs = torch.cat((torch.arange(audio_start,device=device), top_attention_rank_index, torch.arange(audio_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions)
                    if M and idx==M-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )
                elif K and (random==True):
                    ratio = self.config.image_prune_ratio
                    retain_token_num = None
                    FASTV_image_token_length = self.config.image_token_num if self.config.image_token_num else None
                    image_start = self.config.image_token_start if self.config.image_token_start else None
                    image_end = image_start + FASTV_image_token_length
                    if idx<K:
                        casual_mask=attention_mask
                    elif idx == K and seq_length > 1:
                        device = hidden_states.device
                        image_indices = torch.arange(image_start, image_end, device=device)
                        perm = torch.randperm(FASTV_image_token_length, device=device)
                        num_tokens_to_keep = max(1, int(FASTV_image_token_length * (1-ratio)))
                        selected_image_indices = image_indices[perm[:num_tokens_to_keep]]
                        keep_indexs = torch.cat((torch.arange(image_start,device=device),selected_image_indices, torch.arange(image_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==K-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                 )
                elif M and (random==True) and (frame==False):
                    ratio = self.config.audio_prune_ratio
                    retain_token_num = None
                    FASTV_audio_token_length = self.config.audio_token_num if self.config.audio_token_num else None
                    audio_start = self.config.audio_token_start if self.config.audio_token_start else None
                    audio_end = audio_start + FASTV_audio_token_length
                    if idx<M:
                        casual_mask=attention_mask
                    elif idx == M and seq_length > 1:
                        device = hidden_states.device
                        audio_indices = torch.arange(audio_start, audio_end, device=device)
                        perm = torch.randperm(FASTV_audio_token_length, device=device)
                        num_tokens_to_keep = max(1, int(FASTV_audio_token_length * (1-ratio)))
                        selected_audio_indices = audio_indices[perm[:num_tokens_to_keep]]
                        keep_indexs = torch.cat((torch.arange(audio_start,device=device),selected_audio_indices, torch.arange(audio_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==M-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,   
                 )
                elif M and (frame==True) and (random==False):
                    ratio = self.config.audio_prune_ratio
                    retain_token_num = None
                    FASTV_audio_token_length = self.config.audio_token_num if self.config.audio_token_num else None
                    audio_start = self.config.audio_token_start if self.config.audio_token_start else None
                    audio_end = audio_start + FASTV_audio_token_length
                    if idx<M:
                        casual_mask=attention_mask
                    elif idx == M and seq_length > 1:
                        device = hidden_states.device
                        audio_indices = torch.arange(audio_start, audio_end, device=device)
                        
                        if ratio is not None:
                            num_tokens_to_keep = max(1, int(FASTV_audio_token_length * (1 - ratio)))
                            
                            if num_tokens_to_keep >= FASTV_audio_token_length:
                                selected_audio_indices = audio_indices
                            else:
                                step = FASTV_audio_token_length / num_tokens_to_keep
                                selected_indices = []
                                for i in range(num_tokens_to_keep):
                                    idx_to_select = int(i * step)
                                    if idx_to_select < FASTV_audio_token_length:
                                        selected_indices.append(idx_to_select)
                                
                                selected_mask = torch.zeros(FASTV_audio_token_length, dtype=torch.bool, device=device)
                                selected_mask[selected_indices] = True
                                selected_audio_indices = audio_indices[selected_mask]
                        else:
                            if retain_token_num is not None:
                                num_tokens_to_keep = min(retain_token_num, FASTV_audio_token_length)
                                step = FASTV_audio_token_length / num_tokens_to_keep
                                selected_indices = []
                                for i in range(num_tokens_to_keep):
                                    idx_to_select = int(i * step)
                                    if idx_to_select < FASTV_audio_token_length:
                                        selected_indices.append(idx_to_select)
                                
                                selected_mask = torch.zeros(FASTV_audio_token_length, dtype=torch.bool, device=device)
                                selected_mask[selected_indices] = True
                                selected_audio_indices = audio_indices[selected_mask]
                            else:
                                selected_audio_indices = audio_indices
                        
                        keep_indexs = torch.cat((torch.arange(audio_start,device=device), selected_audio_indices, torch.arange(audio_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]  
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==M-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )
                else:
                    causal_mask=attention_mask
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )


                if K and idx == K - 1 and seq_length > 1 :
                     self.last_attention = layer_outputs[1]
                if M and idx== M - 1 and seq_length > 1:
                     self.last_attention = layer_outputs[1]
            hidden_states = layer_outputs[0]
            output_attentions=False

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen25OmniThinkerText. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):

            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5OmniConfig,
        past_key_values: Cache,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.get_text_config().sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.get_text_config().sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


QWEN2_5OMNITHINKER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing images).
        pixel_values_videos(`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing videos).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`list(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.""",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniThinkerConfig"),
)
class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniThinkerConfig
    base_model_prefix = "thinker"
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )

        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Qwen2_5OmniThinkerCausalLMOutputWithPast, config_class="Qwen2_5OmniThinkerConfig"
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_ids is not None and input_ids.shape[1] != 1:
            if input_features is not None:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                audio_outputs = self.audio_tower(
                    input_features,
                    feature_lens=feature_lens,
                    aftercnn_lens=audio_feat_lengths,
                )
                audio_features = audio_outputs.last_hidden_state
                if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                    raise ValueError("length of audio_features should match audio_output_lengths")
                audio_mask = (
                    (input_ids == self.config.audio_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs



@dataclass
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    thinker_reply_part: torch.FloatTensor = None


@add_start_docstrings(
    "The bare Qwen2.5OmniTalker Model outputting raw hidden-states without any specific head on top.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniTalkerConfig"),
)
class Qwen2_5OmniTalkerModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniTalkerConfig
    _no_split_modules = ["Qwen2_5OmniTalkerDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if K and (random==False):
                    ratio = self.config.image_prune_ratio
                    retain_token_num = None
                    FASTV_image_token_length = self.config.image_token_num if self.config.image_token_num else None
                    image_start = self.config.image_token_start if self.config.image_token_start else None
                    image_end = image_start + FASTV_image_token_length
                    if idx<K:
                        casual_mask=attention_mask
                    elif idx == K and seq_length > 1:
                        device = hidden_states.device
                        last_attention=layer_outputs[1]
                        last_layer_attention_avg = torch.mean(last_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        last_layer_attention_avg_last_tok_image=last_layer_attention_avg_last_tok[image_start:image_end]

                        if ratio is not None:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(int(FASTV_image_token_length * (1 - ratio))).indices + image_start
                        else:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(retain_token_num).indices + image_start

                        keep_indexs = torch.cat((torch.arange(image_start,device=device), top_attention_rank_index, torch.arange(image_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==K-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings = position_embeddings,
                 )
                elif M and (random==False) and (frame==False):
                    ratio = self.config.audio_prune_ratio
                    retain_token_num = None
                    FASTV_audio_token_length = self.config.audio_token_num if self.config.audio_token_num else None
                    audio_start = self.config.audio_token_start if self.config.audio_token_start else None
                    audio_end = audio_start + FASTV_audio_token_length
                    if idx<M:
                        casual_mask=attention_mask
                    elif idx == M and seq_length > 1:
                        device = hidden_states.device
                        last_attention=layer_outputs[1]
                        last_layer_attention_avg = torch.mean(last_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        last_layer_attention_avg_last_tok_audio=last_layer_attention_avg_last_tok[audio_start:audio_end]

                        if ratio is not None:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_audio.topk(int(FASTV_audio_token_length * (1 - ratio))).indices + audio_start
                        else:
                            top_attention_rank_index = last_layer_attention_avg_last_tok_audio.topk(retain_token_num).indices + audio_start

                        keep_indexs = torch.cat((torch.arange(audio_start,device=device), top_attention_rank_index, torch.arange(audio_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions)
                    if M and idx==M-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )
                elif K and (random==True):
                    ratio = self.config.image_prune_ratio
                    retain_token_num = None
                    FASTV_image_token_length = self.config.image_token_num if self.config.image_token_num else None
                    image_start = self.config.image_token_start if self.config.image_token_start else None
                    image_end = image_start + FASTV_image_token_length
                    if idx<K:
                        casual_mask=attention_mask
                    elif idx == K and seq_length > 1:
                        device = hidden_states.device
                        image_indices = torch.arange(image_start, image_end, device=device)
                        perm = torch.randperm(FASTV_image_token_length, device=device)
                        num_tokens_to_keep = max(1, int(FASTV_image_token_length * (1-ratio)))
                        selected_image_indices = image_indices[perm[:num_tokens_to_keep]]
                        keep_indexs = torch.cat((torch.arange(image_start,device=device),selected_image_indices, torch.arange(image_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==K-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )
                elif M and (random==True) and (frame==False):
                    ratio = self.config.audio_prune_ratio
                    retain_token_num = None
                    FASTV_audio_token_length = self.config.audio_token_num if self.config.audio_token_num else None
                    audio_start = self.config.audio_token_start if self.config.audio_token_start else None
                    audio_end = audio_start + FASTV_audio_token_length
                    if idx<M:
                        casual_mask=attention_mask
                    elif idx == M and seq_length > 1:
                        device = hidden_states.device
                        audio_indices = torch.arange(audio_start, audio_end, device=device)
                        perm = torch.randperm(FASTV_audio_token_length, device=device)
                        num_tokens_to_keep = max(1, int(FASTV_audio_token_length * (1-ratio)))
                        selected_audio_indices = audio_indices[perm[:num_tokens_to_keep]]
                        keep_indexs = torch.cat((torch.arange(audio_start,device=device),selected_audio_indices, torch.arange(audio_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]   
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==M-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )
                elif M and (frame==True) and (random==False):
                    ratio = self.config.audio_prune_ratio
                    retain_token_num = None
                    FASTV_audio_token_length = self.config.audio_token_num if self.config.audio_token_num else None
                    audio_start = self.config.audio_token_start if self.config.audio_token_start else None
                    audio_end = audio_start + FASTV_audio_token_length
                    if idx<M:
                        casual_mask=attention_mask
                    elif idx == M and seq_length > 1:
                        device = hidden_states.device
                        audio_indices = torch.arange(audio_start, audio_end, device=device)
                        
                        if ratio is not None:
                            num_tokens_to_keep = max(1, int(FASTV_audio_token_length * (1 - ratio)))
                            
                            if num_tokens_to_keep >= FASTV_audio_token_length:
                                selected_audio_indices = audio_indices
                            else:
                                step = FASTV_audio_token_length / num_tokens_to_keep
                                selected_indices = []
                                for i in range(num_tokens_to_keep):
                                    idx_to_select = int(i * step)
                                    if idx_to_select < FASTV_audio_token_length:
                                        selected_indices.append(idx_to_select)
                                
                                selected_mask = torch.zeros(FASTV_audio_token_length, dtype=torch.bool, device=device)
                                selected_mask[selected_indices] = True
                                selected_audio_indices = audio_indices[selected_mask]
                        else:
                            if retain_token_num is not None:
                                num_tokens_to_keep = min(retain_token_num, FASTV_audio_token_length)
                                step = FASTV_audio_token_length / num_tokens_to_keep
                                selected_indices = []
                                for i in range(num_tokens_to_keep):
                                    idx_to_select = int(i * step)
                                    if idx_to_select < FASTV_audio_token_length:
                                        selected_indices.append(idx_to_select)
                                
                                selected_mask = torch.zeros(FASTV_audio_token_length, dtype=torch.bool, device=device)
                                selected_mask[selected_indices] = True
                                selected_audio_indices = audio_indices[selected_mask]
                            else:
                                selected_audio_indices = audio_indices
                        
                        keep_indexs = torch.cat((torch.arange(audio_start,device=device), selected_audio_indices, torch.arange(audio_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = position_ids[:, :, keep_indexs]
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==M-1 and seq_length>1:
                        output_attentions=False
                    else:
                        output_attentions=False
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )
                else:
                    causal_mask=attention_mask
                    layer_outputs = decoder_layer(
                     hidden_states,
                     attention_mask=causal_mask,
                     position_ids=position_ids,
                     past_key_value=past_key_values,
                     output_attentions=output_attentions,
                     use_cache=use_cache,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                 )


                if K and idx == K - 1 and seq_length > 1 :
                     self.last_attention = layer_outputs[1]
                if M and idx== M - 1 and seq_length > 1:
                     self.last_attention = layer_outputs[1]
            hidden_states = layer_outputs[0]
            output_attentions=False

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen25OmniTalker. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5OmniConfig,
        past_key_values: Cache,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.get_text_config().sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.get_text_config().sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)

        self.thinker_to_talker_proj = nn.Linear(config.embedding_size, config.hidden_size)

        self.model = Qwen2_5OmniTalkerModel(config)
        self.codebook_size = config.vocab_size
        self.codec_head = nn.Linear(config.hidden_size, self.codebook_size, bias=False)

        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_eos_token = config.tts_codec_end_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.codec_mask_token = config.tts_codec_mask_token_id

        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id

        self.spatial_merge_size = self.config.spatial_merge_size
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Qwen2_5OmniTalkerCausalLMOutputWithPast, config_class="Qwen2_5OmniTalkerConfig"
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        input_text_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5OmniTalkerCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2_5OmniTalkerForConditionalGeneration

        >>> model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )

                inputs_embeds[:, -1, :] += self.get_input_embeddings()(
                    torch.tensor([self.codec_bos_token], dtype=torch.long, device=inputs_embeds.device)
                )
                inputs_embeds[:, -2, :] += self.get_input_embeddings()(
                    torch.tensor([self.codec_pad_token], dtype=torch.long, device=inputs_embeds.device)
                )
                self.rope_deltas = rope_deltas

            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            codec_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]

        talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_lm_input,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.codec_head(hidden_states)
        logits = logits.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        inputs_embeds = model_kwargs.pop("inputs_embeds")
        model_kwargs = super()._get_initial_cache_position(input_ids, model_kwargs)
        model_kwargs["inputs_embeds"] = inputs_embeds
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part

        return model_kwargs



class Qwen2_5OmniDiTRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
            freqs = torch.stack((freqs, freqs), dim=-1)
            freqs = freqs.reshape(*freqs.shape[:-2], -1)
            freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))
            cos = freqs.cos()
            sin = freqs.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):

        if max_len is None:
            max_len = length.max().long().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)

        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        attention = self.conv(self.tanh(self.tdnn(attention)))

        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SqueezeExcitationRes2NetBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual


class ECAPA_TimeDelayNet(torch.nn.Module):

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        hidden_states = self.asp(hidden_states)

        hidden_states = self.fc(hidden_states)

        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


class DiTInputEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        self.spk_encoder = ECAPA_TimeDelayNet(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,
        drop_audio_cond: Optional[bool] = False,
        code_embed_uncond: Optional[bool] = None,
        apply_cfg: Optional[bool] = True,
    ):
        if apply_cfg:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            speaker_embedding = torch.cat([speaker_embedding, torch.zeros_like(speaker_embedding)], dim=0)
            condition_vector = torch.cat([condition_vector, torch.zeros_like(condition_vector)], dim=0)
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:
            condition_vector = torch.zeros_like(condition_vector)
            speaker_embedding = torch.zeros_like(speaker_embedding)
        condition_vector = self.spk_encoder(condition_vector).unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        hidden_states = self.proj(torch.cat((hidden_states, condition_vector, code_embed, speaker_embedding), dim=-1))

        return hidden_states


class DiTCodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = torch.zeros_like(code)
        code_embed = self.codec_embed(code)

        code_embed = torch.repeat_interleave(code_embed, repeats=self.repeats, dim=1)
        return code_embed


class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class Qwen2_5_OmniAdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states


class DiTMLP(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = nn.ModuleList(
            [
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.ff:
            hidden_states = layer(hidden_states)
        return hidden_states


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):

    def rotate_half_codec(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class DiTAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self._attn_implementation = config._attn_implementation
        self.is_causal = False

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)])

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        nterface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        attention_weights, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )

        attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)

        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)

        return attention_output


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, hidden_states, scale=1000):
        device = hidden_states.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * hidden_states.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.type_as(hidden_states)


class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.ModuleList([nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)])

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)
        return time_hidden


class DiTDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = Qwen2_5_OmniAdaLayerNormZero(config.hidden_size)

        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = DiTMLP(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(
        self, hidden_states, timestep, position_embeddings=None, block_diff=None
    ):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep)

        attn_output = self.attn(
            hidden_states=norm,
            position_embeddings=position_embeddings,
            attention_mask=(block_diff >= -float(self.look_backward_block))
            & (block_diff <= float(self.look_ahead_block)),
        )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states


class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]

        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise ValueError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=self._get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=self._get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=self._get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
            ]
        )

        self.num_layers = len(self.convs1) + len(self.convs2)

        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = hidden_states
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


@add_start_docstrings(
    "The full Qwen2.5Omni Token2WavBigVGAN model. Which take mel spectrogram as input and predict waveform.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniBigVGANConfig"),
)
class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniBigVGANConfig

    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__(config)
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 7, 1, padding=3)

        ups = [
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**layer_idx),
                        config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ]
            )
            for layer_idx, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ]
        self.ups = nn.ModuleList(ups)

        self.resblocks = nn.ModuleList(
            [
                AMPBlock(config.upsample_initial_channel // (2 ** (layer_idx + 1)), kernel_size, dilation)
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(config.upsample_initial_channel // (2**self.num_upsample_layers))
        )
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2**self.num_upsample_layers), 1, 7, 1, padding=3, bias=False
        )

    def normalize_spectrogram(self, spectrogram, max_value, min_db):
        return torch.clamp((2 * max_value) * ((spectrogram - min_db) / (-min_db)) - max_value, -max_value, max_value)

    def amplitude_to_db(self, amplitude, min_db_level):
        min_level = torch.exp(
            torch.tensor(min_db_level / 20.0 * np.log(10), device=amplitude.device, dtype=amplitude.dtype)
        )
        return 20 * torch.log10(torch.clamp(amplitude, min=min_level))

    def process_mel_spectrogram(self, mel_spectrogram):
        amplitude_spectrum = torch.exp(mel_spectrogram)
        decibel_spectrum = self.amplitude_to_db(amplitude_spectrum, -115) - 20
        return self.normalize_spectrogram(decibel_spectrum, 1, -115)

    def forward(self, mel_spectrogram):
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output

        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze().cpu()


class RungeKutta4ODESolver:
    def __init__(self, function, initial_value):
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_step(self, function, time_start, time_step, time_end, value_start, function_value_start=None):
        k1 = function_value_start if function_value_start is not None else function(time_start, value_start)
        k2 = function(time_start + time_step * self._one_third, value_start + time_step * k1 * self._one_third)
        k3 = function(time_start + time_step * self._two_thirds, value_start + time_step * (k2 - k1 * self._one_third))
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        function_value_start = function(time_start, value_start)
        return self._rk4_step(
            function, time_start, time_step, time_end, value_start, function_value_start=function_value_start
        ), function_value_start

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points):
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            delta_value, _ = self._compute_step(self.function, time_start, time_step, time_end, current_value)
            next_value = current_value + delta_value

            while current_index < len(time_points) and time_end >= time_points[current_index]:
                solution[current_index] = self._linear_interpolation(
                    time_start, time_end, current_value, next_value, time_points[current_index]
                )
                current_index += 1

            current_value = next_value

        return solution


@add_start_docstrings(
    "The full Qwen2.5Omni Token2WavDiT model. Which take speech tokens as input and predict mel spectrogram.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniDiTConfig"),
)
class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniDiTConfig
    _no_split_modules = ["DiTDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)

        self.text_embed = DiTCodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = DiTInputEmbedding(config)

        self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config.head_dim)

        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )

        self.norm_out = Qwen2_5_OmniAdaLayerNormZero_Final(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

    def _create_block_diff(self, hidden_states):
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = torch.arange(seq_len, device=hidden_states.device) // self.block_size

        block_i = block_indices.unsqueeze(1)
        block_j = block_indices.unsqueeze(0)
        block_diff = block_j - block_i

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
    ):
        batch_size = hidden_states.shape[0]
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        position_embeddings = self.rotary_embed(hidden_states)
        blockwise_difference = self._create_block_diff(hidden_states)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)

        return output

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        noise_initialization = torch.randn([1, 30000, self.mel_dim], dtype=reference_mel_spectrogram.dtype)
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(quantized_code.device)
        batch_size = reference_mel_spectrogram.shape[0]
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        if batch_size != 1:
            raise ValueError("Only batch size = 1 is currently supported")

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
                return prediction

            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        time_embedding = torch.linspace(
            initial_time, 1, num_steps, device=quantized_code.device, dtype=conditioning_vector.dtype
        )

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        return generated_mel_spectrogram


@add_start_docstrings(
    "The full Qwen2.5Omni Token2Wav model. Consists a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniToken2WavConfig"),
)
class Qwen2_5OmniToken2WavModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniToken2WavConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5OmniToken2WavDiTModel", "Qwen2_5OmniToken2WavBigVGANModel"]

    def __init__(self, config: Qwen2_5OmniToken2WavConfig):
        super().__init__(config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel must inference with fp32, but flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen2_5OmniToken2WavModel will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):

        mel_spectrogram = self.code2wav_dit_model.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        waveform = self.code2wav_bigvgan_model(mel_spectrogram)

        return waveform




@add_start_docstrings(
    """
    The full Qwen2.5Omni model, a multimodal model composed of 3 sub-models:
    - [`Qwen2_5OmniThinkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes text, audio, image, video as input and predict text tokens.
    - [`Qwen2_5OmniTalkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes thinker hidden states and response as input and predict speech tokens.
    - [`Qwen2_5OmniToken2WavModel`]:
    a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.
    """,
    QWEN2_5OMNI_START_DOCSTRING.format(config_class=Qwen2_5OmniConfig),
)
class Qwen2_5OmniForConditionalGeneration(Qwen2_5OmniPreTrainedModel, GenerationMixin):
    config_class = Qwen2_5OmniConfig
    _no_split_modules = [
        "Qwen2_5OmniTalkerForConditionalGeneration",
        "Qwen2_5OmniToken2WavModel",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)

        self.has_talker = config.enable_audio_output
        self.speaker_map = {}
        if config.enable_audio_output:
            self.enable_talker()

    def enable_talker(self):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config)
        self.token2wav.float()
        self.has_talker = True

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        spk_path = cached_file(
            pretrained_model_name_or_path,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{spk_path} not exists""")
        model.load_speakers(spk_path)

        return model

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            speaker (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if speaker not in self.speaker_map:
            raise ValueError(f"{speaker} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value

        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[speaker]

        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(self.talker.device)
        thinker_token_embeds = [
            token_hidden_states[0].to(self.talker.device) for token_hidden_states in thinker_result.hidden_states
        ]
        thinker_hidden_states = [
            token_hidden_states[-1].to(self.talker.device) for token_hidden_states in thinker_result.hidden_states
        ]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(self.talker.device),
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.talker.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token, device=self.talker.device),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.talker.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.talker.device),
            ],
            dim=1,
        )

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device)
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(self.talker.device)
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_embedding = thinker_embed_tokens(
            torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.thinker.device)
        ).to(self.talker.device)

        pad_embedding = thinker_embed_tokens(
            torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.thinker.device)
        ).to(self.talker.device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        talker_attention_mask = None
        if "attention_mask" in kwargs:
            talker_attention_mask = torch.cat(
                [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
            ).to(self.talker.device)

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(self.talker.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

        if self.token2wav.dtype != torch.float:
            self.token2wav.float()

        wav = self.token2wav(
            talker_generate_codes.to(self.token2wav.device),
            conditioning=speaker_params["cond"].to(self.token2wav.device).float(),
            reference_mel=speaker_params["ref_mel"].to(self.token2wav.device).float(),
            **token2wav_kwargs,
        )

        return thinker_result.sequences, wav.float()


__all__ = [
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniThinkerTextModel",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
]
