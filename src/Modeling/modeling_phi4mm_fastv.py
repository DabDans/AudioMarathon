import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from .configuration_phi4mm import Phi4MMConfig
from .processing_phi4mm import InputMode
from .vision_siglip_navit import get_siglip_vision_model
from .speech_conformer_encoder import ConformerEncoder

import pdb


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "TBA"
_CONFIG_FOR_DOC = "Phi4MMConfig"

_IMAGE_SPECIAL_TOKEN_ID = 200010
_AUDIO_SPECIAL_TOKEN_ID = 200011
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000]


class Phi4MMImageEmbedding(nn.Module):

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__()

        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        logger.info(f"create image tower {config.img_processor}")
        enable_gradient_checkpointing = kwargs.get('enable_gradient_checkpointing', False)

        self.img_processor = get_siglip_vision_model(
            _flash_attn_2_enabled=config._attn_implementation == 'flash_attention_2'
        )

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = pe_weight.size()
        H = int(math.sqrt(L))
        assert H**2 == L
        if H % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        image_dim_out = D
        self.num_img_tokens = (H//2)**2
        self.base_feat_height_target = H

        if enable_gradient_checkpointing:
            self.img_processor.encoder.gradient_checkpointing = True

        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.image_attention_mask = None

        self.use_hd_transform = kwargs.get('use_hd_transform', False)
        self.with_learnable_separator = kwargs.get('with_learnable_separator', False)
        self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        self.freeze_img_processor = kwargs.get('freeze_img_processor', False)
        self.crop_size = kwargs.get('crop_size', 336)
        logger.info(f'freeze_img_processor = {self.freeze_img_processor}')

        self.image_token_compression_cls = kwargs.get('image_token_compression_cls', None)
        if self.image_token_compression_cls == 'avg_pool_2d':
            self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
            self.base_feat_height_reduction = 1
            self.base_feat_height_target = self.base_feat_height_target // 2
        elif self.image_token_compression_cls is None:
            self.image_token_compression = None
            self.base_feat_height_reduction = 2
        else:
            raise NotImplementedError(f'image_token_compression_cls = {self.image_token_compression_cls}, not implemented')

        assert self.use_hd_transform == self.with_learnable_separator, 'use_hd_transform and with_learnable_separator should have same value'
        if self.with_learnable_separator:
            assert self.use_hd_transform, 'learnable separator is only for hd transform'
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * self.base_feat_height_reduction**2]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2]))
            logger.info(f'learnable separator enabled for hd transform, hd_transform_order = {self.hd_transform_order}')

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.img_projection = nn.Linear(image_dim_out, hidden_size)
        elif projection_cls == 'mlp' and self.use_hd_transform:
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out * self.base_feat_height_reduction**2, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        elif projection_cls == 'mlp':
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.vocab_size = config.vocab_size
        self.img_features = None

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'

    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def set_img_attn_mask(self, image_attention_mask: torch.FloatTensor) -> None:
        self.image_attention_mask = image_attention_mask

    def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        if self.freeze_img_processor:
            with torch.no_grad():
                if attention_mask is not None:
                    img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask)
                else:
                    img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
                img_feature = img_processor_output.hidden_states[LAYER_IDX]
        else:
            if attention_mask is not None:
                img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask)
            else:
                img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
            img_feature = img_processor_output.hidden_states[LAYER_IDX]

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature
            if self.image_token_compression is not None:
                width = int(math.sqrt(patch_feature.size(1)))
                patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                patch_feature = patch_feature.permute(0, 3, 1, 2)
                if getattr(self, 'img_processor_padding', None) is not None:
                    patch_feature = self.img_processor_padding(patch_feature)
                patch_feature = self.image_token_compression(patch_feature)
                patch_feature = patch_feature.permute(0, 2, 3, 1)
                patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
            elif getattr(self, 'img_processor_padding', None) is not None:
                width = int(math.sqrt(patch_feature.size(1)))
                patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                patch_feature = patch_feature.permute(0, 3, 1, 2)
                patch_feature = self.img_processor_padding(patch_feature)
                patch_feature = patch_feature.permute(0, 2, 3, 1)
                patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            if self.image_token_compression is not None:
                patch_feature = img_feature[:, 1:]
                cls_feature = img_feature[:, 0]
                width = math.sqrt(patch_feature.size(1))
                patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                patch_feature = self.image_token_compression(patch_feature)
                patch_feature = patch_feature.view(-1, patch_feature.size(-2) * patch_feature.size(-1))
                img_feature = torch.cat([cls_feature, patch_feature], dim=1)
            return img_feature

        logger.info(f'processed img feature size = {img_feature.size()}')
        raise NotImplementedError

    def spatiotemporal_pool(self, x, num_img_tokens, batch_size=1, T=1):

        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h * w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        new_x = []
        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        new_x.append(spatial_avg_pool_x)

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        new_x.append(temporal_avg_pool_x)

        x = torch.cat(new_x, dim=1).view(-1, self.image_dim_out)
        num_img_tokens += T
        return x, num_img_tokens

    def forward(self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, image_sizes=None, **kwargs) -> torch.FloatTensor:

        if isinstance(input_ids, tuple):
            input_ids, input_embeds = input_ids

        img_embeds = input_embeds
        if image_sizes is None and 'image_sizes' in kwargs:
            image_sizes = kwargs['image_sizes']
        img_sizes = image_sizes

        if self.img_features is not None:
            img_embeds = self.img_features.clone()
            self.img_features = None

        if self.img_sizes is not None:
            img_sizes = self.img_sizes

        dtype = self.img_processor.embeddings.patch_embedding.weight.dtype
        if img_embeds is not None:
            img_embeds = img_embeds.to(dtype)

        if self.image_attention_mask is not None:
            image_attention_mask = self.image_attention_mask.clone()
            self.image_attention_mask = None
        elif 'image_attention_mask' in kwargs:
            image_attention_mask = kwargs['image_attention_mask']
        else:
            image_attention_mask = None
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True)

        fake_image_forward = False
        select = False
        hd_transform = False

        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        num_img_tokens = self.num_img_tokens
        if len(positions.tolist()) > 0:
            if self.use_hd_transform and img_sizes is not None and len(img_sizes):
                hd_transform = True
                assert img_embeds.ndim == 5, f'(branch 1) img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform'

                bs = img_embeds.shape[0]
                if image_attention_mask is not None and len(image_attention_mask) > 0:
                    img_features = self.get_img_features(img_embeds.flatten(0, 1), attention_mask=image_attention_mask.type(torch.BoolTensor).flatten(0,1).to(target_device))
                else:
                    img_features = self.get_img_features(img_embeds.flatten(0, 1))

                base_feat_height_target = self.base_feat_height_target
                base_resolution = self.crop_size
                base_feat_height_reduction = self.base_feat_height_reduction

                base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))

                assert base_feat_height == base_feat_height_target and base_feat_width == base_feat_height_target, f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect {base_feat_height_target} features for hd transform'

                img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
                C = self.image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // base_resolution
                    w = w // base_resolution
                    B_ = h * w

                    global_img_feature = img_features[_bs, :1]

                    glb_img = global_img_feature.reshape(1,H,H,C).reshape(1,H//base_feat_height_reduction,base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction,C).contiguous().permute(0,1,3,2,4,5).reshape(1,H//base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction*base_feat_height_reduction*C).contiguous()
                    temp_glb_GN = self.sub_GN.repeat(1, H//base_feat_height_reduction, 1, 1)

                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1,-1,base_feat_height_reduction*base_feat_height_reduction*C)

                    sub_img = img_features[_bs, 1:]
                    sub_img = sub_img[:B_]

                    sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//base_feat_height_reduction,base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction,C).contiguous().permute(0,1,3,2,4,5).reshape(B_,-1,base_feat_height_reduction*base_feat_height_reduction*C).contiguous()
                    sub_img = sub_img.reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction, -1).permute(0,1,3,2,4,5).reshape(1,h*base_feat_height//base_feat_height_reduction,w*base_feat_width//base_feat_height_reduction,base_feat_height_reduction*base_feat_height_reduction*C)

                    if image_attention_mask is not None and len(image_attention_mask) > 0:
                        reshaped_image_attention_mask = image_attention_mask[_bs,1:B_+1,0::2,0::2].reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction).permute(0,1,3,2,4).reshape(1,h*base_feat_height//base_feat_height_reduction,w*base_feat_width//base_feat_height_reduction)
                        useful_height = int(reshaped_image_attention_mask[0,:,0].sum().item())
                        useful_width = int(reshaped_image_attention_mask[0,0,:].sum().item())
                        sub_img = sub_img[:,:useful_height, :useful_width]
                        temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                        temp_len = int(image_attention_mask[_bs,:B_+1,0::2,0::2].sum().item()) + (useful_height+1) + base_feat_height//base_feat_height_reduction
                    else:
                        temp_sub_GN = self.sub_GN.repeat(1, h*base_feat_height//base_feat_height_reduction, 1, 1)
                        temp_len = int((h*w+1)*self.num_img_tokens+ 1 + (h+1)*base_feat_height//base_feat_height_reduction)

                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1,-1,base_feat_height_reduction*base_feat_height_reduction*C)

                    if self.hd_transform_order == 'glb_sub':
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.hd_transform_order == 'sub_glb':
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        raise NotImplementedError(f'hd_transform_order = {self.hd_transform_order}, not implemented')

                    assert temp_len == output_imgs[-1].shape[1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
                    output_len.append(temp_len)

                num_img_tokens = output_len
                img_set_tensor = []
                for _output_img in output_imgs:
                    img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                    img_set_tensor.append(img_feature_proj)

            else:
                raise NotImplementedError
            select = True
        else:
            if self.training:
                img_embeds = torch.zeros(1, 3, self.crop_size, self.crop_size, dtype=target_dtype, device=input_ids.device)

                tt = (
                    self.get_img_features(img_embeds)
                    .to(target_device)
                    .to(target_dtype)
                    .reshape(-1, 1024)
                )
                if self.use_hd_transform:
                    img_set_tensor = self.img_projection(tt.reshape(-1, self.image_dim_out*self.base_feat_height_reduction**2) * self.glb_GN[0] * self.sub_GN[0, 0])
                else:
                    img_set_tensor = self.img_projection(tt)
                fake_image_forward = True

        hidden_states = kwargs['wte'](input_ids)

        if select:
            if hd_transform:
                assert all([_img_set_tensor.shape[0] == 1 for _img_set_tensor in img_set_tensor]), 'img_set_tensor should have shape (1, N_tokens, C)'
                merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
                merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    new_hidden_states = hidden_states.index_put(
                        indices=positions_tuple,
                        values=merged_img_set_tensor,
                        accumulate=False
                    )
                hidden_states = new_hidden_states
            else:
                raise NotImplementedError

        if fake_image_forward and self.training:
            hidden_states = hidden_states + (0 * img_set_tensor[0].to(hidden_states.dtype).to(hidden_states.device)).sum()

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states


class Phi4MMAudioEmbedding(nn.Module):

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size

        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        audio_dim_out = None
        logger.info(f"create audio processor {config.audio_processor}")
        self.layer_idx = -2

        if isinstance(config.audio_processor, dict) and config.audio_processor.get('name', None) == "cascades":
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None
            self.encoder = ConformerEncoder(**encoder_config)

            self.encoder.post_init({})

            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError

        assert audio_dim_out is not None, "Remember to set values for audio_dim_out"
        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        self.freeze_audio_processor = kwargs.get('freeze_audio_processor', False)
        logger.info(f'freeze_audio_processor = {self.freeze_audio_processor}')

        self.downsample_rate = kwargs.get('downsample_rate', 1)

        enable_gradient_checkpointing = kwargs.get('enable_gradient_checkpointing', False)
        if enable_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info(f'gradient checkpointing enabled for audio processor')

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == 'mlp':
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = self.downsample_rate

            layers_for_speech = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            for _ in range(1, depth):
                layers_for_speech.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            audio_projection_for_speech = nn.Sequential(*layers_for_speech)

            layers_for_vision = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            for _ in range(1, depth):
                layers_for_vision.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            audio_projection_for_vision = nn.Sequential(*layers_for_vision)

            self.audio_projection = nn.ModuleDict({
                'speech': audio_projection_for_speech,
                'vision': audio_projection_for_vision
            })
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        if audio_config.get('name', None) == "cascades":
            init_model_config = audio_config.get("init_model", {})
            self.encoder.post_init(init_model_config)
            if "init_model" in audio_config:
                audio_config.pop("init_model")

    def set_audio_embeds(self, input_embeds: torch.FloatTensor) -> None:
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(self, input_embeds: torch.FloatTensor, audio_attention_mask: torch.Tensor, audio_projection_mode: str='speech'):

        if self.freeze_audio_processor:
            with torch.no_grad():
                audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        if isinstance(self.audio_projection, nn.Sequential):
            audio_set_tensor = self.audio_projection(audio_features)
        elif isinstance(self.audio_projection, nn.ModuleDict):
            audio_set_tensor = self.audio_projection[audio_projection_mode](audio_features)
        else:
            raise NotImplementedError

        return audio_set_tensor

    def forward(self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, audio_embed_sizes=None, audio_attention_mask=None, audio_projection_mode='speech', **kwargs) -> torch.FloatTensor:
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.clone()
        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.clone()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        MAX_INPUT_ID = int(1e9)

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)

        if isinstance(self.audio_projection, nn.Sequential):
            target_device = self.audio_projection[0].bias.device
            target_dtype = self.audio_projection[0].bias.dtype
        elif isinstance(self.audio_projection, nn.ModuleDict):
            target_device = self.audio_projection[audio_projection_mode][0].bias.device
            target_dtype = self.audio_projection[audio_projection_mode][0].bias.dtype
        else:
            target_device = self.audio_projection.bias.device
            target_dtype = self.audio_projection.bias.dtype

        if input_embeds is not None:
            input_embeds = input_embeds.to(target_device).to(target_dtype)

        if len(positions.tolist()) > 0:
            audio_set_tensor = self.get_audio_features(input_embeds, audio_attention_mask, audio_projection_mode)
        else:
            if self.training:
                audio_embeds = torch.zeros(1, 500, self.audio_dim_in).to(target_device).to(target_dtype)
                audio_attention_mask = audio_embeds.new_ones(audio_embeds.size()[:2]).long()
                audio_set_tensor = self.get_audio_features(audio_embeds, audio_attention_mask, audio_projection_mode)

        hidden_states = kwargs['wte'](input_ids)

        if len(positions.tolist()) > 0:

            assert audio_embed_sizes.sum().item() == len(positions), \
                f"please ensure the encoder outputs have the same length as defined in input_ids! \n audio_embed_sizes.sum().item(): {audio_embed_sizes.sum().item()} \n len(positions): {len(positions)} \n audio_embed_sizes: {audio_embed_sizes} \n positions: {positions} \n input_ids.shape \n {input_ids.shape}"

            merged_audio_set_tensor = torch.cat([
                audio_set_tensor[i, :audio_embed_sizes[i], :]
                for i in range(len(audio_embed_sizes))
            ], dim=0)
            merged_audio_set_tensor = merged_audio_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                new_hidden_states = hidden_states.index_put(
                    indices=positions_tuple,
                    values=merged_audio_set_tensor,
                    accumulate=False
                )
            hidden_states = new_hidden_states
        else:
            if self.training:
                hidden_states  = hidden_states + (0 * audio_set_tensor[:,0].to(hidden_states.dtype).to(hidden_states.device)).sum()

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states



class Phi4MMImageAudioEmbedding(nn.Module):

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size

        self.image_input_id = kwargs.get('image_input_id', -1)
        self.audio_input_id = kwargs.get('audio_input_id', -10000)
        assert self.image_input_id != self.audio_input_id, 'image_input_id and audio_input_id should be different'

        self.image_embd_layer_kwargs = kwargs['image_embd_layer']
        self.image_embed = Phi4MMImageEmbedding(config, **self.image_embd_layer_kwargs)
        self.audio_embd_layer_kwargs = kwargs['audio_embd_layer']
        self.audio_embed = Phi4MMAudioEmbedding(config, **self.audio_embd_layer_kwargs)

        self.input_image_embeds = None
        self.image_sizes = None
        self.image_attention_mask = None
        self.input_audio_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        self.audio_embed.post_init(audio_config)

    def set_input_image_embeds(self, input_image_embeds: torch.FloatTensor) -> None:
        self.input_image_embeds = input_image_embeds

    def set_image_sizes(self, image_sizes: torch.LongTensor) -> None:
        self.image_sizes = image_sizes

    def set_img_attn_mask(self, image_attention_mask: torch.FloatTensor) -> None:
        self.image_attention_mask = image_attention_mask

    def set_input_audio_embeds(self, input_audio_embeds: torch.FloatTensor) -> None:
        self.input_audio_embeds = input_audio_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds,
        input_image_embeds: Optional[torch.FloatTensor]=None,
        input_audio_embeds: Optional[torch.FloatTensor]=None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode='speech',
        wte=None,
    ) -> torch.FloatTensor:
        MAX_INPUT_ID = int(1e9)
        assert -MAX_INPUT_ID < self.audio_input_id < self.image_input_id

        if self.input_image_embeds is not None:
            assert input_image_embeds is None
            input_image_embeds = self.input_image_embeds.clone()
            self.input_image_embeds = None

        if self.image_sizes is not None:
            assert image_sizes is None
            image_sizes = self.image_sizes

        if self.input_audio_embeds is not None:
            assert input_audio_embeds is None
            input_audio_embeds = self.input_audio_embeds.clone()
            self.input_audio_embeds = None

        if self.audio_embed_sizes is not None:
            assert audio_embed_sizes is None
            audio_embed_sizes = self.audio_embed_sizes.clone()

        if self.image_attention_mask is not None:
            assert image_attention_mask is None
            image_attention_mask = self.image_attention_mask.clone()
            self.image_attention_mask = None

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            new_input_ids = input_ids.clone()
            new_input_ids[(input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0]) &
                        (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1])] = _IMAGE_SPECIAL_TOKEN_ID
            new_input_ids[(input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0]) &
                        (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1])] = _AUDIO_SPECIAL_TOKEN_ID
            input_ids = new_input_ids

        with torch.no_grad():
            image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
            non_image_position_mask = ~image_position_mask

        assert input_embeds is None
        if self.training:
            assert input_image_embeds is not None or input_audio_embeds is not None

        if input_image_embeds is not None:
            image_hidden_states = self.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                wte=wte,
                image_attention_mask=image_attention_mask
            )
        if input_audio_embeds is not None:
            audio_hidden_states = self.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=wte,
                audio_projection_mode=audio_projection_mode,
            )

        if input_image_embeds is not None and input_audio_embeds is not None:
            dtype = image_hidden_states.dtype
            hidden_states = image_hidden_states * image_position_mask.to(dtype).unsqueeze(-1) + audio_hidden_states * non_image_position_mask.to(dtype).unsqueeze(-1)
        elif input_image_embeds is not None:
            hidden_states = image_hidden_states
        elif input_audio_embeds is not None:
            hidden_states = audio_hidden_states
        else:
            assert wte is not None
            hidden_states = wte(input_ids)

        return hidden_states


class Phi4MMRMSNorm(nn.Module):
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


class Phi4MMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi4MMSuScaledRotaryEmbedding(Phi4MMRotaryEmbedding):
    def __init__(self, dim, config, device=None):
        warnings.warn(
            "The class Phi4MMSuScaledRotaryEmbedding is deprecated and will be removed in version 5 of Transformers. Please"
            " use Phi4MMLongRoPEScaledRotaryEmbedding instead.",
            FutureWarning,
        )
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)
        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi4MMYarnScaledRotaryEmbedding(Phi4MMRotaryEmbedding):
    def __init__(self, dim, config, device=None):
        warnings.warn(
            "The class Phi4MMYarnScaledRotaryEmbedding is deprecated and will be removed in version 5 of Transformers",
            FutureWarning,
        )
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = 0.1 * math.log(scale) + 1.0

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi4MMLongRoPEScaledRotaryEmbedding(Phi4MMRotaryEmbedding):
    def __init__(self, dim, config, device=None):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = seq_len or torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class Phi4MMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Phi4MMAttention(nn.Module):

    def __init__(self, config: Phi4MMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.rotary_ndims = int(self.head_dim * config.partial_rotary_factor)
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = Phi4MMRotaryEmbedding(
                self.rotary_ndims,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = Phi4MMLongRoPEScaledRotaryEmbedding(self.rotary_ndims, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        logger.warning_once("You are not running the flash-attention implementation, expect numerical differences.")

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Phi4MMFlashAttention2(Phi4MMAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        rotary_seq_len = (
            max(kv_seq_len, position_ids[:, -1].max().item() + 1) if position_ids is not None else kv_seq_len
        )

        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len, position_ids=position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0


        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

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

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=attn_dropout,
            sliding_window=getattr(self.config, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Phi4MMSdpaAttention(Phi4MMAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "Phi4MMModel is using Phi4MMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


PHI4MM_ATTENTION_CLASSES = {
    "eager": Phi4MMAttention,
    "flash_attention_2": Phi4MMFlashAttention2,
    "sdpa": Phi4MMSdpaAttention,
}


class Phi4MMDecoderLayer(nn.Module):
    def __init__(self, config: Phi4MMConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.self_attn = PHI4MM_ATTENTION_CLASSES["sdpa"](config, layer_idx=layer_idx)

        self.mlp = Phi4MMMLP(config)
        self.input_layernorm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


PHI4MM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`Phi4MMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Phi-4-MM model outputting raw hidden-states without any specific head on top.",
    PHI4MM_START_DOCSTRING,
)
class Phi4MMPreTrainedModel(PreTrainedModel):
    config_class = Phi4MMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi4MMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    _version = "0.0.5"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


PHI4MM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Phi-4-MM model outputting raw hidden-states without any specific head on top.",
    PHI4MM_START_DOCSTRING,
)


@add_start_docstrings(
    "The bare Phi-4-MM model outputting raw hidden-states without any specific head on top.",
    PHI4MM_START_DOCSTRING,
)
class Phi4MMModel(Phi4MMPreTrainedModel):

    def __init__(self, config: Phi4MMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)

        self.embed_tokens_extend = None
        if isinstance(config.embd_layer, dict):
            embedding_config = {
                'embedding_cls': config.embd_layer['embedding_cls'],
                **config.embd_layer
            }
            self.embed_tokens_extend = Phi4MMImageAudioEmbedding(config, **embedding_config)

        self.layers = nn.ModuleList(
            [Phi4MMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(PHI4MM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
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

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens_extend(
                input_ids=input_ids,
                input_embeds=inputs_embeds,
                input_image_embeds=input_image_embeds,
                input_audio_embeds=input_audio_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
                wte=self.embed_tokens,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
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
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

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
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
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
            and attention_mask.device.type == "cuda"
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
        config: Phi4MMConfig,
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
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask
    





class FastV_zichen(Phi4MMModel):


    def __init__(self, config: Phi4MMConfig):
        self.last_attention = None
        self.config = config
        super().__init__(config)

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
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

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens_extend(
                input_ids=input_ids,
                input_embeds=inputs_embeds,
                input_image_embeds=input_image_embeds,
                input_audio_embeds=input_audio_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
                wte=self.embed_tokens,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        batch_size, seq_length = inputs_embeds.shape[:2]

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        K = self.config.image_layer_idx if self.config.image_layer_idx else None
        M = self.config.audio_layer_idx if self.config.audio_layer_idx else None
        output_attentions=True
        random=self.config.random
        for idx,decoder_layer in enumerate(self.layers):
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
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(retain_token_num).indices + image_start

                        keep_indexs = torch.cat((torch.arange(image_start,device=device), top_attention_rank_index, torch.arange(image_end,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values

                        cache_position = cache_position[keep_indexs]

                        hidden_states = hidden_states[:,keep_indexs,:]
                        position_ids = keep_indexs.unsqueeze(0)                    
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==K-1 and seq_length>1:
                        output_attentions=True
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
                elif M and (random==False):
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
                        position_ids = keep_indexs.unsqueeze(0)                    
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions)
                    if M and idx==M-1 and seq_length>1:
                        output_attentions=True
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
                        position_ids = keep_indexs.unsqueeze(0)                    
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==K-1 and seq_length>1:
                        output_attentions=True
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
                elif M and (random==True):
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
                        position_ids = keep_indexs.unsqueeze(0)                    
                        causal_mask = self._update_causal_mask(
                                        None, hidden_states, cache_position, None, output_attentions
                                    )
                    if idx==M-1 and seq_length>1:
                        output_attentions=True
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
                 )


                if K and idx == K - 1 and seq_length > 1 :
                     self.last_attention = layer_outputs[1]
                if M and idx== M - 1 and seq_length > 1:
                     self.last_attention = layer_outputs[1]
            hidden_states = layer_outputs[0]
            output_attentions=False
            if use_cache:
                next_decoder_cache=(layer_outputs[2 if output_attentions else 1])

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi4MMForCausalLM(Phi4MMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = FastV_zichen(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        assert getattr(config, "vision_lora", None) is not None
        from peft import LoraConfig, get_peft_model
        vision_lora_config = LoraConfig(
            r=config.vision_lora['r'],
            lora_alpha=config.vision_lora['lora_alpha'],
            target_modules=config.vision_lora['layer'],
            lora_dropout=config.vision_lora['dp'],
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(self.model, vision_lora_config, adapter_name="vision")
        self.config.vision_lora['r'] = config.vision_lora['r']
        self.config.vision_lora['lora_alpha'] = config.vision_lora['lora_alpha']
        self.config.vision_lora['layer'] = config.vision_lora['layer']
        self.config.vision_lora['dp'] = config.vision_lora['dp']

        assert getattr(config, "speech_lora", None) is not None
        speech_lora_config = LoraConfig(
            r=config.speech_lora['r'],
            lora_alpha=config.speech_lora['lora_alpha'],
            target_modules=config.speech_lora['layer'],
            lora_dropout=config.speech_lora['dp'],
            task_type="CAUSAL_LM",
        )
        peft_model.base_model.active_adapter.append("speech")
        peft_model.add_adapter("speech", speech_lora_config)
        self.config.speech_lora['r'] = config.speech_lora['r']
        self.config.speech_lora['lora_alpha'] = config.speech_lora['lora_alpha']
        self.config.speech_lora['layer'] = config.speech_lora['layer']
        self.config.speech_lora['dp'] = config.speech_lora['dp']

    def set_lora_adapter(self, adapter_name) -> None:
        from peft.tuners.lora.layer import LoraLayer
        for module in self.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
                module._disable_adapters = False

    def unset_lora_adapter(self) -> None:
        from peft.tuners.lora.layer import LoraLayer
        for module in self.modules():
            if isinstance(module, LoraLayer):
                for layer_name in module.adapter_layer_names:
                    layer = getattr(module, layer_name)
                    layer.requires_grad_(False)
                module._disable_adapters = True

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(PHI4MM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (
            use_cache
            and self.config.rope_scaling
            and cache_position is not None
            and cache_position[0] == self.config.original_max_position_embeddings
        ):
            logger.warning(
                f"If you are not using the generate method, you may encounter nonsensical outputs after the {self.config.original_max_position_embeddings}th token, as the KV cache needs to be recomputed."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_mode, torch.Tensor):
            assert len(input_mode) == 1
            input_mode = input_mode[0].item()
        input_mode = InputMode(input_mode)

        if input_mode in [InputMode.VISION_SPEECH, InputMode.VISION]:
            self.set_lora_adapter('vision')
            audio_projection_mode = 'vision'
        elif input_mode == InputMode.SPEECH:
            self.set_lora_adapter('speech')
            audio_projection_mode = 'speech'
        elif input_mode == InputMode.LANGUAGE:
            self.unset_lora_adapter()
            audio_projection_mode = 'speech'
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            audio_projection_mode=audio_projection_mode,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        input_image_embeds=None,
        image_sizes=None,
        image_attention_mask=None,
        input_audio_embeds=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs
    ):

        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            input_mode=input_mode,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        return model_inputs


@add_start_docstrings(
    """
    The [`Phi4MMModel`] with a sequence classification head on top (linear layer).
    [`Phi4MMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    PHI4MM_START_DOCSTRING,
)

class Phi4MMForSequenceClassification(Phi4MMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Phi4MMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(PHI4MM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = model_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


@add_start_docstrings(
    """
    [`Phi4MMModel`] with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    PHI4MM_START_DOCSTRING,
)
class Phi4MMForTokenClassification(Phi4MMPreTrainedModel):
    def __init__(self, config: Phi4MMConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = Phi4MMModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    @add_start_docstrings_to_model_forward(PHI4MM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = model_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


AutoConfig.register("phi4mm", Phi4MMConfig)
AutoModelForCausalLM.register(Phi4MMConfig, Phi4MMForCausalLM)
Phi4MMConfig.register_for_auto_class()
Phi4MMForCausalLM.register_for_auto_class("AutoModelForCausalLM")