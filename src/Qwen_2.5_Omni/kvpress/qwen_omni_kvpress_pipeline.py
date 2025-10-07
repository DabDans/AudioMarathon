#!/usr/bin/env python3
"""
Qwen2.5-Omni Dedicated KV Press Pipeline
Integrated audio-aware KV compression, optimized for Qwen2.5-Omni model architecture
"""

import os
import sys
import time
import logging
import contextlib
import traceback
from typing import Optional, Dict, List, Tuple, Union, Any

import torch
import numpy as np
from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY


try:
    from kvpress.presses.base_press import BasePress
    from kvpress.presses.observed_attention_press import ObservedAttentionPress
    from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
    KV_PRESS_AVAILABLE = True
except ImportError:
    print("[Warning] KV Press library unavailable")
    KV_PRESS_AVAILABLE = False


_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648

logger = logging.getLogger(__name__)


class AudioTokenDetector:
    """Simple audio token detector"""
    
    @staticmethod
    def identify_audio_tokens(input_ids):
        """Identify positions of audio tokens"""
        if input_ids is None:
            return []
        
        audio_ranges = []
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            token_ids = input_ids[batch_idx].tolist()
            
            if _AUDIO_BOS_TOKEN_ID in token_ids and _AUDIO_EOS_TOKEN_ID in token_ids:
                audio_start = token_ids.index(_AUDIO_BOS_TOKEN_ID)
                audio_end = token_ids.index(_AUDIO_EOS_TOKEN_ID)
                audio_ranges.append((batch_idx, audio_start, audio_end))
                
        return audio_ranges
    
    @staticmethod
    def create_audio_mask(input_ids, seq_len, device):
        """Create mask for audio tokens"""
        audio_ranges = AudioTokenDetector.identify_audio_tokens(input_ids)
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        
        audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for batch_idx, start_pos, end_pos in audio_ranges:
            if batch_idx < batch_size and end_pos < seq_len:
                audio_mask[batch_idx, start_pos:end_pos+1] = True
                
        return audio_mask


class AudioAwarePress:
    """
    Audio-aware KV compressor, properly follows official KVPress design pattern
    Inherits original Press functions, adds audio token-aware compression
    """
    
    def __init__(self, base_press, input_ids=None):
        """
        Args:
            base_press: Original KVPress compressor (e.g. KnormPress, SnapKVPress, etc.)
            input_ids: Input token sequence, used to identify audio token positions
        """
        self.base_press = base_press
        self.input_ids = input_ids
        

        if hasattr(base_press, 'compression_ratio'):
            self.compression_ratio = base_press.compression_ratio
        
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Implements audio-aware compression logic
        """

        if not hasattr(self, 'compression_ratio') or self.compression_ratio == 0 or self.input_ids is None:
            return self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
        

        is_flash_attn = hasattr(module, '_attn_implementation') and module._attn_implementation == 'flash_attention_2'
        

        if is_flash_attn:
            if not keys.is_contiguous():
                keys = keys.contiguous()
            if not values.is_contiguous():
                values = values.contiguous()
        

        audio_detected = False
        if self.input_ids is not None and self.input_ids.numel() > 0:
            audio_detected = (_AUDIO_BOS_TOKEN_ID in self.input_ids[0] and _AUDIO_EOS_TOKEN_ID in self.input_ids[0])
        
        if audio_detected:

            seq_len = keys.shape[2]
            device = keys.device
            

            audio_mask = AudioTokenDetector.create_audio_mask(self.input_ids, seq_len, device)
            
            if audio_mask.any():

                audio_positions = torch.where(audio_mask[0])[0]
                
                if len(audio_positions) > 0:

                    if hasattr(self.base_press, 'score'):

                        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
                    else:

                        scores = keys.norm(dim=-1)  # [batch, heads, seq_len]
                    

                    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
                    

                    if len(audio_positions) > 0:
                        audio_scores = scores[0, 0, audio_positions]  # [audio_len]

                        audio_compression_ratio = min(0.8, self.compression_ratio * 2)
                        n_audio_kept = max(1, int(len(audio_positions) * (1 - audio_compression_ratio)))
                        

                        if n_audio_kept < len(audio_positions):
                            _, top_indices = torch.topk(audio_scores, n_audio_kept)
                            kept_audio_positions = audio_positions[top_indices]
                            

                            keep_mask[audio_positions] = False
                            keep_mask[kept_audio_positions] = True
                    

                    kept_indices = torch.where(keep_mask)[0]
                    compressed_keys = keys[:, :, kept_indices]
                    compressed_values = values[:, :, kept_indices]
                    

                    if is_flash_attn:
                        compressed_keys = compressed_keys.contiguous()
                        compressed_values = compressed_values.contiguous()
                    
                    if hasattr(module, 'layer_idx') and module.layer_idx < 3:
                        logger.info(f"[AudioPress] Layer {module.layer_idx}: audio tokens {len(audio_positions)} -> {n_audio_kept if 'n_audio_kept' in locals() else len(audio_positions)}, total {seq_len} -> {len(kept_indices)}")
                    
                    return compressed_keys, compressed_values
        

        return self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
    
    def forward_hook(self, module, input, kwargs, output):
        """
        Use original press's forward_hook, but inject audio-aware logic
        """

        if self.input_ids is not None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is not None:
                seq_len = hidden_states.shape[1]
                device = hidden_states.device
                audio_mask = AudioTokenDetector.create_audio_mask(self.input_ids, seq_len, device)
                kwargs['audio_compression_mask'] = audio_mask
        

        return self.base_press.forward_hook(module, input, kwargs, output)
    
    def __call__(self, model):
        """
        Directly use base_press's context manager, but replace compress method
        """

        original_compress = self.base_press.compress
        self.base_press.compress = self.compress
        
        try:

            return self.base_press(model)
        finally:

            self.base_press.compress = original_compress


class QwenOmniKVPressAudioPipeline(Pipeline):
    """
    Audio KV Press Pipeline designed for Qwen2.5-Omni
    
    Integrates:
    1. Qwen2.5-Omni model architecture adaptation
    2. Audio-aware KV compression
    3. Intelligent compression ratio control
    4. Audio token identification
    """
    
    def __init__(self, model, processor, **kwargs):
        """
        Initialize Qwen2.5-Omni KV Press Pipeline
        
        Args:
            model: Qwen2_5OmniForConditionalGeneration model
            processor: Qwen2_5OmniProcessor processor
        """

        self._qwen_processor = processor
        

        super().__init__(model=model, tokenizer=processor.tokenizer, **kwargs)
        

        self._inject_audio_config()
        
    def _inject_audio_config(self):
        """Inject audio-related config into model"""
        try:

            if hasattr(self.model, 'thinker') and hasattr(self.model.thinker, 'config'):
                cfg = self.model.thinker.config
                if not hasattr(cfg, 'audio_token_id'): cfg.audio_token_id = _AUDIO_TOKEN_ID
                if not hasattr(cfg, 'audio_bos_token_id'): cfg.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
                if not hasattr(cfg, 'audio_eos_token_id'): cfg.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
                if not hasattr(cfg, 'image_token_id'): cfg.image_token_id = 151655
                if not hasattr(cfg, 'video_token_id'): cfg.video_token_id = 151656
                if not hasattr(cfg, 'audio_layer_idx'): cfg.audio_layer_idx = None
                if not hasattr(cfg, 'audio_prune_ratio'): cfg.audio_prune_ratio = 0
                

            if (hasattr(self.model, 'thinker') and 
                hasattr(self.model.thinker, 'model') and 
                hasattr(self.model.thinker.model, 'config')):
                cfg2 = self.model.thinker.model.config
                if not hasattr(cfg2, 'audio_token_id'): cfg2.audio_token_id = _AUDIO_TOKEN_ID
                if not hasattr(cfg2, 'audio_bos_token_id'): cfg2.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
                if not hasattr(cfg2, 'audio_eos_token_id'): cfg2.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
                if not hasattr(cfg2, 'image_token_id'): cfg2.image_token_id = 151655
                if not hasattr(cfg2, 'video_token_id'): cfg2.video_token_id = 151656
                if not hasattr(cfg2, 'audio_layer_idx'): cfg2.audio_layer_idx = None
                if not hasattr(cfg2, 'audio_prune_ratio'): cfg2.audio_prune_ratio = 0
                
            logger.info("[Qwen-Omni] Audio config injected successfully")
            
        except Exception as e:
            logger.warning(f"[Qwen-Omni] Config injection failed: {e}")
    
    def _sanitize_parameters(
        self,
        prompt: Optional[str] = None,
        audio_path: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        compression_ratio: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Parameter preprocessing"""
        

        if compression_ratio is not None and compression_ratio >= 1.0:
            logger.warning(f"Compression ratio {compression_ratio} exceeds 1.0, setting to 0.99")
            compression_ratio = 0.99
        
        preprocess_kwargs = {
            "prompt": prompt or "",
            "audio_path": audio_path,
            "messages": messages,
        }
        
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in preprocess_kwargs}
        gen_kwargs.update({
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        })
        
        forward_kwargs = {
            "press": press,
            "generation_kwargs": gen_kwargs,
            "compression_ratio": compression_ratio,
        }
        
        return preprocess_kwargs, forward_kwargs, {}
    
    def preprocess(self, prompt: str, audio_path: Optional[str] = None, messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Preprocess using Qwen2.5-Omni Processor
        
        Args:
            prompt: Text prompt (only used when messages is not provided)
            audio_path: Audio file path
            messages: Pre-built message list (optional, prioritized over prompt)
        """
        try:

            sys.path.append("/data/to/your/Audio_Longbench/path/")
            from qwen_omni_utils import process_mm_info
            

            if messages is not None:

                final_messages = messages

                if audio_path and os.path.exists(audio_path):

                    for msg in reversed(final_messages):
                        if msg.get("role") == "user":
                            content = msg.get("content", [])
                            if isinstance(content, str):

                                content = [{"type": "text", "text": content}]

                            has_audio = any(item.get("type") == "audio" for item in content if isinstance(item, dict))
                            if not has_audio:
                                content.insert(0, {"type": "audio", "audio": audio_path})
                                msg["content"] = content
                            break
            else:

                content = [{"type": "text", "text": prompt}]
                if audio_path and os.path.exists(audio_path):
                    content.insert(0, {"type": "audio", "audio": audio_path})
                
                final_messages = [
                    {"role": "user", "content": content},
                ]
            

            audios, images, videos = process_mm_info(final_messages, use_audio_in_video=True)
            

            text = self._qwen_processor.apply_chat_template(
                final_messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(text, list):
                text = text[0]
            

            inputs = self._qwen_processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            )
            

            inputs = inputs.to(self.model.device)
            inputs = {
                k: (v.to(self.model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
                for k, v in inputs.items()
            }
            
            return {
                "model_inputs": inputs,
                "input_text": text,
                "has_audio": audio_path is not None and os.path.exists(audio_path),
                "processed_messages": final_messages
            }
            
        except Exception as e:
            logger.error(f"[Qwen-Omni] Preprocessing failed: {e}")
            traceback.print_exc()
            raise
    
    def _forward(
        self,
        model_inputs: Dict[str, Any],
        press: Optional[BasePress] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        compression_ratio: Optional[float] = None,
        measure_time: bool = True,
    ):
        """KV Press forward pass"""
        if generation_kwargs is None:
            generation_kwargs = {}
        
        inputs = model_inputs["model_inputs"]
        input_text = model_inputs["input_text"]
        has_audio = model_inputs["has_audio"]
        

        metrics = {
            "prefill_time": 0.0,
            "generation_time": 0.0,
            "total_time": 0.0,
            "has_audio": has_audio,
            "compression_enabled": press is not None
        }
        
        try:

            start_time = time.time()
            

            if press is not None:

                try:
                    from kvpress.presses.observed_attention_press import ObservedAttentionPress as _ObservedPress
                except Exception:
                    _ObservedPress = tuple()

                attn_impl = getattr(getattr(self.model, 'config', None), 'attn_implementation', None)
                if isinstance(press, _ObservedPress):
                    if attn_impl != 'eager':
                        logger.warning(
                            "ObservedAttentionPress requires attn_implementation='eager' and explicitly enabling output_attentions during prefill. "
                            "Current value is '%s' and using generate call, this will not collect attention weights. "
                            "Please use official KVPressTextGenerationPipeline style prefill + greedy decoding, or switch to token-level compression methods (e.g. Knorm/Snap/TOVA/LagKV/Expected).",
                            str(attn_impl)
                        )
                

                input_ids = inputs.get('input_ids', None)
                audio_detected = False
                
                if input_ids is not None:
                    audio_detected = (_AUDIO_BOS_TOKEN_ID in input_ids[0] and _AUDIO_EOS_TOKEN_ID in input_ids[0]) if input_ids.numel() > 0 else False
                
                if audio_detected:
                    logger.info("[Qwen-Omni] Audio token detected, enabling audio-aware KV compression")

                    audio_press = AudioAwarePress(press, input_ids)
                    with torch.no_grad(), audio_press(self.model):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                else:
                    logger.info("[Qwen-Omni] No audio token detected, using standard KV compression")
                    with torch.no_grad(), press(self.model):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
            else:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            
            end_time = time.time()
            metrics["generation_time"] = end_time - start_time
            metrics["total_time"] = metrics["generation_time"]
            

            generated_text = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            

            if "assistant\n" in generated_text:
                generated_text = generated_text.split("assistant\n")[-1].strip()
            

            output_tokens = outputs.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            input_tokens = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            
            return {
                "generated_text": generated_text,
                "input_text": input_text,
                "metrics": metrics,
                "output_tokens": output_tokens,
                "input_tokens": input_tokens,
                "compression_info": {
                    "enabled": press is not None,
                    "press_type": type(press).__name__ if press else None,
                    "compression_ratio": compression_ratio,
                    "audio_detected": has_audio,
                }
            }
            
        except Exception as e:
            logger.error(f"[Qwen-Omni] Generation failed: {e}")
            traceback.print_exc()
            raise
    
    def postprocess(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess generated text"""
        text = model_outputs["generated_text"]
        

        for prefix in ["spoken content:", "content:", "transcription:", "text:"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        

        text = text.replace("<transcribed text here>", "").strip()
        
        result = {
            "generated_text": text,
            "input_text": model_outputs["input_text"],
            "metrics": model_outputs["metrics"],
            "output_tokens": model_outputs["output_tokens"],
            "input_tokens": model_outputs["input_tokens"],
            "compression_info": model_outputs["compression_info"]
        }
        
        return result
    
    def __call__(
        self,
        prompt: Optional[str] = None,
        audio_path: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 256,
        compression_ratio: Optional[float] = None,
        measure_time: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Pipeline call interface
        
        Args:
            prompt: Text prompt (only used when messages is not provided)
            audio_path: Audio file path
            messages: Pre-built message list (optional, prioritized over prompt)
            press: KV compression object
            max_new_tokens: Maximum number of generated tokens
            compression_ratio: Compression ratio
            measure_time: Whether to measure time
            **kwargs: Other parameters
        """
        

        pre_params, forward_params, post_params = self._sanitize_parameters(
            prompt=prompt,
            audio_path=audio_path,
            messages=messages,
            press=press,
            max_new_tokens=max_new_tokens,
            compression_ratio=compression_ratio,
            **kwargs
        )
        

        model_inputs = self.preprocess(**pre_params)
        forward_params["measure_time"] = measure_time
        outputs = self._forward(model_inputs, **forward_params)
        return self.postprocess(outputs)



if KV_PRESS_AVAILABLE:
    PIPELINE_REGISTRY.register_pipeline(
        "qwen-omni-kv-press-audio",
        pipeline_class=QwenOmniKVPressAudioPipeline,
        pt_model=None,
    )
    
    logger.info("[Qwen-Omni] KV Press Audio Pipeline registered successfully")
