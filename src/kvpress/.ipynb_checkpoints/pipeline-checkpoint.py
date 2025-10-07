# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
import time
import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress

logger = logging.getLogger(__name__)

class KVPressTextGenerationPipeline(Pipeline):
    """
    Pipeline for key-value compression in causal language models.
    This pipeline allows you to compress a long prompt using a key-value press
    and then generate answers using greedy decoding.
    """

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        The user can either provide a single question or a list of questions to be asked about the context.

        Parameters
        ----------
        question : str, optional
            The question to be asked about the context. Exclusive with `questions`.
        questions : list[str], optional
            A list of questions to be asked about the context. Exclusive with `question`.
        answer_prefix : str, optional
            The prefix to be added to the generated answer.
        press : BasePress, optional
            The key-value press to use for compression.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer.
        max_context_length : int, optional
            The maximum number of tokens in the context. By default will use the maximum length supported by the model.
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).
        **kwargs : dict
            Additional keyword arguments, currently ignored.

        Returns
        -------
        Tuple[dict, dict, dict]
            A tuple containing three dictionaries:
                - preprocess_kwargs: The keyword arguments for the preprocess function.
                - forward_kwargs: The keyword arguments for the forward function.
                - postprocess_kwargs: The keyword arguments for the postprocess function.
        """

        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
        }
        forward_kwargs = {"press": press, "max_new_tokens": max_new_tokens, "cache": cache}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
    ):
        """
        Apply the chat template to the triplet (context, questions, answer_prefix) and tokenize it.

        Returns
        -------
        dict[str, GenericTensor]
            A dictionary containing the tokenized context (key: "context_ids") and questions (key: "questions_ids").

        """

        # Apply chat template if available
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"  # to separate the question from the answer
        else:
            separator = "\n" + "#" * len(context)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}], add_generation_prompt=True, tokenize=False
            )
            context, question_suffix = context.split(separator)

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix + answer_prefix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        # Truncate context
        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
    ):
        """
        Forward pass of the kv-press pipeline.

        Parameters
        ----------
        input_tensors : dict[str, GenericTensor]
            A dictionary containing the tokenized context and questions.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer. Defaults to 50.
        press : BasePress, optional
            The key-value press to use for compression. Defaults to None.
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).

        Returns
        -------
        list[str]
            A list of generated answers.
        """

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        # Prefilling using the press on the context
        if cache is None:
            cache = DynamicCache()

        with press(self.model) if press is not None else contextlib.nullcontext():
            self.model(
                input_ids=context_ids,
                past_key_values=cache,
                output_attentions=self.output_attentions(press),
                num_logits_to_keep=1,
            )

        logger.debug(f"Context Length: {context_length}")
        logger.debug(f"Compressed Context Length: {cache.get_seq_length()}")

        # Greedy decoding for each question
        answers = []
        for question_ids in input_tensors["questions_ids"]:
            answer = self.generate_answer(
                question_ids=question_ids.to(self.model.device),
                cache=cache,
                context_length=(cache.get_seq_length() if isinstance(press, KeyRerotationPress) else context_length),
                max_new_tokens=max_new_tokens,
            )
            answers.append(answer)

        return answers

    def output_attentions(self, press: BasePress):
        if isinstance(press, ObservedAttentionPress):
            return True
        if isinstance(press, (KeyRerotationPress, PerLayerCompressionPress)) and isinstance(
            press.press, ObservedAttentionPress
        ):
            return True
        return False

    def postprocess(self, model_outputs, single_question):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}

    def generate_answer(
        self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int
    ) -> str:
        """
        Generate an answer to a question using greedy decoding.

        Parameters
        ----------
        question_ids : torch.Tensor
            The tokenized question.
        cache : Cache
            The compressed key-value cache.
        context_length : int
            The length of the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Returns
        -------
        str
            The generated answer.
        """

        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        # if the user doesn't provide a question, skip forward pass
        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break
        answer = self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

        # Remove the generated tokens from the cache
        cache.key_cache = [
            cache.key_cache[layer_idx][:, :, :sequence_length]
            for layer_idx, sequence_length in enumerate(cache_seq_lengths)
        ]
        cache.value_cache = [
            cache.value_cache[layer_idx][:, :, :sequence_length]
            for layer_idx, sequence_length in enumerate(cache_seq_lengths)
        ]
        if hasattr(cache, "_quantized_key_cache"):
            cache._quantized_key_cache = [
                cache._quantized_key_cache[layer_idx][:, :, :sequence_length]
                for layer_idx, sequence_length in enumerate(cache_seq_lengths)
            ]
            cache._quantized_value_cache = [
                cache._quantized_value_cache[layer_idx][:, :, :sequence_length]
                for layer_idx, sequence_length in enumerate(cache_seq_lengths)
            ]

        return answer

class KVPressAudioTranscriptionPipeline(Pipeline):
    """
    专为音频转录设计的 KV Press 管道。
    该管道支持传入 prompt（ASR提示词）以及 audios（格式为 List[Tuple(audio_array, sampling_rate)]）,
    并通过 input_mode 指定音频模式（默认2）。
    
    提供三个关键方法：
      - _sanitize_parameters：构造预处理、前向和后处理的参数字典
      - preprocess：调用 processor 对文本和音频进行预处理，并将 tensor 转移到模型设备
      - _forward：在 KV Press 上下文中调用 model.generate 生成转录文本
      - postprocess：后处理生成的文本，清理常见前缀和特殊标记
      
    同时重写 __call__ 方法以支持直接传入关键字参数调用。
    """

    def __init__(self, *args, **kwargs):
        # 使用不同名称保存处理器，避免与Pipeline基类冲突
        self._audio_processor = kwargs.pop("processor", None)
        self._AUDIO_SPECIAL_TOKEN_ID = kwargs.pop("audio_special_token_id", None)
        if self._audio_processor is None:
            raise ValueError("必须提供 processor 参数，例如：processor=AutoProcessor.from_pretrained(model_id)")
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(
        self,
        prompt: Optional[str] = None,
        audios: Optional[List[Tuple]] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 1100,
        do_sample: bool = False,
        input_mode: int = 2,  # 默认音频模式
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # 与官方 KVPressTextGenerationPipeline 类似，构造三个阶段使用的参数字典
        preprocess_kwargs = {"prompt": prompt, "audios": audios, "input_mode": input_mode}
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in preprocess_kwargs}
        gen_kwargs.update({
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        })
        forward_kwargs = {"press": press, "generation_kwargs": gen_kwargs, "input_mode": input_mode}
        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(self, prompt: str, audios: Optional[List[Tuple]] = None, input_mode: int = 2) -> Dict[str, Any]:
        """
        调用 processor 对文本和音频进行预处理，生成模型输入张量，并获取原始输入文本用于后续裁剪。
        """
        # 使用 _audio_processor 而不是 processor
        inputs = self._audio_processor(
            text=prompt,
            audios=audios,
            return_tensors="pt",
        )
        # 添加输入模式
        inputs["input_mode"] = torch.tensor([input_mode])
        # 将所有 tensor 转入模型设备
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # 解码原始文本（作为生成文本裁剪依据）
        input_text = ""
        if inputs.get("input_ids", None) is not None:
            decoded = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            if decoded:
                input_text = decoded[0]
        return {"model_inputs": inputs, "input_text": input_text}

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        press: Optional[BasePress] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        input_mode: int = 2,
        measure_time: bool = True,
        skip_invalid_responses: bool = False,  # 添加跳过无效响应的参数
        min_output_tokens: int = 25,  # 最小输出标记数
    ):
        """
        在 KV Press 上下文中调用 model.generate 生成文本。
        支持测量prefill和生成时间，以及识别音频特殊token位置。
        增加了跳过无效响应的功能，无效响应将被明确标记，不参与WER计算。
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        
        # 确保 measure_time 不会传递给模型
        if "measure_time" in generation_kwargs:
            del generation_kwargs["measure_time"]
        
        inputs = model_inputs["model_inputs"]
        input_text = model_inputs["input_text"]
        
        # 查找音频特殊token位置
        audio_token_info = {}
        if inputs.get("input_ids") is not None:
            _AUDIO_SPECIAL_TOKEN_ID = getattr(self, "_AUDIO_SPECIAL_TOKEN_ID", None)
            if _AUDIO_SPECIAL_TOKEN_ID and _AUDIO_SPECIAL_TOKEN_ID in inputs["input_ids"][0]:
                token_ids = inputs["input_ids"][0].tolist()
                audio_token_start_index = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end_index = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end_index - audio_token_start_index + 1
                audio_token_info = {
                    "start": audio_token_start_index,
                    "end": audio_token_end_index,
                    "length": audio_token_length
                }
        
        # 初始化度量指标字典，确保即使发生异常也能返回该字典
        metrics = {
            "prefill_time": 0.0,
            "generation_time": 0.0,
            "total_time": 0.0,
            "is_valid": True  # 添加有效性标志
        }
        

        # 测量prefill阶段
        if measure_time:
            prefill_start = time.time()

            with torch.no_grad(), (press(self.model) if press is not None else contextlib.nullcontext()):
                # 先执行prefill阶段，存入cache
                prefill_outputs = self.model(
                    **inputs,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                prefill_time = time.time() - prefill_start
                metrics["prefill_time"] = prefill_time

                # 使用缓存进行生成
                generation_start = time.time()
                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_kwargs
                )
                generation_time = time.time() - generation_start
                metrics["generation_time"] = generation_time
                metrics["total_time"] = prefill_time + generation_time
        else:
            # 不测量时间时直接生成
            with torch.no_grad(), (press(self.model) if press is not None else contextlib.nullcontext()):
                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_kwargs
                )

        generated_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        
        # 仅保留新生成部分，若生成文本包含输入文本前缀
        if input_text and len(generated_text) > len(input_text):
            generated_text = generated_text[len(input_text):]
        
        # 检查生成的文本是否有效
        output_tokens = len(outputs.sequences[0]) - len(inputs.get("input_ids", [[]])[0])
        is_valid = True
        
        if skip_invalid_responses and output_tokens < min_output_tokens:
            is_valid = False
            print(f"\n警告: 生成了无效响应 (tokens: {output_tokens}<{min_output_tokens})，标记为无效，将不参与WER计算")
            print(f"响应内容: {generated_text.strip() if generated_text.strip() else '空'}")
            
            # 对于无效响应，可以考虑将其替换为特殊标记或提示，让下游任务知道这是无效的
            # 这里我们将生成的文本替换为特殊标记，确保不会计入WER
            generated_text = "[INVALID_RESPONSE]" 

        metrics["is_valid"] = is_valid

        result = {
            "generated_text": generated_text, 
            "input_text": input_text,
            "audio_token_info": audio_token_info,
            "metrics": metrics,
            "output_tokens": output_tokens,  # 添加输出标记数量
            "is_valid": is_valid,  # 添加有效性标志
            "skip_wer_computation": not is_valid  # 明确指示是否应跳过WER计算
        }
        
        return result


    def postprocess(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理生成文本，清理常见前缀和特殊标记后，返回最终结果。
        同时保留音频token信息、时间测量结果和有效性标志。
        无效响应将被特殊标记，不参与后续WER计算。
        """
        text = model_outputs["generated_text"]
        
        # 如果是标记为无效的响应，直接返回特殊标记，不进行后处理
        if not model_outputs.get("is_valid", True) or text == "[INVALID_RESPONSE]":
            result = {
                "text": "[INVALID_RESPONSE]",
                "is_valid": False,
                "skip_wer_computation": True
            }
        else:
            # 正常处理有效响应
            for marker in ["spoken content:", "content:"]:
                if marker.lower() in text.lower():
                    parts = text.lower().split(marker.lower(), 1)
                    if len(parts) > 1:
                        text = parts[1].strip()
                        break
            text = text.replace("<transcribed text here>", "").strip()
            
            result = {"text": text, "is_valid": True, "skip_wer_computation": False}
        
        # 保留其他信息
        for key in ["audio_token_info", "metrics", "output_tokens"]:
            if key in model_outputs:
                result[key] = model_outputs[key]
                
        return result

    def __call__(self, prompt: str, audios: Optional[List[Tuple]] = None, press: Optional[BasePress] = None, 
                input_mode: int = 2, measure_time: bool = True, skip_invalid_responses: bool = True, 
                min_output_tokens: int = 25, **kwargs) -> Dict[str, Any]:
        """
        重写 __call__ 方法，将关键字参数转换为标准 pipeline 流程调用
        增加了跳过无效响应的选项
        """
        # 确保 measure_time 不会被传递到 model.generate() 中
        if "measure_time" in kwargs:
            del kwargs["measure_time"]
            
        pre_proc_params, forward_params, post_proc_params = self._sanitize_parameters(
            prompt=prompt, audios=audios, press=press, input_mode=input_mode, **kwargs
        )
        model_inputs = self.preprocess(**pre_proc_params)
        # 在这里传递 measure_time 参数和无效响应处理参数
        forward_params["measure_time"] = measure_time
        forward_params["skip_invalid_responses"] = skip_invalid_responses
        forward_params["min_output_tokens"] = min_output_tokens
        outputs = self._forward(model_inputs, **forward_params)
        return self.postprocess(outputs)

# 将 pipeline 注册为 kv-press-audio-transcription 任务
PIPELINE_REGISTRY.register_pipeline(
    "kv-press-audio-transcription",
    pipeline_class=KVPressAudioTranscriptionPipeline,
    pt_model=None,  # 使用已加载的模型对象时，不需自动加载
)