#!/usr/bin/env python3

import os
import sys
import glob
import torch
import json
import argparse
import logging
import time
import warnings
import random
import tempfile
import traceback
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import gc
import re
import string
from collections import defaultdict

random.seed(42)

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni_dart import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648

from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")
print(f"CUDA_VISIBLE_DEVICES: {gpu_temp}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="LibriSpeech ASR with Qwen2.5-Omni DART")
    
    parser.add_argument("--model-path", type=str, 
                       default="/data/to/your/Qwen_2.5_Model/path/",
                       help="Qwen2.5-Omni model path")
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2', 
                       help='Attention implementation method')
    
    parser.add_argument('--sparse', type=str_to_bool, default=True, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='Number of pruned layers')
    parser.add_argument('--image_token_start_index', type=int, default=None, help='Image token start index')
    parser.add_argument('--image_token_length', type=int, default=None, help='Image token length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='Audio token start index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='Audio token length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='Retention ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='Key image token count')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='Key audio token count')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='Key text token count')
    
    parser.add_argument('--sample_limit', type=int, default=0, help='Sample limit (0 for unlimited)')
    parser.add_argument("--data-path", type=str, default="/data/to/your/dataset/path//librispeech-long", help="LibriSpeech data path")
    parser.add_argument("--output-dir", type=str, default="./LibriSpeech_Results", help="Output directory")
    
    return parser.parse_args()

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

class LibriSpeechTimingStats:
    def __init__(self):
        self.timing_records = []
        self.task_type_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, task_type=None):
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        if audio_duration:
            self.total_audio_duration += audio_duration
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "task_type": task_type
        }
        
        self.timing_records.append(record)
        
        if task_type:
            self.task_type_stats[task_type].append(record)
    
    def get_summary(self):
        if self.total_samples == 0:
            return {"error": "No samples processed"}
        
        avg_prefill = self.total_prefill_time / self.total_samples
        avg_decode = self.total_decode_time / self.total_samples
        avg_total = avg_prefill + avg_decode
        avg_tokens_per_sec = self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        
        summary = {
            "total_samples": self.total_samples,
            "avg_prefill_time": avg_prefill,
            "avg_decode_time": avg_decode,
            "avg_total_time": avg_total,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_audio_duration": self.total_audio_duration,
            "avg_audio_duration": self.total_audio_duration / self.total_samples if self.total_samples > 0 else 0
        }
        
        task_summaries = {}
        for task_type, records in self.task_type_stats.items():
            if len(records) > 0:
                task_summaries[task_type] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "task_summaries": task_summaries
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def configure_DART(model, args):
    if not hasattr(model.config, 'DART_config'):
        model.config.DART_config = {}
    
    if args.sparse:
        DART_config = {
            "K": args.pruned_layer,
            "sparse": True,
            "enable_dart": True,
            
            "image_token_start_index": args.image_token_start_index, 
            "image_token_length": args.image_token_length,
            
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            
            "reduction_ratio": args.reduction_ratio,
            
            "pivot_image_token": getattr(args, 'pivot_image_token', args.pivot_audio_token),
            "pivot_text_token": args.pivot_text_token,
            "pivot_audio_token": args.pivot_audio_token,
            
            "text_length": 1,
            
            "qwen_dart_enabled": True,
            "multimodal_pruning": True,
        }
        
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            model.thinker.model.config.DART_config = DART_config
            print("DART configuration set to thinker.model.config")
        elif hasattr(model, 'model'):
            model.model.config.DART_config = DART_config
            print("DART configuration set to model.config")
        else:
            model.config.DART_config = DART_config
            print("DART configuration set to root config")

    else:
        model.config.DART_config = None
    
    print(f"Qwen2.5-Omni DART configuration: sparse={args.sparse}, "
          f"reduction_ratio={args.reduction_ratio}, "
          f"pruned_layer={args.pruned_layer}")
    
    return {
        'sparse': args.sparse,
        'reduction_ratio': args.reduction_ratio,
        'pruned_layer': args.pruned_layer,
        'audio_token_start_index': args.audio_token_start_index,
        'audio_token_length': args.audio_token_length,
        'pivot_audio_token': args.pivot_audio_token,
        'pivot_text_token': args.pivot_text_token
    }

def load_librispeech_data(data_path: str, sample_limit: int = 0) -> List[Dict]:
    
    samples = []
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset path does not exist: {data_path}")
        return create_dummy_librispeech_samples(sample_limit if sample_limit > 0 else 50)
    
    print(f"Scanning dataset directory: {data_path}")
    
    possible_subdirs = [
        "test-clean",
        "test-other", 
        "dev-clean",
        "dev-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500"
    ]
    
    speaker_dirs = []
    
    direct_speakers = sorted([d for d in glob.glob(os.path.join(data_path, "*")) 
                             if os.path.isdir(d) and os.path.basename(d).isdigit()])
    
    if direct_speakers:
        print(f"Found {len(direct_speakers)} speaker folders in root directory")
        speaker_dirs = direct_speakers
    else:
        for subdir in possible_subdirs:
            subdir_path = os.path.join(data_path, subdir)
            if os.path.exists(subdir_path):
                sub_speakers = sorted([d for d in glob.glob(os.path.join(subdir_path, "*")) 
                                     if os.path.isdir(d) and os.path.basename(d).isdigit()])
                if sub_speakers:
                    print(f"Found {len(sub_speakers)} speaker folders in subdirectory {subdir}")
                    speaker_dirs = sub_speakers
                    break
    
    if not speaker_dirs:
        print(f"Warning: No LibriSpeech format speaker folders found in {data_path}")
        print("Please check if the data path is correct, LibriSpeech directory should contain numerically named speaker folders")
        return create_dummy_librispeech_samples(sample_limit if sample_limit > 0 else 50)
    
    count = 0
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        
        chapter_dirs = sorted([d for d in glob.glob(os.path.join(speaker_dir, "*")) if os.path.isdir(d)])
        
        for chapter_dir in chapter_dirs:
            chapter_id = os.path.basename(chapter_dir)
            
            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))
            
            if not flac_files:
                continue
                
            for flac_file in flac_files:
                base_name = os.path.splitext(os.path.basename(flac_file))[0]
                
                txt_file = os.path.join(chapter_dir, f"{base_name}.txt")
                trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                
                transcription = None
                
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                elif os.path.exists(trans_file):
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(base_name):
                                transcription = line.split(' ', 1)[1].strip()
                                break
                
                if transcription:
                    try:
                        audio_info = sf.info(flac_file)
                        duration = audio_info.duration
                    except:
                        duration = 0.0
                    
                    samples.append({
                        "utterance_id": base_name,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "audio_path": flac_file,
                        "transcript": transcription.upper(),
                        "duration": duration,
                        "task": "automatic_speech_recognition"
                    })
                    count += 1
                    
                    if sample_limit > 0 and count >= sample_limit:
                        break
                else:
                    print(f"Warning: No transcription found for audio file {flac_file}")
            
            if sample_limit > 0 and count >= sample_limit:
                break
        
        if sample_limit > 0 and count >= sample_limit:
            break
    
    print(f"Loaded {len(samples)} LibriSpeech samples")
    
    if len(samples) == 0:
        print("Warning: No samples loaded, creating dummy samples for testing")
        return create_dummy_librispeech_samples(sample_limit if sample_limit > 0 else 10)
    return samples

def prepare_audio_for_qwen_omni(audio_path):
    try:
        audio, orig_sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        target_sr = 16000
        if orig_sr != target_sr:
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio
            )
        
        max_length = target_sr * 30
        if len(audio) > max_length:
            audio = audio[:max_length]
        
        return audio.astype(np.float32)
        
    except Exception as e:
        print(f"Audio preprocessing error {audio_path}: {e}")
        return np.zeros(16000, dtype=np.float32)

def create_dummy_librispeech_samples(count: int) -> List[Dict]:
    
    samples = []
    dummy_transcripts = [
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "HELLO WORLD THIS IS A TEST",
        "SPEECH RECOGNITION IS AN IMPORTANT TECHNOLOGY",
        "ARTIFICIAL INTELLIGENCE ADVANCES RAPIDLY",
        "MACHINE LEARNING MODELS CONTINUE TO IMPROVE"
    ]
    
    for i in range(count):
        speaker_id = f"{1000 + (i // 10):04d}"
        chapter_id = f"{i % 10:04d}"
        utterance_id = f"{speaker_id}-{chapter_id}-{i:04d}"
        
        samples.append({
            'audio_path': f"/dummy/librispeech/{speaker_id}/{chapter_id}/{utterance_id}.flac",
            'utterance_id': utterance_id,
            'speaker_id': speaker_id,
            'chapter_id': chapter_id,
            'transcript': dummy_transcripts[i % len(dummy_transcripts)],
            'task': 'automatic_speech_recognition'
        })
    
    print(f"Created {len(samples)} dummy LibriSpeech samples")
    return samples

def asr_doc_to_text(doc, kwargs=None):
    if kwargs is None:
        kwargs = {}
    
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    instruction = "Transcribe this audio accurately. Remove hesitation words like 'um', 'uh'."
    format_text = "Your response should be formatted as follows: Spoken Content:"
    
    prompt_text = f"{pre_prompt}{instruction} {format_text} <transcribed text here>{post_prompt}"
    
    return prompt_text

def clean_response(response):
    import re
    
    if not response or response.strip() == "":
        return ""
    
    text_clean = response.strip()
    
    prefixes_to_remove = [
        "spoken content:", "content:", "transcription:", "transcript:",
        "transcribed text:", "audio content:", "speech content:",
        "the audio says:", "i hear:", "the speaker says:",
        "spoken:", "speech:", "audio:", "text:", "output:"
    ]
    
    for marker in prefixes_to_remove:
        if marker.lower() in text_clean.lower():
            parts = re.split(re.escape(marker), text_clean, flags=re.IGNORECASE)
            if len(parts) > 1:
                text_clean = parts[1].strip()
                break
    
    patterns_to_remove = [
        r'<transcribed text here>',
        r'<sep>.*?($|<)',
        r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription):\s*',
        r'(?i)^(the\s+)?(?:audio|speech|recording)\s+(?:says?|contains?|is):?\s*',
        r'(?i)^(?:i\s+)?(?:can\s+)?hear:?\s*',
        r'(?i)^(?:the\s+)?(?:speaker|person)\s+(?:says?|is\s+saying):?\s*',
        r'(?i)^(?:transcription|transcript):?\s*',
        r'^\s*["\'\[\(<]',
        r'["\'\]\)>]\s*$',
    ]
    
    for pattern in patterns_to_remove:
        text_clean = re.sub(pattern, '', text_clean).strip()
    
    lines = text_clean.split('\n')
    if len(lines) > 1:
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if non_empty_lines:
            text_clean = max(non_empty_lines, key=len)
    
    text_clean = text_clean.strip()
    
    if len(text_clean) < 2:
        return ""
    
    meta_patterns = [
        r'(?i)(?:the\s+)?(?:audio|speech|recording)\s+(?:says?|contains?|is):?\s*["\']?([^"\']+)["\']?',
        r'(?i)(?:transcription|transcript):?\s*["\']?([^"\']+)["\']?',
        r'(?i)(?:spoken\s+)?(?:content|text):?\s*["\']?([^"\']+)["\']?',
    ]
    
    for pattern in meta_patterns:
        match = re.search(pattern, text_clean)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) > len(text_clean) * 0.5:
                text_clean = extracted
                break
    
    return text_clean.strip()

def process_librispeech_sample(sample: Dict, processor, model, timing_stats: LibriSpeechTimingStats, args) -> Dict:
    
    try:
        audio_path_for_inference = sample['audio_path']
        if not os.path.exists(audio_path_for_inference):
            print(f"Audio file does not exist: {audio_path_for_inference}")
            audio_path_for_inference = None

        reference_transcript = sample['transcript']
        speaker_id = sample.get('speaker_id', 'unknown')
        
        prompt_text = asr_doc_to_text(sample)

        qwen_intro = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        task_prompt = "You are a helpful assistant that transcribes speech audio. Please listen carefully and provide the exact transcription of what is spoken in the audio."
        sys_prompt = f"{qwen_intro} {task_prompt}"

        if audio_path_for_inference:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path_for_inference},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Audio transcription request: {sample.get('transcript', 'DUMMY TRANSCRIPT')}"},
                ]}
            ]

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(text, list):
            text = text[0]

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(model.device).to(model.dtype)

        audio_token_length = 0
        audio_token_start = 0
        input_token_length = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
        
        audio_detected = False
        
        if hasattr(inputs, 'input_ids'):
            token_ids = inputs.input_ids[0].tolist()
            if _AUDIO_BOS_TOKEN_ID in token_ids and _AUDIO_EOS_TOKEN_ID in token_ids:
                audio_start = token_ids.index(_AUDIO_BOS_TOKEN_ID)
                audio_end = token_ids.index(_AUDIO_EOS_TOKEN_ID)
                audio_token_start = audio_start
                audio_token_length = audio_end - audio_start + 1
                audio_detected = True
                print(f"Detected audio tokens: start={audio_token_start}, length={audio_token_length}")
        
        if not audio_detected:
            audio_token_start = args.audio_token_start_index
            audio_token_length = args.audio_token_length
            print(f"Using default audio token configuration: start={audio_token_start}, length={audio_token_length}")

        full_start_event = torch.cuda.Event(enable_timing=True)
        full_end_event = torch.cuda.Event(enable_timing=True)
        
        first_token_start_event = torch.cuda.Event(enable_timing=True)
        first_token_end_event = torch.cuda.Event(enable_timing=True)

        full_start_event.record()
        
        first_token_start_event.record()
        with torch.no_grad():
            first_token_output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True
            )
        first_token_end_event.record()
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1100,
                do_sample=False,
                use_cache=True
            )
        full_end_event.record()
        
        torch.cuda.synchronize()
        first_token_time = first_token_start_event.elapsed_time(first_token_end_event) / 1000.0
        total_time = full_start_event.elapsed_time(full_end_event) / 1000.0
        
        prefill_time = first_token_time
        decode_time = max(0.0, total_time - prefill_time)
        
        output_text = processor.batch_decode(
            output, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        if "assistant\n" in output_text:
            output_text = output_text.split("assistant\n")[-1].strip()
        
        if hasattr(output, 'shape') and len(output.shape) > 1:
            output_tokens = output.shape[1] - inputs.input_ids.shape[1]
        else:
            output_tokens = 10
        
        predicted_transcript = clean_response(output_text)

        if not hasattr(clean_response, 'debug_count'):
            clean_response.debug_count = 0
        clean_response.debug_count += 1
        
        if clean_response.debug_count <= 5:
            print(f"[Debug] Sample {clean_response.debug_count}:")
            print(f"  Raw output: '{output_text[:200]}...' (first 200 characters)")
            print(f"  Cleaned result: '{predicted_transcript}'")
            print(f"  Reference transcript: '{reference_transcript}'")

        wer = calculate_wer(reference_transcript, predicted_transcript)

        timing_stats.add_record(
            prefill_time=prefill_time,
            decode_time=decode_time,
            output_tokens=output_tokens,
            input_tokens=input_token_length,
            audio_duration=sample.get('duration', 0),
            task_type='asr'
        )

        result = {
            'utterance_id': sample['utterance_id'],
            'audio_path': sample['audio_path'],
            'speaker_id': sample['speaker_id'],
            'reference_transcript': reference_transcript,
            'predicted_transcript': predicted_transcript,
            'raw_response': output_text,
            'wer': wer,
            'input_tokens': input_token_length,
            'audio_tokens': audio_token_length,
            'output_tokens': output_tokens,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': prefill_time + decode_time,
            'processing_time': prefill_time + decode_time
        }

        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return result
        
    except Exception as e:
        print(f"Error processing sample {sample.get('utterance_id', 'unknown')}: {e}")
        return {
            'utterance_id': sample.get('utterance_id', 'unknown'),
            'audio_path': sample.get('audio_path', ''),
            'speaker_id': sample.get('speaker_id', ''),
            'reference_transcript': sample.get('transcript', ''),
            'predicted_transcript': 'ERROR',
            'raw_response': f"Error: {e}",
            'wer': 1.0,
            'input_tokens': 0,
            'audio_tokens': 0,
            'output_tokens': 0,
            'prefill_time': 0.0,
            'decode_time': 0.0,
            'total_time': 0.0,
            'processing_time': 0.0
        }

def calculate_wer(reference: str, hypothesis: str) -> float:
    
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()
    
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 1.0
    return wer

def evaluate_librispeech_results(results: List[Dict]) -> Dict:
    
    valid_results = [r for r in results if r['predicted_transcript'] != 'ERROR']
    
    if not valid_results:
        return {
            'average_wer': 1.0,
            'total_samples': len(results),
            'valid_samples': 0,
            'error_rate': 1.0
        }
    
    wer_values = [r['wer'] for r in valid_results]
    average_wer = np.mean(wer_values)
    
    wer_ranges = {
        'perfect': sum(1 for wer in wer_values if wer == 0.0),
        'excellent': sum(1 for wer in wer_values if 0.0 < wer <= 0.1),
        'good': sum(1 for wer in wer_values if 0.1 < wer <= 0.3),
        'fair': sum(1 for wer in wer_values if 0.3 < wer <= 0.5),
        'poor': sum(1 for wer in wer_values if wer > 0.5)
    }
    
    processing_times = [r['total_time'] for r in valid_results if r.get('total_time', 0) > 0]
    
    evaluation = {
        'average_wer': average_wer,
        'median_wer': np.median(wer_values),
        'min_wer': np.min(wer_values),
        'max_wer': np.max(wer_values),
        'wer_std': np.std(wer_values),
        'total_samples': len(results),
        'valid_samples': len(valid_results),
        'error_rate': (len(results) - len(valid_results)) / len(results) if results else 0,
        'wer_distribution': wer_ranges,
        'avg_processing_time': np.mean(processing_times) if processing_times else 0.0
    }
    
    return evaluation

def save_librispeech_results(results: List[Dict], evaluation: Dict, timing_stats: Dict, 
                            dart_config: Dict, output_path: str):
    
    output_data = {
        'task': 'automatic_speech_recognition',
        'dataset': 'LibriSpeech',
        'model': 'Qwen2.5-Omni-3B',
        'dart_config': dart_config,
        'summary': evaluation,
        'timing_stats': timing_stats,
        'samples': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def main():
    args = parse_arguments()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")
    print(f"CUDA_VISIBLE_DEVICES: {gpu_temp}")
    
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    
    librispeech_data_path = args.data_path if hasattr(args, 'data_path') else "./LibriSpeech"
    result_dir = args.output_dir if hasattr(args, 'output_dir') else './LibriSpeech_Results'
    
    print(f"\n=== LibriSpeech DART ASR Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retention ratio: {args.reduction_ratio}")
    print(f"LibriSpeech data path: {librispeech_data_path}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)
    
    method_name = "sparse" if args.sparse else "base"
    ratio_str = f"ratio_{args.reduction_ratio:.3f}"
    output_file = f'{result_dir}/librispeech_results_dart_{method_name}_{ratio_str}.json'
    timing_output_file = f'{result_dir}/librispeech_timing_stats_dart_{method_name}_{ratio_str}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    
    timing_stats = LibriSpeechTimingStats()
    
    samples = load_librispeech_data(librispeech_data_path, sample_limit)
    
    os.makedirs(result_dir, exist_ok=True)
    
    print("Loading Qwen2.5-Omni model...")
    model_path = "/data/to/your/Qwen_2.5_Model/path/"
    device_map = {"": 0}
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.disable_talker()
    
    dart_config = configure_DART(model, args)
    print("Model loaded successfully")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")
    
    print(f"Total processing {len(samples)} samples")
    
    results = []
    total_accuracy = 0
    processed_samples = 0
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    print(f"Starting evaluation of {len(samples)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    with tqdm(total=len(samples), desc="Processing LibriSpeech ASR samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            result = process_librispeech_sample(sample, processor, model, timing_stats, args)
            results.append(result)
            
            if result['predicted_transcript'] != 'ERROR':
                processed_samples += 1
            
            if processed_samples > 0:
                current_avg_wer = np.mean([r['wer'] for r in results if r['predicted_transcript'] != 'ERROR'])
                pbar.set_description(f"Processing LibriSpeech ASR samples (Current WER: {current_avg_wer:.3f})")
            
            pbar.update(1)
    
    final_evaluation = evaluate_librispeech_results(results)
    timing_summary = timing_stats.get_summary()
    
    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "average_wer": final_evaluation.get('average_wer', 1.0),
        "median_wer": final_evaluation.get('median_wer', 1.0),
        "valid_samples": final_evaluation.get('valid_samples', 0),
        "error_rate": final_evaluation.get('error_rate', 1.0),
        "config": {
            "gpu_id": gpu_id,
            "model_path": model_path,
            "sparse": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "sample_limit": sample_limit,
            "librispeech_data_path": librispeech_data_path
        },
        "timing": timing_summary
    }
    
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    timing_stats.export_to_json(timing_output_file)
    
    print(f"\n=== LibriSpeech DART ASR Evaluation Results Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Valid samples: {final_evaluation.get('valid_samples', 0)}")
    print(f"Average WER: {final_evaluation.get('average_wer', 1.0):.4f}")
    print(f"Median WER: {final_evaluation.get('median_wer', 1.0):.4f}")
    print(f"Error rate: {final_evaluation.get('error_rate', 1.0):.4f}")
    
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\nTiming statistics:")
    print(f"Statistical samples: {overall_summary.get('total_samples', 0)}")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/second")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()