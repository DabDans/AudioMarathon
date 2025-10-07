import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
from scipy.io import wavfile
from scipy import signal
import librosa
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import random
import jiwer

random.seed(42)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON compatibility"""
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
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

_AUDIO_TOKEN_ID = 151646        
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.insert(0, current_dir)

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

class GlobalTimingStats:
    """Simplified global timing statistics class"""
    
    def __init__(self):
        self.timing_records = []
        self.first_sample_skipped = False
        
    def add_sample(self, wall_time, prefill_time, total_gpu_time):
        """Add sample timing record, skip first sample"""
        if not self.first_sample_skipped:
            self.first_sample_skipped = True
            return
            
        record = {
            "wall_time": wall_time,
            "prefill_time": prefill_time,
            "total_gpu_time": total_gpu_time
        }
        
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get statistics summary"""
        if len(self.timing_records) == 0:
            return {
                "count": 0,
                "avg_wall_time": 0.0,
                "avg_prefill_time": 0.0,
                "avg_total_gpu_time": 0.0
            }
        
        total_wall = sum(r["wall_time"] for r in self.timing_records)
        total_prefill = sum(r["prefill_time"] for r in self.timing_records)
        total_gpu = sum(r["total_gpu_time"] for r in self.timing_records)
        count = len(self.timing_records)
        
        return {
            "count": count,
            "avg_wall_time": total_wall / count,
            "avg_prefill_time": total_prefill / count,
            "avg_total_gpu_time": total_gpu / count
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file - HAD style"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def load_librispeech_long_dataset(base_dir, split="test-clean"):
    """Load local LibriSpeech-Long dataset"""
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Error: Dataset path does not exist: {split_dir}")
        return []
    
    print(f"Scanning dataset directory: {split_dir}")
    
    speaker_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
    
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        
        chapter_dirs = sorted([d for d in glob.glob(os.path.join(speaker_dir, "*")) if os.path.isdir(d)])
        
        for chapter_dir in chapter_dirs:
            chapter_id = os.path.basename(chapter_dir)
            
            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))
            
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
                    
                    dataset.append({
                        "id": base_name,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "path": flac_file,
                        "transcription": transcription,
                        "duration": duration,
                        "audio": {
                            "path": flac_file,
                            "array": None,  
                            "sampling_rate": None
                        }
                    })
    
    print(f"Loaded {len(dataset)} audio samples")
    return dataset

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio file according to Qwen2.5-Omni requirements"""
    
    try:
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio.shape}, sample_rate={sr}Hz")
        except Exception as e:
            print(f"librosa loading failed: {e}")
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"soundfile processing successful: shape={audio.shape}, sample_rate={sr}Hz")
                
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generated silence as replacement audio")
        
        if len(audio) == 0:
            print("Warning: Audio is empty, creating 3-second silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
            
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def librispeech_doc_to_audio(doc):
    """Load audio data from LibriSpeech document"""
    if "audio" not in doc:
        return None
    
    if doc["audio"]["array"] is None:
        try:
            audio_data = prepare_audio_for_qwen_omni(doc["audio"]["path"])
            doc["audio"]["array"] = audio_data
            doc["audio"]["sampling_rate"] = 16000  
        except Exception as e:
            print(f"Unable to load audio file {doc['audio']['path']}: {e}")
            return None
    
    return doc["audio"]["array"], doc["audio"]["sampling_rate"]

def asr_doc_to_text(doc, kwargs=None):
    """Generate prompt for English ASR task"""
    if kwargs is None:
        kwargs = {}
    
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    instruction = "Transcribe this audio accurately. Remove hesitation words like 'um', 'uh'."
    format_text = "Your response should be formatted as follows: Spoken Content:"
    
    prompt_text = f"{pre_prompt}{instruction} {format_text} <transcribed text here>{post_prompt}"
    
    return prompt_text

def clean_response(response):
    """Clean ASR response"""
    if not response or response.strip() == "":
        return ""
    
    for marker in ["spoken content:", "content:", "transcription:", "transcript:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    response = re.sub(r'<transcribed text here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    response = re.sub(r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription):\s*', '', response.strip())
    
    return response.strip()

def standardize_text(text):
    """Standardize text for fair comparison and WER calculation"""
    if not text:
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    try:
        import jiwer
        
        ref_standardized = standardize_text(reference)
        hyp_standardized = standardize_text(hypothesis)
        
        if not ref_standardized or not hyp_standardized:
            return 100.0
        
        wer = jiwer.wer(ref_standardized, hyp_standardized)
        return wer * 100  
    except ImportError:
        print("Warning: jiwer library not installed, cannot calculate WER")
        return 0.0
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 0.0

def calculate_librispeech_metrics(references, hypotheses):
    """
    Simplified LibriSpeech ASR metrics calculation, only computing WER
    
    Args:
        references: List of true transcriptions
        hypotheses: List of predicted transcriptions
        
    Returns:
        dict: Dictionary containing WER-related metrics
    """
    valid_indices = []
    clean_references = []
    clean_hypotheses = []
    wer_scores = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref and hyp:  
            valid_indices.append(i)
            clean_references.append(ref)
            clean_hypotheses.append(hyp)
            
            wer = calculate_wer(ref, hyp)
            wer_scores.append(wer)
    
    if len(clean_references) == 0:
        return {
            'wer_mean': 100.0,
            'wer_std': 0.0,
            'wer_min': 100.0,
            'wer_max': 100.0,
            'perfect_predictions': 0,
            'valid_samples': 0,
            'total_samples': len(references),
            'word_accuracy': 0.0
        }
    
    wer_mean = np.mean(wer_scores)
    wer_std = np.std(wer_scores)
    wer_min = np.min(wer_scores)
    wer_max = np.max(wer_scores)
    
    perfect_predictions = len([wer for wer in wer_scores if wer == 0.0])
    
    word_accuracy = 100.0 - wer_mean
    
    return {
        'wer_mean': float(wer_mean),
        'wer_std': float(wer_std),
        'wer_min': float(wer_min),
        'wer_max': float(wer_max),
        'perfect_predictions': int(perfect_predictions),
        'valid_samples': len(clean_references),
        'total_samples': len(references),
        'word_accuracy': float(word_accuracy),
        'wer_scores': wer_scores
    }

def main():
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
    prune_method = os.environ.get("PRUNE_METHOD", "base")

    use_random = (prune_method == "random")
    use_frame = (prune_method == "frame")
    if use_random == False and use_frame == False:
        prune_method = "fast_v"
    
    if prune_ratio == 0:
        method_is = "base"
    else:
        method_is = prune_method

    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    librispeech_path = '/data/to/your/dataset/path//librispeech-long'
    result_dir = os.environ.get("RESULTS_DIR", './LibriSpeech_Results')
    
    librispeech_path = os.path.abspath(librispeech_path)
    result_dir = os.path.abspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    output_file = f'{result_dir}/librispeech_results_qwen25.json'
    timing_output_file = f'{result_dir}/timing_stats_qwen25_{method_is}_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = GlobalTimingStats()

    print(f"\n=== LibriSpeech ASR Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {librispeech_path}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

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
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        if not hasattr(model.thinker.model.config, 'image_layer_idx'):
            model.thinker.model.config.image_layer_idx = False
        if not hasattr(model.thinker.model.config, 'audio_layer_idx'):
            model.thinker.model.config.audio_layer_idx = None
        if not hasattr(model.thinker.model.config, 'audio_token_num'):
            model.thinker.model.config.audio_token_num = None
        if not hasattr(model.thinker.model.config, 'audio_token_start'):
            model.thinker.model.config.audio_token_start = None
        if not hasattr(model.thinker.model.config, 'audio_prune_ratio'):
            model.thinker.model.config.audio_prune_ratio = 0
        if not hasattr(model.thinker.model.config, 'random'):
            model.thinker.model.config.random = False
        if not hasattr(model.thinker.model.config, 'frame'):
            model.thinker.model.config.frame = False
        print(f"Initialized thinker.model.config pruning configuration parameters")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Loading LibriSpeech dataset: {librispeech_path}")
    dataset = load_librispeech_long_dataset(librispeech_path, "test-clean")
    
    if not dataset:
        print("Error: Failed to load any data")
        return
    
    if sample_limit > 0 and len(dataset) > sample_limit:
        dataset = dataset[:sample_limit]
        print(f"Applied sample limit, processing {len(dataset)} samples")

    speaker_stats = defaultdict(int)
    for sample in dataset:
        speaker_id = sample.get("speaker_id", "unknown")
        speaker_stats[speaker_id] += 1
    
    print(f"Speaker statistics: {len(speaker_stats)} speakers")
    print(f"Sample distribution: {dict(list(speaker_stats.items())[:5])}...")

    results = []
    total_wer = 0.0
    processed_samples = 0

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,      
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(dataset)} samples...")
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="LibriSpeech ASR Evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, doc in progress_bar:
        try:
            audio_data_result = librispeech_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            
            reference = doc.get("transcription", "")
            speaker_id = doc.get("speaker_id", "unknown")
            
            prompt_text = asr_doc_to_text(doc)

            task_instruction = "You are a helpful assistant that transcribes speech audio. Please listen carefully and provide the exact transcription of what is spoken in the audio."
            full_user_prompt = f"{task_instruction}\n\n{prompt_text}"

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_np},
                        {"type": "text", "text": full_user_prompt}
                    ]
                }
            ]
            
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            
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
                
                bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                
                if bos_positions and eos_positions:
                    audio_token_start = bos_positions[0]
                    audio_token_end = eos_positions[0]
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    audio_detected = True
                    
                    model.thinker.model.config.image_layer_idx = False  
                    model.thinker.model.config.audio_layer_idx = prune_layer_idx
                    model.thinker.model.config.audio_token_num = audio_token_length
                    model.thinker.model.config.audio_token_start = audio_token_start
                    model.thinker.model.config.audio_prune_ratio = prune_ratio
                    model.thinker.model.config.random = use_random
                    model.thinker.model.config.frame = use_frame 
                    
            if not audio_detected:
                model.thinker.model.config.audio_layer_idx = None
                model.thinker.model.config.audio_prune_ratio = 0

            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            
            audio_tokens = 0
            if hasattr(processor.tokenizer, 'audio_bos_token_id') and hasattr(processor.tokenizer, 'audio_eos_token_id'):
                input_ids = inputs['input_ids'][0]
                audio_bos_positions = (input_ids == processor.tokenizer.audio_bos_token_id).nonzero(as_tuple=True)[0]
                audio_eos_positions = (input_ids == processor.tokenizer.audio_eos_token_id).nonzero(as_tuple=True)[0]
                
                if len(audio_bos_positions) > 0 and len(audio_eos_positions) > 0:
                    for bos_pos in audio_bos_positions:
                        eos_candidates = audio_eos_positions[audio_eos_positions > bos_pos]
                        if len(eos_candidates) > 0:
                            eos_pos = eos_candidates[0]
                            audio_tokens += eos_pos - bos_pos - 1
                            
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
                    if hasattr(model.thinker.model.config, 'sparse_attention_config'):
                        model.thinker.model.config.sparse_attention_config['audio_tokens'] = audio_tokens.item() if hasattr(audio_tokens, 'item') else audio_tokens
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prefill_end_event.record()
            
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
            
            decode_start_event.record()
            out_ids = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=1100,  
                thinker_do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            decode_end_event.record()
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0  

            resp = processor.batch_decode(
                out_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            if "assistant\n" in resp:
                assistant_start = resp.rfind("assistant\n") + len("assistant\n")
                resp = resp[assistant_start:].strip()
            
            hypothesis = clean_response(resp)

            wer = calculate_wer(reference, hypothesis)
            total_wer += wer
            processed_samples += 1

            current_avg_wer = total_wer / processed_samples
            
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            if sample_count % update_interval == 0 or sample_count == len(dataset):
                progress_bar.set_postfix({
                    'WER': f'{current_avg_wer:.2f}%',
                    'speaker': speaker_id,
                    'duration': f'{doc.get("duration", 0):.1f}s'
                })
                
                if is_screen_env:
                    print(f"  Progress: {sample_count}/{len(dataset)} ({sample_count/len(dataset)*100:.1f}%), "
                          f"WER: {current_avg_wer:.2f}%")
            else:
                progress_bar.set_postfix({
                    'WER': f'{current_avg_wer:.2f}%',
                    'speaker': speaker_id,
                    'duration': f'{doc.get("duration", 0):.1f}s'
                })

            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "speaker_id": speaker_id,
                "chapter_id": doc.get("chapter_id", ""),
                "path": doc.get("path", ""),
                "duration": doc.get("duration", 0),
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
                "response_text": resp
            })

            wall_time = prefill_time + decode_time
            timing_stats.add_sample(wall_time, prefill_time, wall_time)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            continue

    final_wer = total_wer / processed_samples if processed_samples > 0 else 100.0

    speaker_wer = defaultdict(list)
    for result in results:
        speaker_wer[result["speaker_id"]].append(result["wer"])
    
    speaker_avg_wer = {}
    for speaker_id, wers in speaker_wer.items():
        speaker_avg_wer[speaker_id] = sum(wers) / len(wers)

    references = [r["reference"] for r in results]
    hypotheses = [r["hypothesis"] for r in results]
    
    detailed_metrics = calculate_librispeech_metrics(references, hypotheses)

    final_results = {
        "model_name": f"Qwen2.5-Omni-3B",
        "dataset": "LibriSpeech",
        "total_samples": len(results),
        "valid_samples": detailed_metrics['valid_samples'],
        
        "wer_mean": detailed_metrics['wer_mean'],
        "word_accuracy": detailed_metrics['word_accuracy'],
        "wer_std": detailed_metrics['wer_std'],
        "wer_min": detailed_metrics['wer_min'],
        "wer_max": detailed_metrics['wer_max'],
        "perfect_predictions": detailed_metrics['perfect_predictions'],
        
        "speaker_count": len(speaker_stats),
        "speaker_wer": speaker_avg_wer,
        
        "timing_stats": timing_stats.get_summary(),
        
        "config": {
            "device": f"cuda:{gpu_id}",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "gpu_id": str(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')),
            "pruning_config": {
                "prune_layer_idx": prune_layer_idx,
                "prune_ratio": prune_ratio, 
                "prune_method": method_is
            },
            "sample_limit": sample_limit,
            "data_path": librispeech_path
        },
        
        "detailed_results": results
    }
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== LibriSpeech ASR Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Average WER: {detailed_metrics['wer_mean']:.2f}%")
    print(f"Word accuracy: {detailed_metrics['word_accuracy']:.2f}%")
    print(f"Number of speakers: {len(speaker_stats)}")
    
    print(f"\n=== WER Statistics ===")
    print(f"WER standard deviation: {detailed_metrics['wer_std']:.2f}%")
    print(f"Minimum WER: {detailed_metrics['wer_min']:.2f}%")
    print(f"Maximum WER: {detailed_metrics['wer_max']:.2f}%")
    print(f"Perfect predictions: {detailed_metrics['perfect_predictions']} ({detailed_metrics['perfect_predictions']/detailed_metrics['valid_samples']*100:.1f}%)")
    
    print(f"\nTop 5 speakers WER:")
    for speaker_id, avg_wer in list(speaker_avg_wer.items())[:5]:
        sample_count = speaker_stats[speaker_id]
        print(f"  {speaker_id}: {avg_wer:.2f}% ({sample_count} samples)")
    
    timing_summary = timing_stats.get_summary()
    print(f"\n=== Inference Time Statistics ===")
    print(f"Statistics sample count: {timing_summary['count']} (excluding first sample)")
    print(f"Average inference time: {timing_summary['avg_wall_time']:.4f} seconds")
    print(f"Average prefill time: {timing_summary['avg_prefill_time']:.4f} seconds")
    print(f"Average total GPU time: {timing_summary['avg_total_gpu_time']:.4f} seconds")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    print("="*80)

if __name__ == "__main__":
    main()