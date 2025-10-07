import os
import sys
import torch
import json
import time
import gc
import re
import glob
import soundfile as sf
import numpy as np
import librosa
import warnings
import traceback
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from tqdm import tqdm
from collections import defaultdict

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

# Suppress transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("Successfully imported required modules")

class LibriSpeechTimingStats:
    """Tracks inference timing statistics for LibriSpeech ASR tasks"""
    def __init__(self):
        self.timing_records = []
        self.speaker_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.total_audio_duration = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, speaker_id=None):
        """Add a timing record"""
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
            "speaker_id": speaker_id
        }
        
        self.timing_records.append(record)
        
        if speaker_id:
            self.speaker_stats[speaker_id].append(record)
    
    def get_summary(self):
        """Get overall statistics summary"""
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
        
        # Add speaker statistics
        speaker_summaries = {}
        for speaker_id, records in self.speaker_stats.items():
            if len(records) > 0:
                speaker_summaries[speaker_id] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "speaker_summaries": speaker_summaries
        }
    
    def export_to_json(self, output_file):
        """Export statistics data to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

class CudaEventTimingStats:
    """CUDA Event batch timing statistics class"""
    
    def __init__(self):
        self.timing_records = []
        self.prefill_times = []
        self.decode_times = []
        self.total_times = []
    
    def add_timing_record(self, prefill_time, decode_time, total_time):
        """Add a timing measurement record"""
        self.prefill_times.append(prefill_time)
        self.decode_times.append(decode_time)
        self.total_times.append(total_time)
        
        self.timing_records.append({
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time
        })
    
    def get_time_statistics(self, times_list, name=""):
        """Calculate timing statistics (average only)"""
        if not times_list:
            return {}
        
        stats = {
            f"{name}_avg": sum(times_list) / len(times_list),
            f"{name}_count": len(times_list)
        }
        return stats
    
    def get_full_statistics(self):
        """Get full timing statistics"""
        stats = {}
        stats.update(self.get_time_statistics(self.prefill_times, "prefill"))
        stats.update(self.get_time_statistics(self.decode_times, "decode"))
        stats.update(self.get_time_statistics(self.total_times, "total"))
        return stats

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sampling rate"""
    if original_sr == target_sr:
        return audio_array
    
    # Use librosa for resampling
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30-second chunks (480000 samples @ 16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio with librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Load audio using librosa (official recommended method)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate does not match
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Split audio if longer than 30 seconds
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            print(f"Audio length {len(audio)} exceeds 30 seconds limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # Return a single chunk if <= 30s
            return [audio], sample_rate
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        # Return silence chunk list
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_librispeech_long_dataset(base_dir, split="test-clean"):
    """Load local LibriSpeech-Long dataset"""
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Error: Dataset path does not exist: {split_dir}")
        return []
    
    print(f"Scanning dataset directory: {split_dir}")
    
    # Traverse all speaker ID folders
    speaker_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
    
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        
        # Traverse all chapter folders under the speaker
        chapter_dirs = sorted([d for d in glob.glob(os.path.join(speaker_dir, "*")) if os.path.isdir(d)])
        
        for chapter_dir in chapter_dirs:
            chapter_id = os.path.basename(chapter_dir)
            
            # Find all flac files
            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))
            
            for flac_file in flac_files:
                # Infer corresponding txt file path
                base_name = os.path.splitext(os.path.basename(flac_file))[0]
                
                # Find transcription file (.txt or .trans.txt)
                txt_file = os.path.join(chapter_dir, f"{base_name}.txt")
                trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                
                transcription = None
                
                # Prefer single txt file
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                # Otherwise look for line in trans.txt
                elif os.path.exists(trans_file):
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(base_name):
                                # Format: "speaker-chapter-utterance transcription"
                                parts = line.strip().split(' ', 1)
                                if len(parts) > 1:
                                    transcription = parts[1]
                                break
                
                if transcription:
                    try:
                        # Get audio file info
                        audio_info = sf.info(flac_file)
                        
                        # Create dataset item
                        item = {
                            "path": flac_file.replace(base_dir, "/data/to/your/librispeech-long"),
                            "audio": {
                                "path": flac_file.replace(base_dir, "/data/to/your/librispeech-long"),
                                "array": None,  # Lazy load
                                "sampling_rate": audio_info.samplerate
                            },
                            "transcription": transcription,
                            "duration": audio_info.duration,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "language": "en",
                            "id": f"{speaker_id}_{chapter_id}_{base_name}"
                        }
                        
                        dataset.append(item)
                        
                    except Exception as e:
                        print(f"Cannot process audio file {flac_file}: {e}")
                        continue
    
    print(f"Loaded {len(dataset)} audio samples")
    return dataset

def clean_response(response):
    """Clean ASR response"""
    if not response or response.strip() == "":
        return ""
    
    # Clean prefix and special markers
    for marker in ["spoken content:", "content:", "transcription:", "transcript:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    # Remove other common markers
    response = re.sub(r'<transcribed text here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    response = re.sub(r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription):\s*', '', response.strip())
    
    return response.strip()

def standardize_text(text):
    """Standardize text for fair comparison and WER calculation"""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Standardize punctuation
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_wer(reference, hypothesis):
    """Calculate word error rate (WER)"""
    try:
        import jiwer
        
        # Standardize text
        ref_standardized = standardize_text(reference)
        hyp_standardized = standardize_text(hypothesis)
        
        if not ref_standardized or not hyp_standardized:
            return 100.0
        
        # Calculate WER
        wer = jiwer.wer(ref_standardized, hyp_standardized)
        return wer * 100  # Percentage
    except ImportError:
        print("Warning: jiwer library not installed, cannot compute WER")
        return 0.0
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 0.0

def cuda_timing_inference(model, processor, inputs, max_new_tokens=256):
    """
    Inference function using CUDA Event API for accurate GPU timing measurement
    """
    
    # Ensure GPU is idle for precise timing
    torch.cuda.synchronize()
    
    # Create CUDA events
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        # === Stage 1: Prefill timing ===
        event_start.record()
        
        # Prefill computation
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, 
                           output_hidden_states=False, return_dict=True)
        
        event_prefill_end.record()
        
        # === Stage 2: Full Generation timing ===
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True
            )
        
        event_total_end.record()
        
        # === Synchronize events ===
        event_start.synchronize()
        event_prefill_end.synchronize()
        event_total_end.synchronize()
        
        # === Calculate precise time differences ===
        prefill_time = event_start.elapsed_time(event_prefill_end) / 1000.0
        total_time = event_start.elapsed_time(event_total_end) / 1000.0
        decode_time = event_prefill_end.elapsed_time(event_total_end) / 1000.0
        
        # Decode output
        if hasattr(out_ids, 'sequences'):
            tokens = out_ids.sequences[:, inputs['input_ids'].shape[1]:]
        else:
            tokens = out_ids[:, inputs['input_ids'].shape[1]:]
        
        output_tokens = len(tokens[0])
        response_text = processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
        
        return {
            'response_text': response_text,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'output_tokens': output_tokens,
            'generated_ids': out_ids,
            'tokens': tokens,
            'outputs': outputs,
            'tokens_per_second': output_tokens / decode_time if decode_time > 0 else 0
        }
        
    finally:
        pass

def main():
    # Get environment variable config
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    print(f"Using GPU ID: {gpu_id}")

    # Audio pruning config
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
    prune_method = os.environ.get("PRUNE_METHOD", "base")

    # Set flags based on method name
    use_random = (prune_method == "random")
    use_frame = (prune_method == "frame")
    if use_random == False and use_frame == False:
        prune_method = "fast_v"
    
    # Decide method name
    if prune_ratio == 0:
        method_is = "base"
    else:
        method_is = prune_method

    # Sample limit and debug mode
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
    
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    if debug_mode:
        print("Debug mode enabled - will show detailed output")

    # Data path config
    librispeech_path = os.environ.get("LIBRISPEECH_PATH", "/data/to/your/librispeech-long")
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/LibriSpeech_Results')
    os.makedirs(result_dir, exist_ok=True)

    # Output file paths
    output_file = f'{result_dir}/LibriSpeech_Aero1_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    timing_output_file = f'{result_dir}/LibriSpeech_Aero1_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    cuda_event_output_file = f'{result_dir}/LibriSpeech_Aero1_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    print(f"CUDA Event statistics will be saved to: {cuda_event_output_file}")

    # Create timing stats objects
    timing_stats = LibriSpeechTimingStats()
    cuda_event_stats = CudaEventTimingStats()

    print(f"\n=== LibriSpeech ASR Evaluation Config (Aero-1) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Data path: {librispeech_path}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    # Step1: Load Aero-1 model
    print("Loading Aero-1 model...")
    sys.stdout.flush()
    
    # Model path config - use official model name
    model_name = "lmms-lab/Aero-1-Audio-1.5B"
    print(f"Using Aero-1 model: {model_name}")
    sys.stdout.flush()
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        revision="main",
        trust_remote_code=True
    )
    print("Successfully loaded Aero processor")
    sys.stdout.flush()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    model.eval()
    print("Successfully loaded Aero-1 model")
    sys.stdout.flush()

    # Set audio pruning config
    if prune_ratio > 0:
        print(f"Note: Official Aero-1 model, audio pruning feature may not be available")
        print(f"Pruning config: layer={prune_layer_idx}, ratio={prune_ratio}, method={prune_method}")

    # Load LibriSpeech dataset
    print(f"Loading LibriSpeech dataset: {librispeech_path}")
    dataset = load_librispeech_long_dataset(librispeech_path, "test-clean")
    
    if not dataset:
        print("Error: No data loaded")
        return
    
    # Apply sample limit
    if sample_limit > 0 and len(dataset) > sample_limit:
        dataset = dataset[:sample_limit]
        print(f"Applied sample limit, processing {len(dataset)} samples")

    # Speaker statistics
    speaker_stats = defaultdict(int)
    for sample in dataset:
        speaker_id = sample.get("speaker_id", "unknown")
        speaker_stats[speaker_id] += 1
    
    print(f"Speaker statistics: {len(speaker_stats)} speakers")
    print(f"Sample distribution: {dict(list(speaker_stats.items())[:5])}...")

    # Print initial memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")

    results = []
    total_wer = 0.0
    processed_samples = 0

    print(f"Starting evaluation of {len(dataset)} samples...")
    
    # Detect screen or non-interactive environment
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        sys.stdout.flush()
    
    # Set tqdm parameters
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout,
        'mininterval': 0.1,
        'maxinterval': 1.0,
        'disable': False,
        'leave': True,
        'position': 0
    }
    
    if is_screen_env:
        tqdm_kwargs['mininterval'] = 0.05
        tqdm_kwargs['maxinterval'] = 0.5

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="LibriSpeech ASR Evaluation (Aero-1)", **tqdm_kwargs)

    for idx, doc in progress_bar:
        
        # Predefine variables to avoid UnboundLocalError
        prefill_time = 0
        decode_time = 0
        output_tokens = 0
        audio_token_length = 0
        hypothesis = ""
        wer = 100.0
        
        try:
            # Get audio path and reference transcription
            audio_path = doc.get("path", "")
            reference = doc.get("transcription", "")
            speaker_id = doc.get("speaker_id", "unknown")
            duration = doc.get("duration", 0)
            
            # Use official message format - supports multiple audio chunks
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Prepare audio input - returns list of audio chunks
            audio_chunks, sample_rate = prepare_audio_for_processor(audio_path)
            
            # Add audio content for each chunk to message
            for chunk in audio_chunks:
                messages[0]["content"].append({
                    "type": "audio",
                    "audio": "placeholder",  # This will be replaced with actual audio
                })
            
            # Add ASR task text content
            messages[0]["content"].append({
                "type": "text",
                "text": "Transcribe this audio accurately. Remove hesitation words like 'um', 'uh'. Provide only the transcribed text without any additional comments."
            })
            
            # Use chat template to process messages
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Use processor to process text and audio chunks
            inputs = processor(
                text=prompt,
                audios=audio_chunks,  # Pass list of audio chunks
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to("cuda")
            
            # Calculate audio token length (for statistics)
            audio_token_length = 0
            _AUDIO_SPECIAL_TOKEN_ID = 151667
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end - audio_token_start + 1
                
                model.config.image_layer_idx = None
                model.config.audio_layer_idx = prune_layer_idx
                model.config.audio_token_num = audio_token_length
                model.config.audio_token_start = audio_token_start
                model.config.audio_prune_ratio = prune_ratio
                model.config.random = use_random
                model.config.frame = use_frame
            
            # Show detailed info only in debug mode
            if debug_mode:
                print(f"Processing audio: {os.path.basename(audio_path)}")
                print(f"Speaker ID: {speaker_id}")
                print(f"Audio duration: {duration:.2f}s")
                print(f"Number of audio chunks: {len(audio_chunks)}")
                print(f"Reference transcription: {reference}")
                print(f"Estimated audio token length: {audio_token_length}")
                sys.stdout.flush()
            
            # Inference with precise CUDA Event timing
            result = cuda_timing_inference(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=256
            )
            
            # Get results
            output = result['response_text']
            prefill_time = result['prefill_time']
            decode_time = result['decode_time']
            total_time = result['total_time']
            output_tokens = result['output_tokens']
            
            # Clean ASR response
            hypothesis = clean_response(output)
            
            # Calculate WER
            wer = calculate_wer(reference, hypothesis)
            total_wer += wer
            processed_samples += 1

            current_avg_wer = total_wer / processed_samples
            
            # Show detailed results only in debug mode
            if debug_mode:
                print(f"Model output: '{output}'")
                print(f"Cleaned transcription: '{hypothesis}'")
                print(f"Reference transcription: '{reference}'")
                print(f"WER: {wer:.2f}%")
                print(f"Inference time: total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
                print(f"Output tokens: {output_tokens}")
                print("=" * 50)
                sys.stdout.flush()

            # Save detailed results
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "speaker_id": speaker_id,
                "chapter_id": doc.get("chapter_id", ""),
                "path": doc.get("path", ""),
                "duration": duration,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
                "response_text": output,
                "audio_chunks": len(audio_chunks),
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time
            })

            # Collect timing info
            timing_stats.add_record(
                prefill_time, decode_time, 
                output_tokens,
                inputs["input_ids"].shape[1],
                duration,
                speaker_id
            )
            
            # Collect CUDA Event timing info
            cuda_event_stats.add_timing_record(prefill_time, decode_time, total_time)

            # Memory cleanup
            if 'inputs' in locals():
                del inputs
            if 'audio_chunks' in locals():
                del audio_chunks
            if 'result' in locals():
                del result
            
            torch.cuda.empty_cache()
            
            # Deep cleanup every 10 samples
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Print memory usage every 100 samples
                if (idx + 1) % 100 == 0:
                    allocated, reserved = get_gpu_memory_usage()
                    print(f"  [Sample {idx+1}] GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            if debug_mode:
                traceback.print_exc()
            
            # Set error variable values
            hypothesis = "ERROR"
            wer = 100.0
            prefill_time = 0
            decode_time = 0
            output_tokens = 0
            audio_token_length = 0
            
            # Save error result
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "speaker_id": doc.get("speaker_id", "unknown"),
                "chapter_id": doc.get("chapter_id", ""),
                "path": doc.get("path", ""),
                "duration": doc.get("duration", 0),
                "reference": doc.get("transcription", ""),
                "hypothesis": hypothesis,
                "wer": wer,
                "response_text": "ERROR",
                "audio_chunks": 1,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            })
            
            processed_samples += 1
            total_wer += wer
            current_avg_wer = total_wer / processed_samples
            
            continue
        
        # Update progress bar
        update_interval = 50 if is_screen_env else 20
        sample_count = idx + 1
        
        if sample_count % update_interval == 0 or sample_count == len(dataset):
            progress_bar.set_postfix_str(
                f"WER:{current_avg_wer:.1f}%, Speaker:{speaker_id[:8]}, Duration:{duration:.1f}s"
            )
            
            if is_screen_env:
                print(f"Progress: {sample_count}/{len(dataset)} ({sample_count/len(dataset)*100:.1f}%), "
                      f"Average WER: {current_avg_wer:.1f}%")
                sys.stdout.flush()
        
        progress_bar.update()
        
        if is_screen_env and sample_count % 25 == 0:
            sys.stdout.flush()

    # Final statistics
    final_wer = total_wer / processed_samples if processed_samples > 0 else 100.0

    # WER by speaker
    speaker_wer = defaultdict(list)
    for result in results:
        speaker_wer[result["speaker_id"]].append(result["wer"])
    
    speaker_avg_wer = {}
    for speaker_id, wers in speaker_wer.items():
        speaker_avg_wer[speaker_id] = sum(wers) / len(wers)

    # Create result summary
    summary = {
        "total_samples": len(results),
        "processed_samples": processed_samples,
        "overall_wer": final_wer,
        "speaker_count": len(speaker_stats),
        "speaker_wer": speaker_avg_wer,
        "config": {
            "model_name": "Aero-1-Audio-1.5B",
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "data_path": librispeech_path
        },
        "timing": timing_stats.get_summary()
    }

    # Save results
    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Save timing statistics
    timing_stats.export_to_json(timing_output_file)
    
    # Save CUDA Event statistics to separate file
    cuda_event_full_stats = cuda_event_stats.get_full_statistics()
    cuda_event_full_stats['detailed_records'] = cuda_event_stats.timing_records
    
    with open(cuda_event_output_file, "w", encoding="utf-8") as f:
        json.dump(cuda_event_full_stats, f, ensure_ascii=False, indent=2)

    # Output result summary
    print(f"\n=== LibriSpeech ASR Evaluation Summary (Aero-1) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"Prune config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, method={method_is}")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Overall WER: {final_wer:.2f}%")
    print(f"Number of speakers: {len(speaker_stats)}")
    
    print(f"\nTop 5 speakers WER:")
    for speaker_id, avg_wer in list(speaker_avg_wer.items())[:5]:
        sample_count = speaker_stats[speaker_id]
        print(f"  {speaker_id}: {avg_wer:.2f}% ({sample_count} samples)")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\nAverage inference time: {overall_summary.get('avg_total_time', 0):.4f}s")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f}s")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f}s")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/s")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    print(f"CUDA Event statistics saved to: {cuda_event_output_file}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()