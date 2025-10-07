import argparse
import torch
import os
import sys
import json
import time
import gc
import re
import glob
import random
import soundfile as sf
import numpy as np
import librosa
import warnings
import traceback
from collections import defaultdict

try:
    import jiwer
except ImportError:
    print("Warning: jiwer library not installed, unable to compute WER")
    jiwer = None

# Add sklearn evaluation metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import logging
from tqdm import tqdm

# Environment config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'

# Disable transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class LibriSpeechTimingStats:
    """Track LibriSpeech ASR inference timing statistics"""
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
        """Get overall statistics summary (excluding first sample)"""
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
        """Export statistics to JSON file"""
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
        """Calculate timing statistics (mean only)"""
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

def str_to_bool(value):
    if value.lower() in ('true', 't', '1', 'yes'):
        return True
    elif value.lower() in ('false', 'f', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/Aero-1-Audio-1.5B")
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attn_implementation')
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--image_token_start_index', type=int, default=None, help='image_token_start_index')
    parser.add_argument('--image_token_length', type=int, default=None, help='image_token_length')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio')
    parser.add_argument('--pivot_image_token', type=int, default=None, help='pivot_image_token')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    return parser.parse_args()

def configure_DART_for_aero1(model, args):
    """Configure DART parameters for Aero-1 model"""
    if args.sparse:
        # Aero-1 uses different config
        model.config.image_layer_idx = None
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = 1.0 - args.reduction_ratio  # convert to prune ratio
        model.config.random = False  # default non-random prune
        model.config.frame = False   # default non-frame prune
        print(f"Configuring Aero-1 DART: layer={args.pruned_layer}, prune_ratio={1.0 - args.reduction_ratio}")
    else:
        # Disable prune
        model.config.audio_layer_idx = None
        model.config.audio_prune_ratio = 0.0

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sample rate"""
    if original_sr == target_sr:
        return audio_array
    
    # Use librosa for resampling
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30-second chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio with librosa and split, compatible with Aero-1 official example"""
    
    try:
        # Use librosa to load audio (official recommended)
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        
        # Ensure dtype is float32
        audio = audio.astype(np.float32)
        
        # Downsample if sample rate mismatch
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        
        # If audio longer than 30 seconds, split
        if len(audio) > 480000:  # 30s @ 16kHz
            audio_chunks = split_audio(audio)
            print(f"Audio length {len(audio)} exceeds 30s limit, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            # If not, return single chunk list
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
        print(f"Error: dataset path does not exist: {split_dir}")
        return []
    
    print(f"Scanning dataset directory: {split_dir}")
    
    # Traverse all speaker ID folders
    speaker_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
    
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        
        # Traverse all chapter folders
        chapter_dirs = sorted([d for d in glob.glob(os.path.join(speaker_dir, "*")) if os.path.isdir(d)])
        
        for chapter_dir in chapter_dirs:
            chapter_id = os.path.basename(chapter_dir)
            
            # Find all flac files
            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))
            
            for flac_file in flac_files:
                # Infer corresponding txt path
                base_name = os.path.splitext(os.path.basename(flac_file))[0]
                
                # Look for transcript file (.txt or .trans.txt)
                txt_file = os.path.join(chapter_dir, f"{base_name}.txt")
                trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                
                transcription = None
                
                # Prefer separate txt file
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                # Otherwise look in trans.txt for matching line
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
                            "path": flac_file,
                            "audio": {
                                "path": flac_file,
                                "array": None,  # lazy load
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
                        print(f"Unable to process audio file {flac_file}: {e}")
                        continue
    
    print(f"Loaded {len(dataset)} audio samples")
    return dataset

def remove_duplicated_sentences(text):
    """Remove duplicated sentences from text"""
    if not text:
        return text
        
    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If only one sentence or empty, return as is
    if len(sentences) <= 1:
        return text
        
    # Deduplicate with set while preserving order
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        normalized = sentence.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)
    
    return ' '.join(unique_sentences)

def clean_response(response):
    """Clean ASR response"""
    if not response or response.strip() == "":
        return ""
    
    # Clean common prefixes
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
    
    # Remove duplicate content
    response = remove_duplicated_sentences(response)
    
    return response.strip()

def standardize_text(text):
    """Standardize text for fair comparison and WER calculation"""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Standardize abbreviations
    text = re.sub(r'st\.', 'st', text)
    text = re.sub(r'mr\.', 'mr', text)
    text = re.sub(r'mrs\.', 'mrs', text)
    text = re.sub(r'dr\.', 'dr', text)
    text = re.sub(r'prof\.', 'prof', text)
    
    # Standardize punctuation
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class ASRErrorAnalyzer:
    """Simplified ASR error analyzer"""
    def __init__(self):
        self.substitution_table = {}
        self.deletion_table = {}
        self.insertion_table = {}
    
    def analyze(self, references, hypotheses):
        """Analyze transcription errors"""
        if jiwer is None:
            return {'wer': 100.0, 'ser': 100.0}
        
        # Standardize reference and prediction for accurate WER
        standardized_refs = [standardize_text(ref) for ref in references]
        standardized_hyps = [standardize_text(hyp) for hyp in hypotheses]
        
        # Set jiwer transformation for same standardization
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords()
        ])
        
        # Compute overall WER
        wer = jiwer.wer(standardized_refs, standardized_hyps, 
                        truth_transform=transformation, 
                        hypothesis_transform=transformation)
        
        # Sentence error rate (SER)
        ser = sum(1 for ref, hyp in zip(standardized_refs, standardized_hyps) 
                  if ref.strip() != hyp.strip()) / len(references)
        
        return {'wer': wer * 100, 'ser': ser * 100}

def evaluate_asr_results(results):
    """Evaluate ASR results"""
    if not results:
        return 100.0, {}
    
    # Extract references and predictions
    refs, hyps = [], []
    for result in results:
        refs.append(result["gt"])
        hyps.append(result["pred"])
    
    # Error analyzer
    analyzer = ASRErrorAnalyzer()
    
    # Analyze errors
    analysis_results = analyzer.analyze(refs, hyps)
    
    # Analysis report
    report = {
        "total_samples": len(results),
        "WER": analysis_results['wer'],
        "SER": analysis_results['ser']
    }
    
    return analysis_results['wer'], report

def calculate_wer(reference, hypothesis):
    """Compute Word Error Rate (WER)"""
    try:
        if jiwer is None:
            return 0.0
        
        # Standardize text
        ref_standardized = standardize_text(reference)
        hyp_standardized = standardize_text(hypothesis)
        
        if not ref_standardized or not hyp_standardized:
            return 100.0
        
        # Compute WER
        wer = jiwer.wer(ref_standardized, hyp_standardized)
        return wer * 100  # percent
    except ImportError:
        print("Warning: jiwer library not installed, unable to compute WER")
        return 0.0
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 0.0

def wer_to_quality_category(wer):
    """Map WER to quality category for classification analysis"""
    if wer <= 5.0:
        return "excellent"
    elif wer <= 15.0:
        return "good"
    elif wer <= 30.0:
        return "fair"
    else:
        return "poor"

def generate_sklearn_librispeech_dart_evaluation_report(references, hypotheses, speaker_ids=None, wer_scores=None):
    """
    Generate detailed LibriSpeech ASR evaluation report using sklearn (DART version)
    
    Args:
        references: list of reference transcripts
        hypotheses: list of predicted transcripts  
        speaker_ids: list of speaker IDs for speaker-level analysis
        wer_scores: list of WER scores for quality analysis
    
    Returns:
        dict: metrics dictionary
    """
    if not references or not hypotheses or len(references) != len(hypotheses):
        return {"error": "Invalid input data for evaluation"}
    
    # Filter valid references and predictions
    valid_indices = []
    valid_references = []
    valid_hypotheses = []
    valid_wers = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        # Check for valid transcription
        if (ref and hyp and 
            ref.strip() and hyp.strip() and 
            hyp.strip().upper() not in ['ERROR', 'EMPTY', 'FAILED']):
            valid_indices.append(i)
            valid_references.append(ref)
            valid_hypotheses.append(hyp)
            
            # Compute WER for sample
            if wer_scores and i < len(wer_scores):
                valid_wers.append(wer_scores[i])
            else:
                sample_wer = calculate_wer(ref, hyp)
                valid_wers.append(sample_wer)
    
    if not valid_references:
        return {"error": "No valid transcriptions for evaluation"}
    
    # Map WER to quality category for classification
    quality_categories_true = []
    quality_categories_pred = []
    
    for i, (ref, hyp, wer) in enumerate(zip(valid_references, valid_hypotheses, valid_wers)):
        # Reference transcript length based "true" quality (simplified assumption)
        ref_words = len(standardize_text(ref).split())
        if ref_words <= 5:
            true_quality = "short"
        elif ref_words <= 15:
            true_quality = "medium"
        else:
            true_quality = "long"
        
        # Predicted quality by WER
        pred_quality = wer_to_quality_category(wer)
        
        quality_categories_true.append(true_quality)
        quality_categories_pred.append(pred_quality)
    
    # Compute transcription quality metrics
    try:
        quality_accuracy = accuracy_score(quality_categories_true, quality_categories_pred)
        quality_precision, quality_recall, quality_f1, _ = precision_recall_fscore_support(
            quality_categories_true, quality_categories_pred, average='weighted', zero_division=0
        )
        
        # Quality classification report
        quality_labels = sorted(list(set(quality_categories_true + quality_categories_pred)))
        quality_classification_rep = classification_report(
            quality_categories_true, quality_categories_pred,
            target_names=quality_labels,
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        print(f"Error computing quality classification metrics: {e}")
        quality_accuracy = 0.0
        quality_precision = quality_recall = quality_f1 = 0.0
        quality_classification_rep = {}
    
    # Overall ASR performance metrics
    avg_wer = sum(valid_wers) / len(valid_wers) if valid_wers else 100.0
    wer_std = np.std(valid_wers) if len(valid_wers) > 1 else 0.0
    
    # Sample counts by WER range
    wer_distribution = {
        "excellent_count": sum(1 for w in valid_wers if w <= 5.0),
        "good_count": sum(1 for w in valid_wers if 5.0 < w <= 15.0),
        "fair_count": sum(1 for w in valid_wers if 15.0 < w <= 30.0),
        "poor_count": sum(1 for w in valid_wers if w > 30.0)
    }
    
    # Build evaluation report
    evaluation_report = {
        "overall_metrics": {
            "average_wer": avg_wer,
            "wer_std": wer_std,
            "quality_accuracy": quality_accuracy,
            "quality_precision_weighted": quality_precision,
            "quality_recall_weighted": quality_recall,
            "quality_f1_weighted": quality_f1,
            "total_valid_samples": len(valid_references)
        },
        "wer_distribution": wer_distribution,
        "quality_classification_report": quality_classification_rep,
        "sample_statistics": {
            "total_samples": len(references),
            "valid_samples": len(valid_references),
            "invalid_samples": len(references) - len(valid_references),
            "avg_reference_length": sum(len(ref.split()) for ref in valid_references) / len(valid_references) if valid_references else 0,
            "avg_hypothesis_length": sum(len(hyp.split()) for hyp in valid_hypotheses) / len(valid_hypotheses) if valid_hypotheses else 0
        }
    }
    
    # Speaker-level analysis
    if speaker_ids and len(speaker_ids) == len(references):
        speaker_analysis = defaultdict(lambda: {"references": [], "hypotheses": [], "wers": []})
        
        for i in valid_indices:
            if i < len(speaker_ids):
                speaker_id = speaker_ids[i]
                valid_idx = valid_indices.index(i)
                speaker_analysis[speaker_id]["references"].append(valid_references[valid_idx])
                speaker_analysis[speaker_id]["hypotheses"].append(valid_hypotheses[valid_idx])
                speaker_analysis[speaker_id]["wers"].append(valid_wers[valid_idx])
        
        speaker_summaries = {}
        for speaker_id, data in speaker_analysis.items():
            if len(data["references"]) > 0:
                speaker_avg_wer = sum(data["wers"]) / len(data["wers"])
                speaker_wer_std = np.std(data["wers"]) if len(data["wers"]) > 1 else 0.0
                
                speaker_summaries[speaker_id] = {
                    "sample_count": len(data["references"]),
                    "average_wer": speaker_avg_wer,
                    "wer_std": speaker_wer_std,
                    "min_wer": min(data["wers"]),
                    "max_wer": max(data["wers"]),
                    "avg_ref_length": sum(len(ref.split()) for ref in data["references"]) / len(data["references"]),
                    "avg_hyp_length": sum(len(hyp.split()) for hyp in data["hypotheses"]) / len(data["hypotheses"])
                }
        
        evaluation_report["speaker_level_analysis"] = speaker_summaries
    
    return evaluation_report

def cuda_timing_inference(model, processor, inputs, max_new_tokens=256):
    """Inference with precise GPU timing using CUDA Event API"""
    
    torch.cuda.synchronize()
    
    event_start = torch.cuda.Event(enable_timing=True)
    event_prefill_end = torch.cuda.Event(enable_timing=True)
    event_total_end = torch.cuda.Event(enable_timing=True)
    
    try:
        event_start.record()
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False, 
                           output_hidden_states=False, return_dict=True)
        
        event_prefill_end.record()
        
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
        
        event_start.synchronize()
        event_prefill_end.synchronize()
        event_total_end.synchronize()
        
        prefill_time = event_start.elapsed_time(event_prefill_end) / 1000.0
        total_time = event_start.elapsed_time(event_total_end) / 1000.0
        decode_time = event_prefill_end.elapsed_time(event_total_end) / 1000.0
        
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

def process_librispeech_results(item, response, timing_info):
    """Process LibriSpeech result"""
    gt = item["transcription"].strip()
    pred = clean_response(response)
    
    result = {
        "id": item["id"],
        "gt": gt,
        "pred": pred,
        "source": item["path"],
        "timing": timing_info,
        "gt_standardized": standardize_text(gt),
        "pred_standardized": standardize_text(pred)
    }
    
    return result

def main():
    random.seed(42)

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
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
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]
    
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")
    if debug_mode:
        print("Debug mode enabled - detailed output will be shown")

    librispeech_path = os.environ.get("LIBRISPEECH_PATH", "/data/to/your/librispeech-long/path")
    result_dir = os.environ.get("RESULTS_DIR", "/data/to/your/results/path")
    
    librispeech_path = os.path.abspath(librispeech_path)
    result_dir = os.path.abspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    output_file = f'{result_dir}/LibriSpeech_Aero1_DART_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    timing_output_file = f'{result_dir}/LibriSpeech_Aero1_DART_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    cuda_event_output_file = f'{result_dir}/LibriSpeech_Aero1_DART_cuda_event_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    print(f"CUDA Event statistics will be saved to: {cuda_event_output_file}")

    _AUDIO_SPECIAL_TOKEN_ID = 151667

    timing_stats = LibriSpeechTimingStats()
    cuda_event_stats = CudaEventTimingStats()

    print(f"\n=== LibriSpeech ASR Evaluation Config (Aero-1 + DART) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Prune layer index: {prune_layer_idx}")
    print(f"Prune ratio: {prune_ratio}")
    print(f"Prune method: {method_is}")
    print(f"Data path: {librispeech_path}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda"
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU available, using CPU")
    
    print(f"Loading LibriSpeech dataset: {librispeech_path}")
    dataset = load