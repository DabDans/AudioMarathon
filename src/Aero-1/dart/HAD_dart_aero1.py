import argparse
import os
import sys
import warnings
import torch
import time
import json
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import gc
import traceback
import glob

# Disable warnings and set environment
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'

# Audio special token ID for Aero-1
_AUDIO_SPECIAL_TOKEN_ID = 151667

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
    parser.add_argument('--sparse', type=str_to_bool, default=False, help='Enable DART sparse mode')
    parser.add_argument('--pruned_layer', default=2, type=int, help='prune_layer')
    parser.add_argument('--audio_token_start_index', type=int, default=35, help='audio_token_start_index')
    parser.add_argument('--audio_token_length', type=int, default=576, help='audio_token_length')
    parser.add_argument('--reduction_ratio', type=float, default=0.778, help='retained_ratio for DART')
    parser.add_argument('--pivot_audio_token', type=int, default=4, help='pivot_audio_token')
    parser.add_argument('--pivot_text_token', type=int, default=4, help='pivot_text_token')
    parser.add_argument('--use_random', type=str_to_bool, default=False, help='Use random pruning')
    parser.add_argument('--use_frame', type=str_to_bool, default=False, help='Use frame pruning')
    return parser.parse_args()

def configure_DART(model, args):
    """Configure DART sparse attention mechanism for Aero-1"""
    if args.sparse:
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = args.reduction_ratio
        model.config.random = args.use_random
        model.config.frame = args.use_frame
        
        DART_config = {
            "K": args.pruned_layer,
            "audio_token_start_index": args.audio_token_start_index,
            "audio_token_length": args.audio_token_length,
            "reduction_ratio": args.reduction_ratio,
            "pivot_audio_token": args.pivot_audio_token,
            "pivot_text_token": args.pivot_text_token,
            "text_length": 1,
        }
        model.config.DART_config = DART_config
        print(f"DART sparse attention enabled: layer={args.pruned_layer}, ratio={args.reduction_ratio}")
    else:
        model.config.DART_config = None
        model.config.audio_layer_idx = None
        print("Using base model (no DART optimization)")

# Get GPU ID and configuration
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]

print(f"Using GPU ID: {gpu_id}")
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")
if debug_mode:
    print("Debug mode enabled")

# Data path configuration
data_path_root = '/data/to/your/had_audio_data'
result_dir = '/data/to/your/had_results'
os.makedirs(result_dir, exist_ok=True)

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

class HADTimingStats:
    """Track HAD task inference timing stats, using CUDA Event for accurate measurement"""
    def __init__(self):
        self.timing_records = []
        self.cuda_available = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   audio_duration=None, label=None):
        """Add a timing record"""
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decode_tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "audio_duration": audio_duration,
            "label": label
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get overall timing summary"""
        if not self.timing_records:
            return {"error": "No timing records available"}
        
        df = pd.DataFrame(self.timing_records)
        summary = {
            "total_samples": len(df),
            "avg_total_time": df["total_time"].mean(),
            "avg_prefill_time": df["prefill_time"].mean(),
            "avg_decode_time": df["decode_time"].mean(),
            "avg_decode_tokens_per_sec": df["decode_tokens_per_sec"].mean(),
            "prefill_percentage": (df["prefill_time"].sum() / df["total_time"].sum()) * 100,
            "decode_percentage": (df["decode_time"].sum() / df["total_time"].sum()) * 100
        }
        
        return summary
    
    def export_to_json(self, output_file):
        """Export timing stats to JSON file"""
        summary = self.get_summary()
        data = {
            "summary": summary,
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def downsample_audio(audio_array, original_sr, target_sr):
    """Downsample audio to target sampling rate"""
    if original_sr == target_sr:
        return audio_array
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
    """Load audio using librosa and split, compatible with Aero-1 official example"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:
            audio_chunks = split_audio(audio)
            if debug_mode:
                print(f"Audio length {len(audio)} exceeds 30 seconds, split into {len(audio_chunks)} chunks")
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_had_dataset(root_dir):
    """Load HAD dataset, balance number of real/fake samples"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    all_samples = []
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "real",
                "id": f"real_{os.path.basename(wav_path)}",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "real",
                "task": "Audio_Authenticity_Detection"
            })
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for wav_path in fake_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "fake",
                "id": f"fake_{os.path.basename(wav_path)}",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "fake",
                "task": "Audio_Authenticity_Detection"
            })
    print(f"Total {len(all_samples)} audio samples loaded")
    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"Original sample count: real={len(real_samples)}, fake={len(fake_samples)}")
    min_samples_per_category = min(len(real_samples), len(fake_samples))
    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)
    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)
    balanced_samples = real_samples + fake_samples
    random.shuffle(balanced_samples)
    print(f"Balanced sample count: real={len(real_samples)}, fake={len(fake_samples)}, total={len(balanced_samples)}")
    return balanced_samples

def extract_authenticity_answer(text, choice_a="real", choice_b="fake"):
    """Extract authenticity answer from model output text"""
    text_lower = text.lower().strip()
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b
    if choice_a_lower == "real" and choice_b_lower == "fake":
        real_keywords = ["real", "authentic", "genuine", "original", "natural"]
        fake_keywords = ["fake", "synthetic", "artificial", "generated", "deepfake"]
        real_count = sum(1 for keyword in real_keywords if keyword in text_lower)
        fake_count = sum(1 for keyword in fake_keywords if keyword in text_lower)
        if real_count > fake_count:
            return "real"
        elif fake_count > real_count:
            return "fake"
    return ""

def calculate_had_metrics(predictions, ground_truths):
    """Calculate HAD task metrics like F1 score etc."""
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['real', 'fake'] and t in ['real', 'fake']]
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'valid_samples': 0,
            'total_samples': len(predictions)
        }
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    label_map = {'real': 0, 'fake': 1}
    y_true = [label_map[label] for label in valid_ground_truths]
    y_pred = [label_map[label] for label in valid_predictions]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_samples': len(valid_pairs),
        'total_samples': len(predictions)
    }

def generate_sklearn_had_dart_evaluation_report(y_true, y_pred, labels=None):
    """
    Generate detailed HAD audio authenticity detection evaluation report (DART version) using sklearn
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"error": "Invalid input data for evaluation"}
    valid_indices = []
    valid_y_true = []
    valid_y_pred = []
    valid_label_set = {'real', 'fake'}
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in valid_label_set and pred_label in valid_label_set:
            valid_indices.append(i)
            valid_y_true.append(true_label)
            valid_y_pred.append(pred_label)
    if not valid_y_true:
        return {"error": "No valid labels for evaluation"}
    accuracy = accuracy_score(valid_y_true, valid_y_pred)
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='binary', pos_label='fake', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average=None, labels=['real', 'fake'], zero_division=0
    )
    if labels is None:
        target_names = ['real', 'fake']
    else:
        target_names = labels
    classification_rep = classification_report(
        valid_y_true, valid_y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    tp = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'fake' and p == 'fake')
    fp = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'real' and p == 'fake')
    fn = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'fake' and p == 'real')
    tn = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'real' and p == 'real')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    evaluation_report = {
        "overall_metrics": {
            "accuracy": accuracy,
            "precision_binary": precision_binary,
            "recall_binary": recall_binary,
            "f1_binary": f1_binary,
            "specificity": specificity,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted
        },
        "per_class_metrics": {},
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn
        },
        "classification_report": classification_rep,
        "sample_statistics": {
            "total_samples": len(y_true),
            "valid_samples": len(valid_y_true),
            "invalid_samples": len(y_true) - len(valid_y_true),
            "correct_predictions": sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == p),
            "real_samples": sum(1 for label in valid_y_true if label == 'real'),
            "fake_samples": sum(1 for label in valid_y_true if label == 'fake')
        }
    }
    class_labels = ['real', 'fake']
    for i, class_label in enumerate(class_labels):
        if i < len(precision_per_class):
            evaluation_report["per_class_metrics"][class_label] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1_score": f1_per_class[i],
                "support": int(support_per_class[i]) if i < len(support_per_class) else 0
            }
    return evaluation_report

def cuda_timing_inference(model, processor, inputs, max_new_tokens=10):
    """
    Inference with accurate GPU timing using CUDA Event API, optimized for Aero-1
    """
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

def create_had_prompt_aero1(sample):
    """Create Aero-1 format prompt for HAD task"""
    return "Is this audio real or fake? Answer with 'real' or 'fake' only."

def main():
    random.seed(42)
    args = parse_arguments()
    print(f"\n=== HAD Aero-1+DART Evaluation Configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Model: {args.model_path}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Prune layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Data directory: {data_path_root}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 40)

    if args.sparse:
        if args.use_random:
            method_name = "random"
        elif args.use_frame:
            method_name = "frame"
        else:
            method_name = "fast_v"
        method_suffix = f"_{method_name}_prune{args.reduction_ratio}"
    else:
        method_suffix = "_base"

    output_file = f'{result_dir}/HAD_Aero1_DART_results_gpu{gpu_id}{method_suffix}.json'
    timing_output_file = f'{result_dir}/HAD_Aero1_DART_timing_stats_gpu{gpu_id}{method_suffix}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")
    
    timing_stats = HADTimingStats()
    print("Loading Aero-1 model...")
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision="main",
        trust_remote_code=True
    )
    print("Aero-1 processor loaded successfully")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision="main",
        device_map="cuda",
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    print("Aero-1 model loaded successfully")
    configure_DART(model, args)
    samples = load_had_dataset(data_path_root)
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Sample count limited to: {len(samples)}")
    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)
    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"Category statistics: real samples={real_count}, fake samples={fake_count}")
    all_predictions = []
    all_ground_truths = []
    results = {
        "model_name": "Aero-1-Audio-1.5B",
        "dart_config": {
            "enabled": args.sparse,
            "pruned_layer": args.pruned_layer,
            "reduction_ratio": args.reduction_ratio,
            "use_random": args.use_random,
            "use_frame": args.use_frame,
            "method": method_name if args.sparse else "base"
        },
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
            "metrics": {},
            "timing": {}
        }
    }
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    with tqdm(total=len(samples), desc="Processing HAD audio samples (Aero-1+DART)", position=0, leave=True, **tqdm_kwargs) as pbar:
        for idx, sample in enumerate(samples):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                audio_chunks, sample_rate = prepare_audio_for_processor(sample["audio_path"])
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",
                    })
                text_prompt = create_had_prompt_aero1(sample)
                messages[0]["content"].append({
                    "type": "text",
                    "text": text_prompt
                })
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    text=prompt,
                    audios=audio_chunks,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to("cuda")
                audio_token_length = 0
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    if args.sparse:
                        model.config.audio_token_num = audio_token_length
                        model.config.audio_token_start = audio_token_start
                        if hasattr(model.config, 'DART_config') and model.config.DART_config:
                            model.config.DART_config['audio_token_start_index'] = audio_token_start
                            model.config.DART_config['audio_token_length'] = audio_token_length
                if debug_mode:
                    print(f"Processing audio: {os.path.basename(sample['audio_path'])}")
                    print(f"Audio chunks: {len(audio_chunks)}")
                    print(f"Audio token length: {audio_token_length}")
                    sys.stdout.flush()
                result = cuda_timing_inference(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=10
                )
                output = result['response_text']
                prefill_time = result['prefill_time']
                decode_time = result['decode_time']
                total_time = result['total_time']
                output_tokens = result['output_tokens']
                input_tokens = inputs['input_ids'].shape[1]
                predicted_label = extract_authenticity_answer(output)
                true_label = sample["label"]
                is_correct = predicted_label == true_label
                all_predictions.append(predicted_label)
                all_ground_truths.append(true_label)
                sample_result = {
                    "id": sample["id"],
                    "audio_path": sample["audio_path"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "response": output,
                    "is_correct": is_correct,
                    "audio_chunks": len(audio_chunks),
                    "audio_tokens": audio_token_length,
                    "timing": {
                        "prefill_time": prefill_time,
                        "decode_time": decode_time,
                        "total_time": total_time,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_sec": output_tokens/decode_time if decode_time > 0 else 0
                    }
                }
                results["samples"].append(sample_result)
                if idx > 0 and idx <= 100:
                    timing_stats.add_record(
                        prefill_time,
                        decode_time,
                        output_tokens,
                        input_tokens,
                        label=true_label
                    )
                results["summary"]["total_samples"] += 1
                if is_correct:
                    results["summary"]["correct_samples"] += 1
                if true_label == "real":
                    results["summary"]["real_total"] += 1
                    if is_correct:
                        results["summary"]["real_correct"] += 1
                else:
                    results["summary"]["fake_total"] += 1
                    if is_correct:
                        results["summary"]["fake_correct"] += 1
                pbar.set_postfix({
                    'Acc': f"{results['summary']['correct_samples']/results['summary']['total_samples']:.3f}",
                    'Tokens/s': f"{output_tokens/decode_time:.1f}" if decode_time > 0 else "N/A"
                })
                del inputs
                del audio_chunks
                del result
                torch.cuda.empty_cache()
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                pbar.update(1)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                if debug_mode:
                    traceback.print_exc()
                all_predictions.append("")
                all_ground_truths.append(sample.get("label", "unknown"))
                pbar.update(1)
                continue
    if len(all_predictions) > 0:
        metrics = calculate_had_metrics(all_predictions, all_ground_truths)
        results["summary"]["metrics"] = metrics
        print(f"\n=== HAD Evaluation Results (Aero-1+DART) ===")
        print(f"Model: {args.model_path}")
        print(f"DART config: {results['dart_config']}")
        print(f"Total accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn HAD DART evaluation report ===")
        print(f"Total samples: {len(all_predictions)} (predictions), {len(all_ground_truths)} (ground truths)")
        sklearn_evaluation = generate_sklearn_had_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            labels=['real', 'fake']
        )
        print(f"sklearn HAD DART evaluation report generated:")
        print(f"  Overall accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Precision (binary): {sklearn_evaluation['overall_metrics']['precision_binary']:.4f}")
        print(f"  Recall (binary): {sklearn_evaluation['overall_metrics']['recall_binary']:.4f}")
        print(f"  F1 Score (binary): {sklearn_evaluation['overall_metrics']['f1_binary']:.4f}")
        print(f"  Specificity: {sklearn_evaluation['overall_metrics']['specificity']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
        results["summary"]["sklearn_evaluation"] = sklearn_evaluation
    else:
        print("Warning: No valid predictions, cannot generate sklearn evaluation report")
        results["summary"]["sklearn_evaluation"] = {"error": "No valid predictions for evaluation"}
    timing_summary = timing_stats.get_summary()
    results["summary"]["timing"] = timing_summary
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing stats saved to: {timing_output_file}")
    if "sklearn_evaluation" in results["summary"] and "error" not in results["summary"]["sklearn_evaluation"]:
        sklearn_metrics = results["summary"]["sklearn_evaluation"]["overall_metrics"]
        confusion_matrix = results["summary"]["sklearn_evaluation"]["confusion_matrix"]
        print(f"\n=== Sklearn Evaluation Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision (Binary): {sklearn_metrics['precision_binary']:.4f}")
        print(f"Recall/Sensitivity (Binary): {sklearn_metrics['recall_binary']:.4f}")
        print(f"Specificity: {sklearn_metrics['specificity']:.4f}")
        print(f"F1 Score (Binary): {sklearn_metrics['f1_binary']:.4f}")
        print(f"F1 Score (Macro): {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {sklearn_metrics['f1_weighted']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positive (TP): {confusion_matrix['true_positive']} (correct fake detection)")
        print(f"  False Positive (FP): {confusion_matrix['false_positive']} (incorrectly detected real as fake)")
        print(f"  False Negative (FN): {confusion_matrix['false_negative']} (incorrectly detected fake as real)")
        print(f"  True Negative (TN): {confusion_matrix['true_negative']} (correct real detection)")
        print(f"\nDetailed metrics per class:")
        per_class_metrics = results["summary"]["sklearn_evaluation"]["per_class_metrics"]
        for class_label in sorted(per_class_metrics.keys()):
            metrics_detail = per_class_metrics[class_label]
            print(f"  {class_label}: Precision={metrics_detail['precision']:.4f}, "
                  f"Recall={metrics_detail['recall']:.4f}, F1={metrics_detail['f1_score']:.4f}, "
                  f"Support={metrics_detail['support']}")

if __name__ == "__main__":
    main()