import os
import json
from tqdm import tqdm
import torch
import numpy as np
import soundfile as sf
import warnings
import traceback
import time
import glob
import random
import sys
import io
import re
import librosa
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

random.seed(42)

sys.path.append("/data/to/your/Qwen_2.5_Code/path/")
from modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from processing_qwen2_5_omni import(
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info

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

_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648

gpu_temp=os.environ.get("CUDA_VISIBLE_DEVICES")

gpu_id = gpu_temp[-1]
print(f"Using GPU ID: {gpu_id}")

prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 2))
prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
prune_method = os.environ.get("PRUNE_METHOD", "base")

use_random = (prune_method == "random")
use_frame = (prune_method == "frame")
if use_random==False and use_frame==False:
    prune_method = "fast_v"

if prune_ratio == 0:
    method_is = "base"
else:
    method_is = prune_method

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

data_path_root = '/data/to/your/dataset/path//HAD/concatenated_audio'
result_dir = './HAD_Results'
os.makedirs(result_dir, exist_ok=True)

output_file = f'{result_dir}/HAD_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/HAD_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing stats will be saved to: {timing_output_file}")

class GlobalTimingStats:
    """Global timing statistics class for the first 100 samples (excluding the first one)"""
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_time = 0.0
        self.total_input_tokens = 0
        self.total_audio_tokens = 0
        self.timing_records = []
        self.max_samples = 100
    
    def add_record(self, prefill_time, total_time, output_tokens, input_tokens, audio_tokens, sample_index):
        if sample_index == 0:
            return
        
        if self.samples >= self.max_samples:
            return
            
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_time += total_time
        self.total_input_tokens += input_tokens
        self.total_audio_tokens += audio_tokens
        
        self.timing_records.append({
            "sample_index": sample_index,
            "prefill_time": prefill_time,
            "total_time": total_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "audio_tokens": audio_tokens
        })
    
    def get_summary(self):
        """Get statistics summary"""
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_total_time": 0,
                "avg_input_tokens": 0,
                "avg_audio_tokens": 0
            }
        
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_total_time": self.total_time / self.samples,
            "avg_input_tokens": self.total_input_tokens / self.samples,
            "avg_audio_tokens": self.total_audio_tokens / self.samples
        }
    
    def export_to_json(self, output_file):
        """Export statistics to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def load_audio_for_had(audio_path, audio_cache=None, target_sr=16000):
    """
    Load audio file, return format consistent with Qwen2.5-Omni
    Reference GTZAN audio loading method
    """
    if audio_cache is not None and audio_path in audio_cache:
        audio_np, sr = audio_cache[audio_path]
    else:
        try:
            audio_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio_np.shape}, sample_rate={sr}Hz")
        except Exception as e:
            print(f"Failed to load with librosa: {e}")
            
            try:
                audio_np, sample_rate = sf.read(audio_path)
                
                if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
                    audio_np = np.mean(audio_np, axis=1)
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio_np = signal.resample(audio_np, int(len(audio_np) * target_sr / sample_rate))
                    
                audio_np = audio_np.astype(np.float32)
                sr = target_sr
                print(f"Successfully processed with soundfile: shape={audio_np.shape}, sample_rate={sr}Hz")
                
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                audio_np = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generated silent audio as fallback")
        
        audio_np = audio_np.astype(np.float32)
        
        if audio_cache is not None:
            audio_cache[audio_path] = (audio_np, sr)
    
    return audio_np, sr

def load_had_dataset(root_dir):
    """Load HAD dataset with balanced real and fake samples"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    
    all_samples = []
    
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "real",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'.",
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
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'.",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "fake",
                "task": "Audio_Authenticity_Detection"
            })
    
    print(f"Total loaded {len(all_samples)} audio samples")
    
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
    """Extract audio authenticity answer from model output text - consistent with HAD_test.py"""
    text_lower = text.lower().strip()
    
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)') or text_lower.endswith(' a'):
        return choice_a_lower
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)') or text_lower.endswith(' b'):
        return choice_b_lower
        
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower or " a " in text_lower:
        return choice_a_lower
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower or " b " in text_lower:
        return choice_b_lower
    
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a_lower
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b_lower
    
    if choice_a_lower == "real" and choice_b_lower == "fake":
        real_match = re.search(r'\breal\b|\bauthentic\b|\bgenuine\b|\bcompletely authentic\b', text_lower) is not None
        fake_match = re.search(r'\bfake\b|\bartificial\b|\bsynthetic\b|\bsynthesized\b|\bcontains.*fake\b', text_lower) is not None
        
        if real_match and not fake_match:
            return "real"
        if fake_match and not real_match:
            return "fake"
    
    print(f"DEBUG: Unable to extract answer, original text: '{text}'")
    return ""

def calculate_had_metrics(y_true, y_pred):
    """
    Calculate detailed evaluation metrics for HAD audio authenticity detection
    
    Args:
        y_true: Ground truth label list (real/fake)
        y_pred: Predicted label list (real/fake) 
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in ['real', 'fake'] and pred_label in ['real', 'fake']:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'precision_fake': 0.0,
            'recall_fake': 0.0,
            'f1_fake': 0.0,
            'precision_real': 0.0,
            'recall_real': 0.0,
            'f1_real': 0.0,
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true)
        }
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    labels = ['fake', 'real']
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=labels, average=None, zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='weighted', zero_division=0
    )
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=labels,
        target_names=['Fake Audio', 'Real Audio'],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'precision_fake': float(precision[0]),
        'recall_fake': float(recall[0]),
        'f1_fake': float(f1[0]),
        'precision_real': float(precision[1]),
        'recall_real': float(recall[1]),
        'f1_real': float(f1[1]),
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true)
    }

def main():
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
    
    if hasattr(model, 'config.text_config') and hasattr(model.config.text_config, 'config'):
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
        print(f"Initialized config.text_config pruning parameters")

    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    timing_stats = GlobalTimingStats()
    
    samples = load_had_dataset(data_path_root)
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applied sample limit, will process {len(samples)} samples")
    
    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)
    
    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"Category statistics: real samples={real_count}, fake samples={fake_count}")
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
        }
    }
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    with tqdm(total=len(samples), desc="Processing HAD audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            label = item['label']
            task = item.get('task', 'Audio_Authenticity_Detection')
            
            ground_truth = item["answer_gt"].lower().strip()
            output_text = ""
            predicted_label = ""
            is_correct = False
            prefill_time = 0
            total_time = 0
            output_tokens = 0
            input_token_length = 0
            audio_token_length = 0
            
            try:
                audio_path_for_inference = audio_path
              
                instruction = "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'."
                sys_prompt = "You are a helpful assistant that analyzes audio to determine authenticity."
                
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path_for_inference},
                        {"type": "text", "text": instruction},
                    ]}
                ]
                
                print(f"DEBUG: Preparing to process audio, audio path: {audio_path}")
                
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
                        print(f"DEBUG: Set audio pruning parameters to config.text_config: layer_idx={prune_layer_idx}, ratio={prune_ratio}, tokens={audio_token_length}, start={audio_token_start}")

                
                prefill_start_event = torch.cuda.Event(enable_timing=True)
                prefill_end_event = torch.cuda.Event(enable_timing=True)
                
                prefill_start_event.record()
                with torch.no_grad():
                    prefill_output = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        thinker_max_new_tokens=1,
                        thinker_do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                prefill_end_event.record()
                
                total_start_event = torch.cuda.Event(enable_timing=True)
                total_end_event = torch.cuda.Event(enable_timing=True)

                total_start_event.record()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        thinker_max_new_tokens=5,  
                        thinker_do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                total_end_event.record()
                
                torch.cuda.synchronize()
                prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
                total_time = total_start_event.elapsed_time(total_end_event) / 1000.0
                
                output_text = processor.batch_decode(
                    output, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                if "assistant\n" in output_text:
                    assistant_start = output_text.rfind("assistant\n") + len("assistant\n")
                    output_text = output_text[assistant_start:].strip()
                
                if hasattr(output, 'shape') and len(output.shape) > 1:
                    output_tokens = output.shape[1] - inputs["input_ids"].shape[1]
                else:
                    output_tokens = 0
                
                output_text = output_text.strip()
                
                if output_tokens > 10:
                    words = output_text.lower().split()
                    for word in words:
                        if word in ['real', 'fake', 'authentic', 'synthesized']:
                            if word in ['authentic', 'synthesized']:
                                output_text = 'real' if word == 'authentic' else 'fake'
                            else:
                                output_text = word
                            break
                
                predicted_label = extract_authenticity_answer(output_text)
                
                is_correct = predicted_label == ground_truth
                
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                output_text = "ERROR"
                predicted_label = "error"
                is_correct = False
                prefill_time = 0
                total_time = 0
                output_tokens = 0
                input_token_length = 0
                audio_token_length = 0
                
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            results["summary"]["total_samples"] += 1
            if ground_truth in ["real", "fake"]:
                results["summary"][f"{ground_truth}_total"] += 1
                if is_correct:
                    results["summary"][f"{ground_truth}_correct"] += 1
                    results["summary"]["correct_samples"] += 1
            
            timing_stats.add_record(prefill_time, total_time, output_tokens, input_token_length, audio_token_length, i)
            
            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "audio_label": label,
                "ground_truth": ground_truth,
                "model_output": output_text,
                "extracted_answer": predicted_label,
                "is_correct": is_correct,
                "input_tokens": input_token_length,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "total_time": total_time
            }
            
            results["samples"].append(sample_result)
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                
                pbar.set_postfix_str(
                    f"Accuracy:{current_accuracy:.2%}"
                )
                
                if is_screen_env:
                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.2%}")
            
            pbar.update()
    
    final_stats = timing_stats.get_summary()
    
    total_samples = len(results["samples"])
    correct_samples = sum(1 for result in results["samples"] if result['is_correct'])
    
    results["summary"]["accuracy"] = correct_samples / total_samples if total_samples > 0 else 0
    results["summary"]["real_accuracy"] = results["summary"]["real_correct"] / results["summary"]["real_total"] if results["summary"]["real_total"] > 0 else 0
    results["summary"]["fake_accuracy"] = results["summary"]["fake_correct"] / results["summary"]["fake_total"] if results["summary"]["fake_total"] > 0 else 0
    
    results["summary"]["timing"] = final_stats
    
    y_true = [sample["ground_truth"] for sample in results["samples"]]
    y_pred = [sample["extracted_answer"] for sample in results["samples"]]
    
    detailed_metrics = calculate_had_metrics(y_true, y_pred)
    
    results["summary"]["sklearn_metrics"] = detailed_metrics
    
    tp = results["summary"]["fake_correct"]
    fp = results["summary"]["real_total"] - results["summary"]["real_correct"]
    fn = results["summary"]["fake_total"] - results["summary"]["fake_correct"]
    tn = results["summary"]["real_correct"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results["summary"]["precision"] = precision
    results["summary"]["recall"] = recall
    results["summary"]["f1_score"] = f1_score
    
    json_output_file = f'{result_dir}/HAD_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    
    results = convert_numpy_types(results)
    
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n=== HAD Audio Authenticity Detection Evaluation Results Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Real audio accuracy: {results['summary']['real_accuracy']:.2%} ({results['summary']['real_correct']}/{results['summary']['real_total']})")
    print(f"Fake audio accuracy: {results['summary']['fake_accuracy']:.2%} ({results['summary']['fake_correct']}/{results['summary']['fake_total']})")
    
    metrics = results["summary"]["sklearn_metrics"]
    print(f"\n=== Detailed Evaluation Metrics (sklearn) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (macro avg): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted avg): {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro avg): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro avg): {metrics['recall_macro']:.4f}")
    
    print(f"\n=== Evaluation Metrics by Category ===")
    print(f"Fake Audio - Precision: {metrics['precision_fake']:.4f}, Recall: {metrics['recall_fake']:.4f}, F1: {metrics['f1_fake']:.4f}")
    print(f"Real Audio - Precision: {metrics['precision_real']:.4f}, Recall: {metrics['recall_real']:.4f}, F1: {metrics['f1_real']:.4f}")
    
    print(f"\n=== Traditional Evaluation Metrics (manual calculation) ===")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1_score:.2%}")
    
    timing_sample_count = final_stats["samples"]
    print(f"\n=== Timing Statistics (first 100 samples, excluding the first one) ===")
    print(f"Statistical sample count: {timing_sample_count}")
    print(f"Average total time: {final_stats['avg_total_time']:.4f}s")
    print(f"Average prefill time: {final_stats['avg_prefill_time']:.4f}s")
    print(f"Average input tokens: {final_stats['avg_input_tokens']:.1f}")
    print(f"Average audio tokens: {final_stats['avg_audio_tokens']:.1f}")
    
    print(f"\n=== Detailed Classification Report ===")
    print(metrics['classification_report'])
    
    print(f"Results saved to: {json_output_file}")

if __name__ == "__main__":
    main()