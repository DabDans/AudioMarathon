

import os
import sys
import json
import time
import random
import warnings
import gc
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any, Tuple
random.seed(42)
try:
    import torch
    import soundfile as sf
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from transformers import logging
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please run: pip install torch transformers soundfile numpy scikit-learn")
    sys.exit(1)

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")
random.seed(42)
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

data_path = os.environ.get("VESUS_DATA_PATH", 
    "/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/VESUS")
emotion_json_file = os.path.join(data_path, "audio_emotion_dataset.json")
result_dir = os.environ.get("RESULTS_DIR", './VESUS_Results')
os.makedirs(result_dir, exist_ok=True)

output_file = f'{result_dir}/VESUS_results_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
timing_output_file = f'{result_dir}/VESUS_timing_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing statistics will be saved to: {timing_output_file}")

_AUDIO_SPECIAL_TOKEN_ID = 200011

def calculate_emotion_metrics(predictions, ground_truths, emotion_labels):
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in emotion_labels and t in emotion_labels]
    
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
    
    label_map = {label: idx for idx, label in enumerate(sorted(emotion_labels))}
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
        'total_samples': len(predictions),
        'label_mapping': label_map
    }

class VESUSTimingStats:
    def __init__(self):
        self.timing_records = []
        self.emotion_stats = defaultdict(list)
        self.person_stats = defaultdict(list)
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
        self.use_cuda_events = torch.cuda.is_available()
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, 
                   emotion_label=None, person_id=None):
        self.total_samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        record = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
            "emotion_label": emotion_label,
            "person_id": person_id
        }
        
        self.timing_records.append(record)
        
        if emotion_label:
            self.emotion_stats[emotion_label].append(record)
        
        if person_id:
            self.person_stats[person_id].append(record)
    
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
            "avg_tokens_per_sec": avg_tokens_per_sec
        }
        
        emotion_summaries = {}
        for emotion, records in self.emotion_stats.items():
            if len(records) > 0:
                emotion_summaries[emotion] = {
                    "samples": len(records),
                    "avg_prefill_time": sum(r["prefill_time"] for r in records) / len(records),
                    "avg_decode_time": sum(r["decode_time"] for r in records) / len(records),
                    "avg_total_time": sum(r["total_time"] for r in records) / len(records),
                    "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in records) / len(records)
                }
        
        return {
            "overall_summary": summary,
            "emotion_summaries": emotion_summaries
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def load_vesus_dataset(root_dir):
    meta_file = os.path.join(root_dir, "audio_emotion_dataset.json")
    
    all_samples = []
    
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"From{meta_file}loaded{len(metadata)} samples metadata")
        
        for item in metadata:
            rel_path = item["path"]
            wav_path = os.path.join(root_dir, rel_path)
            
            if not os.path.exists(wav_path):
                print(f"Warning: file does not exist {wav_path}")
                continue
            
            emotion_label = item["emotion_label"]
            answer_gt = item["answer_gt"].strip()
            person_id = item["person_id"]
            
            if emotion_label == "happy":
                if person_id == "2" or person_id == "10":
                    print(f"Skip person{person_id} 's Happy emotion data: {rel_path}")
                    continue
            
            all_samples.append({
                "emotion": emotion_label,
                "audio_path": wav_path,
                "question": item["question"],
                "choices": [item["choice_a"], item["choice_b"], item["choice_c"], item["choice_d"]],
                "answer_gt": answer_gt,
                "task": "Speech_Emotion_Recognition",
                "person_id": person_id
            })
    else:
        print(f"Warning:  metadata file does not exist {meta_file}")
        return []
    
    print(f"Totalloaded {len(all_samples)} validaudio sample")
    
    emotion_counts = {}
    for sample in all_samples:
        group = sample["emotion"]
        emotion_counts[group] = emotion_counts.get(group, 0) + 1
    
    print("Emotion distribution:")
    for group, count in emotion_counts.items():
        print(f"  {group}: {count} samples")
    
    random.shuffle(all_samples)
    
    return all_samples

def prepare_audio_for_processor(audio_path, target_sr=16000):
    try:
        full_audio_path = os.path.join(data_path, audio_path)
        
        if not os.path.exists(full_audio_path):
            print(f"audiofile does not exist: {full_audio_path}")
            return None
        
        audio_data, sample_rate = sf.read(full_audio_path)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        if sample_rate != target_sr:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            except ImportError:
                print("Warning: librosa not installed，Skipresampling")
                if sample_rate != target_sr:
                    try:
                        import scipy.signal
                        audio_data = scipy.signal.resample(audio_data, 
                                                         int(len(audio_data) * target_sr / sample_rate))
                    except ImportError:
                        print("Warning: scipyalso not installed, keeping original sample rate")
        
        return [(audio_data, target_sr)]
        
    except Exception as e:
        print(f"processedaudiofileerror occurred {audio_path}: {e}")
        return None

def extract_emotion_answer(text, choices):
    text_lower = text.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return "A"
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return "B"
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return "C"
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return "D"
    
    option_patterns = {
        'A': ["option a", "choice a", "a)", "(a)"],
        'B': ["option b", "choice b", "b)", "(b)"],
        'C': ["option c", "choice c", "c)", "(c)"],
        'D': ["option d", "choice d", "d)", "(d)"]
    }
    
    for option, patterns in option_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return option
    
    emotion_keywords = {
        'angry': ['anger', 'frustrated', 'mad', 'furious'],
        'happy': ['joy', 'cheerful', 'pleased', 'delighted'],
        'sad': ['sadness', 'melancholy', 'depressed', 'sorrow'],
        'fearful': ['fear', 'anxiety', 'scared', 'afraid'],
        'monotone': ['flat', 'emotionless', 'neutral', 'bland']
    }
    
    for choice_key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
        if choice_key in choices:
            choice_text = choices[choice_key].lower()
            for emotion, keywords in emotion_keywords.items():
                if emotion in choice_text or any(keyword in choice_text for keyword in keywords):
                    if any(keyword in text_lower for keyword in keywords) or emotion in text_lower:
                        return choice_key[-1].upper()
    
    return ""

def create_emotion_prompt(sample):
    question = sample.get("question", "What emotion is expressed in this audio segment?")
    choice_a = sample.get("choice_a", "")
    choice_b = sample.get("choice_b", "")
    choice_c = sample.get("choice_c", "")
    choice_d = sample.get("choice_d", "")
    
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    prompt = f"{user_prompt}\n<audio>\n{question}\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n{assistant_prompt}\n"
    
    return prompt

def main():
    print(f"\n=== VESUSemotion recognition evaluation configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {data_path}")
    print(f"JSONfile: {emotion_json_file}")
    if sample_limit > 0:
        print(f" samplelimit: {sample_limit}")
    print("=" * 40)

    print("LoadPhi-4-multimodal-instructmodel...")
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_vesus_dataset(emotion_json_file)
    
    if not samples:
        print("Error: cannot find any data samples")
        return
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Apply sample limit，processed {len(samples)}  samples")

    timing_stats = VESUSTimingStats()

    results = []
    total_correct = 0
    emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    person_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    print(f"Start evaluation {len(samples)}  samples...")
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="VESUSevaluation")

    for idx, sample in progress_bar:
        try:
            audio_path = sample.get("path", "")
            audio = prepare_audio_for_processor(audio_path)
            
            if audio is None:
                continue
            
            emotion_label = sample.get("emotion_label", "unknown")
            person_id = sample.get("person_id", "unknown")
            answer_gt = sample.get("answer_gt", "").upper()
            
            prompt = create_emotion_prompt(sample)

            inputs = processor(
                text=prompt,
                audios=audio,
                return_tensors="pt",
            ).to(device)
            inputs['input_mode'] = torch.tensor([2])

            audio_token_length = 0
            if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                token_ids = inputs.input_ids[0].tolist()
                audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                rev_ids = token_ids[::-1]
                audio_token_end = len(token_ids) - 1 - rev_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                audio_token_length = audio_token_end - audio_token_start + 1

                model.config.image_layer_idx = None
                model.config.audio_layer_idx = prune_layer_idx
                model.config.audio_token_num = audio_token_length
                model.config.audio_token_start = audio_token_start
                model.config.audio_prune_ratio = prune_ratio
                model.config.random = use_random
                model.config.frame = use_frame
                if use_random:
                    model.config.output_attentions = False
                if use_frame:
                    model.config.output_attentions = False

            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
            prefill_end_event.record()

            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
            
            decode_start_event.record()
            out_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                generation_config=generation_config,
                do_sample=False,
                use_cache=True
            )
            decode_end_event.record()
            
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0

            resp = processor.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            predicted_answer = extract_emotion_answer(resp, sample)
            is_correct = (predicted_answer == answer_gt)

            if is_correct:
                total_correct += 1
            
            emotion_stats[emotion_label]["total"] += 1
            person_stats[person_id]["total"] += 1
            
            if is_correct:
                emotion_stats[emotion_label]["correct"] += 1
                person_stats[person_id]["correct"] += 1

            current_accuracy = total_correct / (idx + 1)
            progress_bar.set_postfix({
                'Acc': f'{current_accuracy:.3f}',
                'Emotion': emotion_label[:8],
                'Person': person_id
            })

            results.append({
                "idx": idx,
                "path": audio_path,
                "emotion_label": emotion_label,
                "person_id": person_id,
                "question": sample.get("question", ""),
                "choices": {
                    "A": sample.get("choice_a", ""),
                    "B": sample.get("choice_b", ""),
                    "C": sample.get("choice_c", ""),
                    "D": sample.get("choice_d", "")
                },
                "answer_gt": answer_gt,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response_text": resp,
                "timing": {
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
            })

            timing_stats.add_record(
                prefill_time, decode_time, 
                out_ids.shape[1] - inputs["input_ids"].shape[1],
                inputs["input_ids"].shape[1],
                emotion_label, person_id
            )

            del inputs, outputs, out_ids
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"processed sample {idx} error occurred: {e}")
            continue

    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    all_predictions = [result["predicted_answer"] for result in results]
    all_ground_truths = [result["answer_gt"] for result in results]
    all_emotion_labels = list(set(all_ground_truths))
    
    emotion_metrics = calculate_emotion_metrics(all_predictions, all_ground_truths, all_emotion_labels)

    emotion_accuracies = {}
    for emotion, stats in emotion_stats.items():
        if stats["total"] > 0:
            emotion_accuracies[emotion] = stats["correct"] / stats["total"]

    person_accuracies = {}
    for person, stats in person_stats.items():
        if stats["total"] > 0:
            person_accuracies[person] = stats["correct"] / stats["total"]

    summary = {
        "total_samples": total_samples,
        "correct_samples": total_correct,
        "overall_accuracy": overall_accuracy,
        "metrics": emotion_metrics,
        "emotion_stats": dict(emotion_stats),
        "emotion_accuracies": emotion_accuracies,
        "person_stats": dict(person_stats),
        "person_accuracies": person_accuracies,
        "config": {
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "data_path": data_path,
            "json_file": emotion_json_file
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== VESUSemotion recognition evaluation result summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"F1 Score: {emotion_metrics['f1_score']:.4f}")
    print(f"Precision: {emotion_metrics['precision']:.4f}")
    print(f"Recall: {emotion_metrics['recall']:.4f}")
    print(f"valid sample: {emotion_metrics['valid_samples']}/{emotion_metrics['total_samples']}")
    print(f"Eachemotion accuracy:")
    for emotion, acc in emotion_accuracies.items():
        correct = emotion_stats[emotion]["correct"]
        total = emotion_stats[emotion]["total"]
        print(f"  {emotion}: {acc:.3f} ({correct}/{total})")
    
    timing_summary = timing_stats.get_summary()
    overall_summary = timing_summary.get("overall_summary", {})
    print(f"\n=== Timing statistics (CUDA Events precise measurement)===")
    print(f"Average inference time: {overall_summary.get('avg_total_time', 0):.4f} seconds")
    print(f"Average Prefill time: {overall_summary.get('avg_prefill_time', 0):.4f} seconds")
    print(f"Average Decode time: {overall_summary.get('avg_decode_time', 0):.4f} seconds")
    print(f"Average throughput: {overall_summary.get('avg_tokens_per_sec', 0):.2f} tokens/ seconds")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    main()
