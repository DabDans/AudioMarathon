import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import numpy as np
import soundfile as sf
import warnings
import traceback
import time
import glob
import random
import sys
import io
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

random.seed(42)

def calculate_metrics(predictions, ground_truths):
    valid_pairs = [(p, t) for p, t in zip(predictions, ground_truths) 
                   if p in ['male', 'female'] and t in ['male', 'female']]
    
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
    
    label_map = {'male': 0, 'female': 1}
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

class GlobalTimingStats:
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_tokens = 0
        self.timing_records = []
    
    def add_record(self, prefill_time, decode_time, output_tokens):
        self.samples += 1
        self.total_prefill_time += prefill_time
        self.total_decode_time += decode_time
        self.total_tokens += output_tokens
        
        self.timing_records.append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        })
    
    def get_summary(self):
        if self.samples == 0:
            return {
                "samples": 0,
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "avg_tokens_per_sec": 0
            }
        
        return {
            "samples": self.samples,
            "avg_prefill_time": self.total_prefill_time / self.samples,
            "avg_decode_time": self.total_decode_time / self.samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        }
    
    def export_to_json(self, output_file):
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def print_summary(self):
        summary = self.get_summary()
        print(f"\n=== Timing Statistics Summary ===")
        print(f"Valid samples: {summary['samples']}")
        print(f"Average Prefill time: {summary['avg_prefill_time']:.4f} seconds")
        print(f"Average Decode time: {summary['avg_decode_time']:.4f} seconds")
        print(f"Average total time: {summary['avg_total_time']:.4f} seconds")
        print(f"Average tokens/sec: {summary['avg_tokens_per_sec']:.2f}")

gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
print(f"Using GPU ID: {gpu_id}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
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

data_path_root = '/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/VoxCeleb/concatenated_audio'
result_dir = './Vox_Results'
os.makedirs(result_dir, exist_ok=True)

output_file = f'{result_dir}/VoxCeleb_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/VoxCeleb_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing statistics will be saved to: {timing_output_file}")

_AUDIO_SPECIAL_TOKEN_ID = 200011

def prepare_audio_for_processor(audio_path, target_sr=16000):
    
    try:
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"soundfile loading failed: {e}")
            
            try:
                import subprocess
                import tempfile
                from scipy.io import wavfile
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav = temp_file.name
                
                print(f"Convert using ffmpeg: {audio_path} -> {temp_wav}")
                subprocess.run([
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ar', str(target_sr), '-ac', '1',
                    temp_wav
                ], stderr=subprocess.DEVNULL)
                
                sample_rate, audio = wavfile.read(temp_wav)
                audio = audio.astype(np.float32)
                if audio.max() > 1.0:
                    audio = audio / 32768.0
                os.remove(temp_wav)
                print(f"ffmpegConversion successful: shape={audio.shape}, sample rate={sample_rate}Hz")
                
            except Exception as e:
                print(f"ffmpeg conversion failed: {e}")
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sample_rate = target_sr
                print("Generate silent replacement audio")
        
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
            print(f"Convert to mono: shape={audio.shape}")
        
        if sample_rate != target_sr and sample_rate > 0:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
            sample_rate = target_sr
            print(f"Resample to{target_sr}Hz:  new length={len(audio)}")
            
        if len(audio) == 0:
            print("Warning: Audio is empty，Create3 seconds silence")
            audio = np.zeros(target_sr * 3, dtype=np.float32)
        
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        return [(audio, sample_rate)]
        
    except Exception as e:
        print(f"Audio processed error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [(silence, target_sr)]

def read_text_file(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Failed to read text file: {e}")
        return ""


def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    meta_file = os.path.join(root_dir, "gender_id_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"From{meta_file}loaded{len(metadata)} samples metadata")
    
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(wav_path):
            print(f"Warning: file does not exist {wav_path}")
            continue
        
        speaker_id = item["speaker_id_original"]
        gender = item["answer_gt"].lower().strip()
        
        all_samples.append({
            "speaker_id": speaker_id,
            "gender": gender,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "answer_gt": gender,
            "task": "Speaker_Gender_Identification"
        })
    
    print(f"Totalloaded {len(all_samples)} validaudio sample")
    
    male_samples = [sample for sample in all_samples if sample["gender"].lower() == "male"]
    female_samples = [sample for sample in all_samples if sample["gender"].lower() == "female"]
    print(f"Original samples count: male={len(male_samples)}, female={len(female_samples)}")
    
    min_samples_per_gender = min(len(male_samples), len(female_samples))
    
    if sample_limit > 0:
        max_per_gender = sample_limit // 2
        min_samples_per_gender = min(min_samples_per_gender, max_per_gender)
        print(f"Apply sample limit: maximum {min_samples_per_gender} samples per gender")
    
    if len(male_samples) > min_samples_per_gender:
        male_samples = random.sample(male_samples, min_samples_per_gender)
    
    if len(female_samples) > min_samples_per_gender:
        female_samples = random.sample(female_samples, min_samples_per_gender)
    
    balanced_samples = male_samples + female_samples
    
    random.shuffle(balanced_samples)
    
    print(f"Final samples count: male={len(male_samples)}, female={len(female_samples)}, Total={len(balanced_samples)}")
    
    return balanced_samples

def extract_gender_answer(text, choice_a="male", choice_b="female"):
    text_lower = text.lower().strip()
    
    choice_a_lower = choice_a.lower().strip() 
    choice_b_lower = choice_b.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a_lower
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b_lower
        
    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a_lower
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b_lower
    
    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a_lower
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b_lower
    
    import re
    if choice_a_lower == "male" and choice_b_lower == "female":
        male_match = re.search(r'\bmale\b', text_lower) is not None
        female_match = re.search(r'\bfemale\b', text_lower) is not None
        
        if male_match and not female_match:
            return "male"
        if female_match and not male_match:
            return "female"
    
    return ""

def main():
    print("LoadPhi-4-multimodal-instructmodel...")
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="balanced_low_0",
        torch_dtype="auto",
        attn_implementation = "sdpa",
        trust_remote_code=True
    )
    model.eval()
    
    timing_stats = GlobalTimingStats()
    
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    
    male_count = sum(1 for s in samples if s["gender"].lower() == "male")
    female_count = sum(1 for s in samples if s["gender"].lower() == "female")
    print(f"Gender statistics: male samples={male_count}, female samples={female_count}")
    
    all_predictions = []
    all_ground_truths = []
    all_sample_results = []
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    print(f"Start processing {len(samples)} samples...")
    with tqdm(total=len(samples), desc="processedVoxCeleb sample", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        for i, sample in enumerate(samples):
            wav_path = sample['wav_path']
            speaker_id = sample["speaker_id"]
            ground_truth = sample["gender"].lower().strip()
            
            instruction = "Listen to this audio and identify the speaker's gender. Is this a male or female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'."
            full_prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
            
            try:
                audio = prepare_audio_for_processor(wav_path)
              
                inputs = processor(
                    text=full_prompt,
                    audios=audio,
                    return_tensors="pt"
                ).to("cuda")
                inputs['input_mode'] = torch.tensor([2])
                
                audio_token_length = 0
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    model.config.image_layer_idx = None
                    model.config.image_prune_ratio= None
                    model.config.audio_layer_idx = prune_layer_idx
                    model.config.audio_token_num = audio_token_length
                    model.config.audio_token_start = audio_token_start
                    model.config.audio_prune_ratio = prune_ratio
                    model.config.random = use_random
                    model.config.frame=use_frame
                
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
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        generation_config=generation_config,
                        return_dict_in_generate=True
                    )
                decode_end_event.record()
                
                torch.cuda.synchronize()
                prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
                decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0
                
                tokens = generate_ids.sequences[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                output = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                predicted_gender = extract_gender_answer(output)
                
                all_predictions.append(predicted_gender)
                all_ground_truths.append(ground_truth)
                
                is_correct = predicted_gender == ground_truth
                
                if i > 0:
                    timing_stats.add_record(prefill_time, decode_time, output_tokens)
                
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_gender = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                
                all_predictions.append(predicted_gender)
                all_ground_truths.append(ground_truth)
            
            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_gender,
                "is_correct": is_correct,
                "audio_tokens": audio_token_length if 'audio_token_length' in locals() else 0,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            }
            
            all_sample_results.append(sample_result)
            torch.cuda.empty_cache()
            
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p in ['male', 'female'] and t in ['male', 'female']) / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p in ['male', 'female'] and t in ['male', 'female']))
            
            pbar.set_postfix({
                ' sample': f'{i+1}/{len(samples)}',
                ' accuracy': f'{current_accuracy:.3f}',
                'speaker': speaker_id[:8] + '...' if len(speaker_id) > 8 else speaker_id
            })
            
            pbar.update()
    
    metrics_result = calculate_metrics(all_predictions, all_ground_truths)
    final_stats = timing_stats.get_summary()
    
    total_samples = len(all_sample_results)
    correct_samples = sum(1 for result in all_sample_results if result['is_correct'])
    
    male_samples = [r for r in all_sample_results if r['ground_truth'] == 'male']
    female_samples = [r for r in all_sample_results if r['ground_truth'] == 'female']
    
    male_correct = sum(1 for r in male_samples if r['is_correct'])
    female_correct = sum(1 for r in female_samples if r['is_correct'])
    
    results = {
        "samples": all_sample_results,
        "summary": {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": correct_samples / total_samples if total_samples > 0 else 0,
            "male_total": len(male_samples),
            "male_correct": male_correct,
            "male_accuracy": male_correct / len(male_samples) if len(male_samples) > 0 else 0,
            "female_total": len(female_samples),
            "female_correct": female_correct,
            "female_accuracy": female_correct / len(female_samples) if len(female_samples) > 0 else 0,
            "metrics": metrics_result,
            "timing": final_stats
        }
    }
    
    json_output_file = f'{result_dir}/VoxCeleb_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n===  evaluation result summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Total accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Male accuracy: {results['summary']['male_accuracy']:.2%} ({results['summary']['male_correct']}/{results['summary']['male_total']})")
    print(f"Female accuracy: {results['summary']['female_accuracy']:.2%} ({results['summary']['female_correct']}/{results['summary']['female_total']})")
    print(f"F1 Score: {metrics_result['f1_score']:.4f}")
    print(f"Precision: {metrics_result['precision']:.4f}")  
    print(f"Recall: {metrics_result['recall']:.4f}")
    print(f"Average inference time: {final_stats['avg_total_time']:.4f} seconds (excluding first samples)")
    print(f"Average Prefill time: {final_stats['avg_prefill_time']:.4f} seconds (excluding first samples)")
    print(f"Average Decode time: {final_stats['avg_decode_time']:.4f} seconds (excluding first samples)")
    print(f"Results saved to: {json_output_file}")

if __name__ == "__main__":
    main()