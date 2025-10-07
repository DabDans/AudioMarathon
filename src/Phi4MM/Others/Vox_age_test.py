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

output_file = f'{result_dir}/VoxCeleb_age_results_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/VoxCeleb_age_timing_stats_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
print(f"Results will be saved to: {output_file}")
print(f"Timing statistics will be saved to: {timing_output_file}")

_AUDIO_SPECIAL_TOKEN_ID = 200011

class FolderTimingStats:
    def __init__(self):
        self.folder_stats = {}
        self.current_folder = None
    
    def set_current_folder(self, folder_name):
        self.current_folder = folder_name
        if folder_name not in self.folder_stats:
            self.folder_stats[folder_name] = {
                "samples": 0,
                "total_prefill_time": 0.0,
                "total_decode_time": 0.0,
                "total_tokens": 0,
                "timing_records": []
            }
    
    def add_record(self, prefill_time, decode_time, output_tokens):
        if self.current_folder is None:
            return
        
        folder_data = self.folder_stats[self.current_folder]
        folder_data["samples"] += 1
        folder_data["total_prefill_time"] += prefill_time
        folder_data["total_decode_time"] += decode_time
        folder_data["total_tokens"] += output_tokens
        
        folder_data["timing_records"].append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        })
    
    def export_to_json(self, output_file):
        result = {
            "folder_summaries": {
                folder: {
                    "folder": folder,
                    "samples": stats["samples"],
                    "avg_prefill_time": stats["total_prefill_time"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_decode_time": stats["total_decode_time"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_total_time": (stats["total_prefill_time"] + stats["total_decode_time"]) / stats["samples"] if stats["samples"] > 0 else 0,
                    "total_tokens": stats["total_tokens"],
                    "avg_tokens": stats["total_tokens"] / stats["samples"] if stats["samples"] > 0 else 0,
                    "avg_tokens_per_sec": stats["total_tokens"] / stats["total_decode_time"] if stats["total_decode_time"] > 0 else 0
                }
                for folder, stats in self.folder_stats.items() if stats["samples"] > 0
            },
            "detailed_records": self.folder_stats
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

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
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
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
        age_group = item["answer_gt"].strip()
        speaker_age = item.get("speaker_age", 0)
        
        all_samples.append({
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "choice_e": item["choice_e"],
            "answer_gt": age_group,
            "task": "Speaker_Age_Classification"
        })
    
    print(f"Totalloaded {len(all_samples)} validaudio sample")
    
    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Apply sample limit: From{len(all_samples)} samplesrandomly selecting{sample_limit}")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"limitafter samples count: {len(all_samples)}")
    
    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age groups distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    random.shuffle(all_samples)
    
    return all_samples

def extract_age_answer(text, choices):
    text_lower = text.lower().strip()
    
    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return choices["choice_d"]
    if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)'):
        return choices["choice_e"]
        
    for option, choice_text in choices.items():
        option_letter = option[-1].lower()
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"{option_letter})" in text_lower:
            return choice_text
    
    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    
    if len(choice_matches) == 1:
        return choice_matches[0]
    
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
    
    timing_stats = FolderTimingStats()
    
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    
    grouped_samples = {}
    for sample in samples:
        speaker_id = sample["speaker_id"]
        if speaker_id not in grouped_samples:
            grouped_samples[speaker_id] = []
        grouped_samples[speaker_id].append(sample)
    
    age_group_counts = {}
    for s in samples:
        group = s["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age groups statistics:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "age_group_stats": {},
            "timing": {
                "avg_prefill_time": 0,
                "avg_decode_time": 0,
                "avg_total_time": 0,
                "total_prefill_time": 0,
                "total_decode_time": 0,
                "total_total_time": 0,
            }
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
    
    with tqdm(total=len(grouped_samples), desc="processed speaker groups", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        for speaker_id, items in grouped_samples.items():
            folder_count += 1
            pbar_folders.set_description(f"processed speaker[{folder_count}/{total_folders}]: {speaker_id}")
            
            timing_stats.set_current_folder(speaker_id)
            sample_count = 0
            total_samples = len(items)            
            for i, item in enumerate(items):
                sample_count = i + 1
                wav_path = item['wav_path']
                task = item.get('task', 'Speaker_Age_Classification')
                
                instruction = "Listen to this audio and identify the speaker's age group. Choose the most appropriate option: (a) Young Adult (18-30), (b) Early Career (31-40), (c) Mid Career (41-50), (d) Senior (51-70), (e) Elderly (71+). Answer with only the letter (a, b, c, d, or e)."
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
                    
                    choices = {
                        "choice_a": item["choice_a"],
                        "choice_b": item["choice_b"],
                        "choice_c": item["choice_c"],
                        "choice_d": item["choice_d"],
                        "choice_e": item["choice_e"]
                    }
                    predicted_age_group = extract_age_answer(output, choices)
                    
                    ground_truth = item["age_group"].strip()
                    is_correct = predicted_age_group == ground_truth
                    
                    results["summary"]["total_samples"] += 1
                    if ground_truth not in results["summary"]["age_group_stats"]:
                        results["summary"]["age_group_stats"][ground_truth] = {
                            "total": 0,
                            "correct": 0
                        }
                    
                    results["summary"]["age_group_stats"][ground_truth]["total"] += 1
                    if is_correct:
                        results["summary"]["age_group_stats"][ground_truth]["correct"] += 1
                        results["summary"]["correct_samples"] += 1
                    
                    if i > 0:
                        results["summary"]["timing"]["total_prefill_time"] += prefill_time
                        results["summary"]["timing"]["total_decode_time"] += decode_time
                        results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                        timing_stats.add_record(prefill_time, decode_time, output_tokens)
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    traceback.print_exc()
                    output = "ERROR"
                    predicted_age_group = "error"
                    is_correct = False
                    prefill_time = 0
                    decode_time = 0
                    output_tokens = 0
                
                sample_result = {
                    "audio_file": os.path.basename(wav_path),
                    "speaker_id": item["speaker_id"],
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "extracted_answer": predicted_age_group,
                    "is_correct": is_correct,
                    "audio_tokens": audio_token_length if 'audio_token_length' in locals() else 0,
                    "output_tokens": output_tokens,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "total_time": prefill_time + decode_time
                }
                results["samples"].append(sample_result)
                torch.cuda.empty_cache()
                
                update_interval = 10 if is_screen_env else 1
                
                if sample_count % update_interval == 0 or sample_count == total_samples:
                    current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                    
                    pbar_folders.set_postfix_str(
                        f" sample:{sample_count}/{total_samples},  accuracy:{current_accuracy:.2%}"
                    )
                    
                    if is_screen_env:
                        print(f"   progress: {sample_count}/{total_samples} ({sample_count/total_samples*100:.1f}%), "
                              f" accuracy: {current_accuracy:.2%}")
            
            pbar_folders.update()

    
    total_samples = results["summary"]["total_samples"]
    timing_sample_count = max(0, total_samples - 1)
    if timing_sample_count > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / timing_sample_count
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / timing_sample_count
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / timing_sample_count
    else:
        results["summary"]["timing"]["avg_prefill_time"] = 0
        results["summary"]["timing"]["avg_decode_time"] = 0
        results["summary"]["timing"]["avg_total_time"] = 0
    
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    
    for age_group, stats in results["summary"]["age_group_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    json_output_file = f'{result_dir}/VoxCeleb_age_results_gpu{gpu_id}_{method_is}_prune_{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n=== Age Classification evaluation result summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Total accuracy: {results['summary']['accuracy']:.2%}")
    
    print("\nEach age groups accuracy:")
    for age_group, stats in results["summary"]["age_group_stats"].items():
        print(f"  {age_group}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"\nAverage inference time: {results['summary']['timing']['avg_total_time']:.4f} seconds (excluding first samples)")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} seconds (excluding first samples)")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} seconds (excluding first samples)")
    print(f"Results saved to: {json_output_file}")

if __name__ == "__main__":
    main()