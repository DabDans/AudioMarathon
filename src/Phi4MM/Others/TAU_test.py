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

data_path_root = '/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/TAU'
audio_dir = os.path.join(data_path_root, 'concatenated_resampled')
result_dir = './TAU_Results'
os.makedirs(result_dir, exist_ok=True)

output_file = f'{result_dir}/TAU_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/TAU_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
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

def load_tau_acoustic_scene_dataset(root_dir):
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"From{meta_file}loaded{len(metadata)} samples metadata")
    
    scene_counts = {}
    
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)
        
        if not os.path.exists(wav_path):
            print(f"Warning: file does not exist {wav_path}")
            continue
        
        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"]
        
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        
        all_samples.append({
            "scene_label": scene_label,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "answer_gt": answer_gt,
            "task": "Acoustic_Scene_Classification"
        })
    
    print(f"Totalloaded {len(all_samples)} validaudio sample")
    
    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count} samples ({count/len(all_samples)*100:.1f}%)")
    
    if sample_limit > 0 and sample_limit < len(all_samples):
        print(f"Due to sample limit setting, randomly selecting{sample_limit} samplesforevaluation")
        all_samples = random.sample(all_samples, sample_limit)
        
    random.shuffle(all_samples)
    
    return all_samples, scene_counts

def extract_acoustic_scene_answer(text, choices=None):
    text_lower = text.lower().strip()
    
    options = ['a', 'b', 'c', 'd']
    
    if text_lower in options:
        return text_lower.upper()
    
    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    
    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    
    if choices:
        best_match = None
        max_overlap = 0
        
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()
            if choice_lower in text_lower:
                return chr(65 + i)
            
            keywords = choice_lower.split(' - ')[0].split()
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        
        if best_match and max_overlap > 1:
            return best_match
    
    return ""

def group_samples_by_scene(samples):
    grouped = {}
    for sample in samples:
        scene = sample["scene_label"]
        if scene not in grouped:
            grouped[scene] = []
        grouped[scene].append(sample)
    return grouped

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
    
    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    
    grouped_samples = group_samples_by_scene(samples)
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "scene_stats": {scene: {"total": 0, "correct": 0} for scene in scene_counts},
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
    
    with tqdm(total=len(grouped_samples), desc="processedscene groups", position=0, leave=True, **tqdm_kwargs) as pbar_folders:
        folder_count = 0
        total_folders = len(grouped_samples)
        
        for scene_label, items in grouped_samples.items():
            folder_count += 1
            pbar_folders.set_description(f"processedscene[{folder_count}/{total_folders}]: {scene_label}")
            
            timing_stats.set_current_folder(scene_label)
            
            sample_count = 0
            total_samples = len(items)

            for i, item in enumerate(items):
                sample_count = i + 1
                wav_path = item['wav_path']
                
                instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.\n"
                instruction += f"A: {item['choice_a']}\nB: {item['choice_b']}\nC: {item['choice_c']}\nD: {item['choice_d']}\n"
                instruction += "Respond with only the letter of your answer (A, B, C, or D)."
                
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
                        model.config.audio_layer_idx = prune_layer_idx
                        model.config.audio_token_num = audio_token_length
                        model.config.audio_token_start = audio_token_start
                        model.config.audio_prune_ratio = prune_ratio
                        model.config.random = use_random
                        model.config.frame = use_frame
                    
                    prefill_start = time.time()
                    with torch.no_grad():
                        outputs = model(
                            **inputs,
                            use_cache=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True
                        )
                    prefill_time = time.time() - prefill_start
                    
                    decode_start = time.time()
                    with torch.no_grad():
                        generate_ids = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            generation_config=generation_config,
                            return_dict_in_generate=True
                        )
                    decode_time = time.time() - decode_start
                    
                    tokens = generate_ids.sequences[:, inputs['input_ids'].shape[1]:]
                    output_tokens = len(tokens[0])
                    output = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                    choices = [item['choice_a'], item['choice_b'], item['choice_c'], item['choice_d']]
                    predicted_answer = extract_acoustic_scene_answer(output, choices)
                    
                    ground_truth = item["answer_gt"].upper()
                    is_correct = predicted_answer == ground_truth
                    
                    results["summary"]["total_samples"] += 1
                    results["summary"]["scene_stats"][scene_label]["total"] += 1
                    
                    if is_correct:
                        results["summary"]["correct_samples"] += 1
                        results["summary"]["scene_stats"][scene_label]["correct"] += 1
                    
                    results["summary"]["timing"]["total_prefill_time"] += prefill_time
                    results["summary"]["timing"]["total_decode_time"] += decode_time
                    results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                    
                    timing_stats.add_record(prefill_time, decode_time, output_tokens)
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    traceback.print_exc()
                    output = "ERROR"
                    predicted_answer = "ERROR"
                    is_correct = False
                    prefill_time = 0
                    decode_time = 0
                    output_tokens = 0
                
                sample_result = {
                    "audio_file": os.path.basename(wav_path),
                    "scene_label": scene_label,
                    "ground_truth": ground_truth if 'ground_truth' in locals() else "ERROR",
                    "model_output": output,
                    "extracted_answer": predicted_answer,
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
    if total_samples > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / total_samples
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / total_samples
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / total_samples
    
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    
    for scene in results["summary"]["scene_stats"]:
        stats = results["summary"]["scene_stats"][scene]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0
    
    json_output_file = f'{result_dir}/TAU_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n===  evaluation result summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Total accuracy: {results['summary']['accuracy']:.2%}")
    
    sorted_scenes = sorted(
        [(scene, stats["accuracy"], stats["correct"], stats["total"]) 
         for scene, stats in results["summary"]["scene_stats"].items()],
        key=lambda x: x[1], reverse=True
    )
    
    print("\nscene accuracy:")
    for scene, acc, correct, total in sorted_scenes:
        print(f"  {scene}: {acc:.2%} ({correct}/{total})")
    
    print(f"\nAverage inference time: {results['summary']['timing']['avg_total_time']:.4f} seconds")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} seconds")
    print(f"Results saved to: {json_output_file}")

if __name__ == "__main__":
    main()