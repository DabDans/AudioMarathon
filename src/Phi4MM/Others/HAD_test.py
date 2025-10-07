
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
random.seed(42)
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

data_path_root = '/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/HAD/concatenated_audio'
result_dir = './HAD_Results'
os.makedirs(result_dir, exist_ok=True)

output_file = f'{result_dir}/HAD_results_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.jsonl'
timing_output_file = f'{result_dir}/HAD_timing_stats_gpu{gpu_id}_{method_is}_prune:{prune_ratio}.json'
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

def load_had_dataset(root_dir):
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    
    all_samples = []
    
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "real",
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
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)?",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "fake",
                "task": "Audio_Authenticity_Detection"
            })
    
    print(f"Totalloaded {len(all_samples)} audio sample")
    
    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"Original samples count: real={len(real_samples)}, fake={len(fake_samples)}")
    
    min_samples_per_category = min(len(real_samples), len(fake_samples))
    
    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)
    
    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)
    
    balanced_samples = real_samples + fake_samples
    
    random.shuffle(balanced_samples)
    
    print(f"After balancing samples count: real={len(real_samples)}, fake={len(fake_samples)}, Total={len(balanced_samples)}")
    
    return balanced_samples

def extract_authenticity_answer(text, choice_a="real", choice_b="fake"):
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
    if choice_a_lower == "real" and choice_b_lower == "fake":
        real_match = re.search(r'\breal\b|\bauthentic\b|\bgenuine\b', text_lower) is not None
        fake_match = re.search(r'\bfake\b|\bartificial\b|\bsynthetic\b|\bsynthesized\b', text_lower) is not None
        
        if real_match and not fake_match:
            return "real"
        if fake_match and not real_match:
            return "fake"
    
    return ""

def main():
    print("LoadPhi-4-multimodal-instructmodel...")
    model_path = "microsoft/Phi-4-multimodal-instruct"
    model_revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        revision=model_revision,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=model_revision,
        device_map="balanced_low_0",
        torch_dtype="auto",
        attn_implementation = "sdpa",
        trust_remote_code=True    )
    model.eval()
    
    timing_stats = FolderTimingStats()
    
    generation_config = GenerationConfig.from_pretrained(
        model_path, 
        revision=model_revision
    )
    
    samples = load_had_dataset(data_path_root)
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Apply sample limit，processed {len(samples)}  samples")
    
    grouped_samples = {"real": [], "fake": []}
    for sample in samples:
        grouped_samples[sample["label"]].append(sample)
    
    real_count = len(grouped_samples["real"])
    fake_count = len(grouped_samples["fake"])
    print(f"Classification statistics: real sample={real_count}, fake sample={fake_count}")
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
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
    
    with tqdm(total=len(samples), desc="processedHADaudio sample", position=0, leave=True, **tqdm_kwargs) as pbar:
        
        timing_stats.set_current_folder("HAD_Audio_Detection")
        
        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            label = item['label']
            task = item.get('task', 'Audio_Authenticity_Detection')
            
            instruction = "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'."
            full_prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
            
            try:
                audio = prepare_audio_for_processor(audio_path)
              
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
                        max_new_tokens=5,
                        generation_config=generation_config,
                        return_dict_in_generate=True
                    )
                decode_time = time.time() - decode_start
                
                tokens = generate_ids.sequences[:, inputs['input_ids'].shape[1]:]
                output_tokens = len(tokens[0])
                output = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                
                predicted_label = extract_authenticity_answer(output)
                
                ground_truth = item["answer_gt"].lower().strip()
                
                is_correct = predicted_label == ground_truth
                
                results["summary"]["total_samples"] += 1
                if ground_truth in ["real", "fake"]:
                    results["summary"][f"{ground_truth}_total"] += 1
                    if is_correct:
                        results["summary"][f"{ground_truth}_correct"] += 1
                        results["summary"]["correct_samples"] += 1
                
                results["summary"]["timing"]["total_prefill_time"] += prefill_time
                results["summary"]["timing"]["total_decode_time"] += decode_time
                results["summary"]["timing"]["total_total_time"] += (prefill_time + decode_time)
                
                timing_stats.add_record(prefill_time, decode_time, output_tokens)
                
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_label = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                output_tokens = 0
                audio_token_length = 0
            
            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "audio_label": label,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_label,
                "is_correct": is_correct,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": prefill_time + decode_time
            }
            
            results["samples"].append(sample_result)
            torch.cuda.empty_cache()
            
            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0
                
                pbar.set_postfix_str(
                    f" accuracy:{current_accuracy:.2%}"
                )
                
                if is_screen_env:
                    print(f"   progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f" accuracy: {current_accuracy:.2%}")
            
            pbar.update()
    
    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_prefill_time"] = results["summary"]["timing"]["total_prefill_time"] / total_samples
        results["summary"]["timing"]["avg_decode_time"] = results["summary"]["timing"]["total_decode_time"] / total_samples
        results["summary"]["timing"]["avg_total_time"] = results["summary"]["timing"]["total_total_time"] / total_samples
    
    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0
    results["summary"]["real_accuracy"] = results["summary"]["real_correct"] / results["summary"]["real_total"] if results["summary"]["real_total"] > 0 else 0
    results["summary"]["fake_accuracy"] = results["summary"]["fake_correct"] / results["summary"]["fake_total"] if results["summary"]["fake_total"] > 0 else 0
    
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
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    timing_stats.export_to_json(timing_output_file)
    
    print("\n=== HAD Audio Authenticity Detection evaluation result summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Total accuracy: {results['summary']['accuracy']:.2%}")
    print(f"realaudio accuracy: {results['summary']['real_accuracy']:.2%} ({results['summary']['real_correct']}/{results['summary']['real_total']})")
    print(f"fakeaudio accuracy: {results['summary']['fake_accuracy']:.2%} ({results['summary']['fake_correct']}/{results['summary']['fake_total']})")
    print(f"Precision (Precision): {precision:.2%}")
    print(f"Recall (Recall): {recall:.2%}")
    print(f"F1 score: {f1_score:.2%}")
    print(f"Average inference time: {results['summary']['timing']['avg_total_time']:.4f} seconds")
    print(f"Average Prefill time: {results['summary']['timing']['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {results['summary']['timing']['avg_decode_time']:.4f} seconds")
    print(f"Results saved to: {json_output_file}")

if __name__ == "__main__":
    main()