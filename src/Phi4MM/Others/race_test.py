import os
import json
import glob
import time
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
from transformers import logging
logging.set_verbosity_error()

class RaceTimingStats:
    def __init__(self):
        self.timing_records = []
        self.total_samples = 0
        self.total_prefill_time = 0
        self.total_decode_time = 0
        self.total_tokens = 0
    
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration):
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
            "audio_duration": audio_duration,
            "tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0
        }
        self.timing_records.append(record)
    
    def get_summary(self):
        if self.total_samples == 0:
            return {}
        
        return {
            "total_samples": self.total_samples,
            "avg_prefill_time": self.total_prefill_time / self.total_samples,
            "avg_decode_time": self.total_decode_time / self.total_samples,
            "avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / self.total_samples,
            "avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0
        }
    
    def export_to_json(self, output_file):
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_file

def clean_text_response(response):
    if not response:
        return ""
    resp = response.strip().upper()
    for ch in resp:
        if ch in ["A","B","C","D"]:
            return ch
    return resp.split()[0] if resp.split() else ""

def load_audio_for_race(audio_path, audio_cache=None):
    if audio_cache is not None and audio_path in audio_cache:
        audio_np, sr = audio_cache[audio_path]
    else:
        audio_np, sr = sf.read(audio_path)
        if len(audio_np.shape) > 1:
            audio_np = audio_np[:, 0]
        
        if audio_cache is not None:
            audio_cache[audio_path] = (audio_np, sr)
    
    return [audio_np], sr

def prepare_audio_for_processor(audio_data, target_sr=16000):
    if isinstance(audio_data, list):
        return [(audio, target_sr) for audio in audio_data]
    else:
        return [(audio_data, target_sr)]

def create_race_prompt(question, options):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted_options += f"{letter}. {opt}\n"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def main():
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
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    data_path_root = os.path.abspath('/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/test/race_audio')
    
    results_dir_name = os.environ.get("RESULTS_DIR", "Race_Results")
    if not os.path.isabs(results_dir_name):
        result_dir = os.path.abspath(results_dir_name)
    else:
        result_dir = results_dir_name
    
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")

    output_file = os.path.join(result_dir, 'race_results.json')
    timing_output_file = os.path.join(result_dir, f'timing_stats_{method_is}_{prune_ratio}.json')
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    _AUDIO_SPECIAL_TOKEN_ID = 200011

    timing_stats = RaceTimingStats()

    print(f"\n=== RACE evaluation configuration ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data directory: {data_path_root}")
    print(f"Results directory: {result_dir}")
    print("=" * 50)

    model_path = "microsoft/Phi-4-multimodal-instruct"
    print("Loadmodeland processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="balanced_low_0",
        torch_dtype="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    model.eval()
    
    generation_config = GenerationConfig.from_pretrained(model_path)

    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: cannot findbenchmarkfile: {bench_path}")
        return
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if sample_limit > 0 and len(benchmark) > sample_limit:
        benchmark = benchmark[:sample_limit]
        print(f" samples countlimit is: {sample_limit}")

    audio_cache = {}
    results = []

    correct_count = 0
    correct_high = 0
    total_high = 0
    correct_middle = 0
    total_middle = 0

    print(f"Start evaluation {len(benchmark)}  samples...")
    
    is_screen_env = not os.sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        tqdm.monitor_interval = 0
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': os.sys.stdout
    }

    progress_bar = tqdm(enumerate(benchmark), total=len(benchmark), 
                       desc="RACEevaluation", **tqdm_kwargs)

    for idx, sample in progress_bar:
        audio_rel = sample["audio_path"]
        audio_full = os.path.join(data_path_root, audio_rel)
        
        if not os.path.exists(audio_full):
            print(f"Warning: Audio file does not exist: {audio_full}")
            continue
            
        audio_raw, sr = load_audio_for_race(audio_full, audio_cache)
        audio = prepare_audio_for_processor(audio_raw[0])
        audio_np, sr = audio_cache[audio_full]
        prompt = create_race_prompt(sample['question'], sample['options'])

        if "high" in audio_rel:
            total_high += 1
        elif "middle" in audio_rel:
            total_middle += 1

        inputs = processor(
            text=prompt,
            audios=audio,
            return_tensors="pt",
        ).to(model.device)
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
        full_time = prefill_time + decode_time

        resp = processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        pred = clean_text_response(resp)

        correct = int(pred == sample["answer"])
        if correct:
            correct_count += 1
            if "high" in audio_rel:
                correct_high += 1
            elif "middle" in audio_rel:
                correct_middle += 1
        
        current_acc = (correct_count / (idx + 1)) * 100
        progress_bar.set_postfix({
            'acc': f'{current_acc:.2f}%', 
            'ans': f'{pred}/{sample["answer"]}',
            'audio_len': f'{len(audio_np)/sr:.1f}s'
        })

        results.append({
            "idx": idx,
            "article_id": sample.get("article_id", ""),
            "question_idx": sample.get("question_idx", idx),
            "pred": pred, 
            "gt": sample["answer"],
            "correct": correct,
            "audio_path": audio_rel,
            "subset": "high" if "high" in audio_rel else "middle"
        })

        if idx > 0:
            timing_stats.add_record(
                prefill_time=prefill_time,
                decode_time=decode_time,
                output_tokens=out_ids.shape[1] - inputs["input_ids"].shape[1],
                input_tokens=inputs["input_ids"].shape[1],
                audio_duration=len(audio_np) / sr
            )

    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "high_accuracy": correct_high / total_high * 100 if total_high > 0 else 0,
        "middle_accuracy": correct_middle / total_middle * 100 if total_middle > 0 else 0,
        "high_correct": correct_high,
        "high_total": total_high,
        "middle_correct": correct_middle,
        "middle_total": total_middle,
        "config": {
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit
        },
        "timing": timing_stats.get_summary()
    }

    final_results = {
        "summary": summary,
        "samples": results
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving resultsto: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"Saving timing statisticsto: {timing_output_file}")
    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== RACE evaluation result summary ===")
    print(f"Total samples: {total}")
    print(f"Total accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    if total_high > 0:
        print(f"HIGHset accuracy: {correct_high/total_high*100:.2f}% ({correct_high}/{total_high})")
    if total_middle > 0:
        print(f"MIDDLEset accuracy: {correct_middle/total_middle*100:.2f}% ({correct_middle}/{total_middle})")
    
    timing_summary = timing_stats.get_summary()
    print(f"Average inference time: {timing_summary.get('avg_total_time', 0):.4f} seconds (excluding first samples)")
    print(f"Average Prefill time: {timing_summary.get('avg_prefill_time', 0):.4f} seconds (excluding first samples)")
    print(f"Average Decode time: {timing_summary.get('avg_decode_time', 0):.4f} seconds (excluding first samples)")
    print(f"Average throughput: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/ seconds (excluding first samples)")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    import sys
    main()
