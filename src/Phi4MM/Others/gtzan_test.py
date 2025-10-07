import os
import json
import time
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
from transformers import logging
logging.set_verbosity_error()

class GTZANTimingStats:
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
        if ch in ["A", "B", "C", "D"]:
            return ch
    return resp.split()[0] if resp.split() else ""

def load_audio_for_gtzan(audio_path, audio_cache=None):
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

def create_gtzan_prompt(question, options):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    instruction = "Listen to this audio segment and identify the music genre based on what you hear."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted_options += f"{letter}. {opt}\n"
    
    prompt = f"{user_prompt}<|audio_1|>{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}{prompt_suffix}{assistant_prompt}"
    
    return prompt

def load_gtzan_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    valid_samples = []
    for item in metadata:
        if all(key in item for key in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            valid_samples.append(item)
    
    print(f"From {len(metadata)}  entries,loaded {len(valid_samples)}  valid samples")
    return valid_samples

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

    data_path_root = '/root/autodl-tmp/project/Phi-4-multimodal-instruct/eval/GTZAN/concatenated_audio'
    metadata_file = os.path.join(data_path_root, 'music_genre_classification_meta.json')
    result_dir = os.environ.get("RESULTS_DIR", './GTZAN_Results')
    os.makedirs(result_dir, exist_ok=True)

    output_file = f'{result_dir}/gtzan_results.json'
    timing_output_file = f'{result_dir}/timing_stats_{method_is}_{prune_ratio}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    _AUDIO_SPECIAL_TOKEN_ID = 200011

    timing_stats = GTZANTimingStats()

    print(f"\n=== GTZAN evaluation configuration ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning layer index: {prune_layer_idx}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {data_path_root}")
    print(f" metadata file: {metadata_file}")
    if sample_limit > 0:
        print(f" samplelimit: {sample_limit}")
    print("=" * 30)

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

    print(f"LoadGTZAN metadata: {metadata_file}")
    if not os.path.exists(metadata_file):
        print(f"Error:  metadata file does not exist: {metadata_file}")
        return
    
    samples = load_gtzan_metadata(metadata_file)
    
    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Apply sample limit，processed {len(samples)}  samples")

    genre_stats = {}
    for sample in samples:
        genre = sample.get("genre_label", "unknown")
        genre_stats[genre] = genre_stats.get(genre, 0) + 1
    
    print(f"genrestatistics: {genre_stats}")

    audio_cache = {}
    results = []
    correct_count = 0
    genre_correct = {genre: 0 for genre in genre_stats.keys()}
    genre_total = {genre: 0 for genre in genre_stats.keys()}

    print(f"Start evaluation {len(samples)}  samples...")
    progress_bar = tqdm(enumerate(samples), total=len(samples), desc="GTZANevaluation")

    for idx, sample in progress_bar:
        audio_rel = sample["path"]
        audio_full = os.path.join(data_path_root, audio_rel)
        
        if not os.path.exists(audio_full):
            print(f"Warning: Audio file does not exist: {audio_full}")
            continue

        audio_raw, sr = load_audio_for_gtzan(audio_full, audio_cache)
        audio = prepare_audio_for_processor(audio_raw[0])
        audio_np, sr = audio_cache[audio_full]

        options = [
            sample["choice_a"],
            sample["choice_b"], 
            sample["choice_c"],
            sample["choice_d"]
        ]

        prompt = create_gtzan_prompt(sample['question'], options)

        current_genre = sample.get("genre_label", "unknown")
        genre_total[current_genre] = genre_total.get(current_genre, 0) + 1

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
            max_new_tokens=3,
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
        pred = clean_text_response(resp)

        correct = int(pred == sample["answer_gt"])
        if correct:
            correct_count += 1
            genre_correct[current_genre] = genre_correct.get(current_genre, 0) + 1

        current_acc = (correct_count / (idx + 1)) * 100
        progress_bar.set_postfix({
            'acc': f'{current_acc:.2f}%', 
            'ans': f'{pred}/{sample["answer_gt"]}',
            'genre': current_genre,
            'audio_len': f'{len(audio_np)/sr:.1f}s'
        })

        results.append({
            "idx": idx,
            "uniq_id": sample.get("uniq_id", idx),
            "genre_label": current_genre,
            "path": audio_rel,
            "question": sample["question"],
            "options": options,
            "prediction": pred,
            "ground_truth": sample["answer_gt"],
            "correct": correct,
            "response_text": resp
        })

        if idx > 0:
            timing_stats.add_record(
                prefill_time, decode_time, 
                out_ids.shape[1] - inputs["input_ids"].shape[1],
                inputs["input_ids"].shape[1],
                len(audio_np) / sr
            )

    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0

    genre_accuracies = {}
    for genre in genre_stats.keys():
        if genre_total.get(genre, 0) > 0:
            genre_accuracies[genre] = genre_correct.get(genre, 0) / genre_total[genre] * 100

    summary = {
        "total_samples": total,
        "correct_samples": sum(r["correct"] for r in results),
        "overall_accuracy": overall_acc,
        "genre_stats": genre_stats,
        "genre_accuracies": genre_accuracies,
        "genre_correct": genre_correct,
        "genre_total": genre_total,
        "config": {
            "gpu_id": gpu_id,
            "prune_layer_idx": prune_layer_idx,
            "prune_ratio": prune_ratio,
            "prune_method": method_is,
            "sample_limit": sample_limit,
            "data_path": data_path_root
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

    print(f"\n=== GTZAN evaluation result summary ===")
    print(f"Total samples: {total}")
    print(f"Total accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    
    print(f"\nEachgenre accuracy:")
    for genre, acc in genre_accuracies.items():
        correct_num = genre_correct.get(genre, 0)
        total_num = genre_total.get(genre, 0)
        print(f"  {genre}: {acc:.2f}% ({correct_num}/{total_num})")
    
    timing_summary = timing_stats.get_summary()
    print(f"\nAverage inference time: {timing_summary.get('avg_total_time', 0):.4f} seconds (excluding first samples)")
    print(f"Average Prefill time: {timing_summary.get('avg_prefill_time', 0):.4f} seconds (excluding first samples)")
    print(f"Average Decode time: {timing_summary.get('avg_decode_time', 0):.4f} seconds (excluding first samples)")
    print(f"Average throughput: {timing_summary.get('avg_tokens_per_sec', 0):.2f} tokens/ seconds")
    print(f"Results saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
    import sys
    main()
