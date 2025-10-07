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
import numpy as np
import librosa
import traceback
import gc
from collections import defaultdict

# Disable warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"
os.environ['PYTHONUNBUFFERED'] = '1'

# Set random seed
random.seed(42)

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

def configure_DART_aero1(model, args):
    """Configure DART sparse attention mechanism for Aero1 model"""
    if args.sparse:
        model.config.image_layer_idx = None  # Do not process image
        model.config.audio_layer_idx = args.pruned_layer
        model.config.audio_prune_ratio = 1.0 - args.reduction_ratio  # Convert to prune ratio
        model.config.random = False  # Use intelligent pruning instead of random
        model.config.frame = False   # Do not use frame pruning
        print(f"DART config applied to Aero1 model: layer={args.pruned_layer}, prune_ratio={1.0 - args.reduction_ratio:.3f}")
    else:
        # Disable pruning
        model.config.image_layer_idx = None
        model.config.audio_layer_idx = None
        model.config.audio_prune_ratio = 0.0
        model.config.random = False
        model.config.frame = False
        print("DART pruning disabled")

def calculate_metrics(predictions, ground_truths):
    """Calculate classification metrics: accuracy, precision, recall, and F1 score"""
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

def generate_sklearn_vox_dart_evaluation_report(y_true, y_pred, speaker_ids=None, labels=None):
    """
    Generate detailed evaluation report for Vox gender identification task (DART version) using sklearn
    
    Args:
        y_true: List of ground truth labels (e.g. ['male', 'female'])
        y_pred: List of predicted labels (e.g. ['male', 'female'])
        speaker_ids: List of speaker IDs for speaker-level analysis
        labels: Label names for classification report
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"error": "Invalid input data for evaluation"}
    valid_indices = []
    valid_y_true = []
    valid_y_pred = []
    valid_label_set = {'male', 'female'}
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in valid_label_set and pred_label in valid_label_set:
            valid_indices.append(i)
            valid_y_true.append(true_label)
            valid_y_pred.append(pred_label)
    if not valid_y_true:
        return {"error": "No valid labels for evaluation"}
    accuracy = accuracy_score(valid_y_true, valid_y_pred)
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        valid_y_true, valid_y_pred, average='binary', pos_label='female', zero_division=0
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
        valid_y_true, valid_y_pred, average=None, labels=['male', 'female'], zero_division=0
    )
    if labels is None:
        target_names = ['male', 'female']
    else:
        target_names = labels
    classification_rep = classification_report(
        valid_y_true, valid_y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    tp_female = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'female' and p == 'female')
    fp_female = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'male' and p == 'female')
    fn_female = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'female' and p == 'male')
    tn_female = sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == 'male' and p == 'male')
    specificity_female = tn_female / (tn_female + fp_female) if (tn_female + fp_female) > 0 else 0
    evaluation_report = {
        "overall_metrics": {
            "accuracy": accuracy,
            "precision_binary": precision_binary,
            "recall_binary": recall_binary,
            "f1_binary": f1_binary,
            "specificity_female": specificity_female,
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
        "per_gender_metrics": {},
        "confusion_matrix": {
            "true_positive_female": tp_female,
            "false_positive_female": fp_female,
            "false_negative_female": fn_female,
            "true_negative_female": tn_female
        },
        "classification_report": classification_rep,
        "sample_statistics": {
            "total_samples": len(y_true),
            "valid_samples": len(valid_y_true),
            "invalid_samples": len(y_true) - len(valid_y_true),
            "correct_predictions": sum(1 for t, p in zip(valid_y_true, valid_y_pred) if t == p),
            "male_samples": sum(1 for label in valid_y_true if label == 'male'),
            "female_samples": sum(1 for label in valid_y_true if label == 'female')
        }
    }
    gender_labels = ['male', 'female']
    for i, gender in enumerate(gender_labels):
        if i < len(precision_per_class):
            evaluation_report["per_gender_metrics"][gender] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1_score": f1_per_class[i],
                "support": int(support_per_class[i]) if i < len(support_per_class) else 0
            }
    if speaker_ids and len(speaker_ids) == len(y_true):
        speaker_analysis = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for i, speaker_id in enumerate(speaker_ids):
            if i in valid_indices:
                valid_index = valid_indices.index(i)
                speaker_analysis[speaker_id]["y_true"].append(valid_y_true[valid_index])
                speaker_analysis[speaker_id]["y_pred"].append(valid_y_pred[valid_index])
        speaker_summaries = {}
        for speaker_id, data in speaker_analysis.items():
            if len(data["y_true"]) > 0:
                speaker_accuracy = accuracy_score(data["y_true"], data["y_pred"])
                speaker_correct = sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
                speaker_summaries[speaker_id] = {
                    "sample_count": len(data["y_true"]),
                    "accuracy": speaker_accuracy,
                    "correct_count": speaker_correct,
                    "true_gender": data["y_true"][0] if len(set(data["y_true"])) == 1 else "mixed"
                }
        evaluation_report["speaker_level_analysis"] = speaker_summaries
    return evaluation_report

class VoxTimingStats:
    """Global timing statistics class for Aero1"""
    def __init__(self):
        self.samples = 0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_tokens = 0
        self.timing_records = []
    def add_record(self, prefill_time, decode_time, output_tokens, input_tokens=0, gender=None):
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
        """Get timing summary"""
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
        """Export timing stats to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return output_file
    def print_summary(self):
        """Print timing summary"""
        summary = self.get_summary()
        print(f"\n=== Timing Summary ===")
        print(f"Valid samples: {summary['samples']}")
        print(f"Average Prefill time: {summary['avg_prefill_time']:.4f}s")
        print(f"Average Decode time: {summary['avg_decode_time']:.4f}s")
        print(f"Average Total time: {summary['avg_total_time']:.4f}s")
        print(f"Average tokens/sec: {summary['avg_tokens_per_sec']:.2f}")

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
    audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resampled

def split_audio(audio_arrays):
    """Split audio into 30s chunks (480000 samples @16kHz)"""
    CHUNK_LIM = 480000
    audio_splits = []
    for i in range(0, len(audio_arrays), CHUNK_LIM):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits

def prepare_audio_for_processor(audio_path, target_sr=16000):
    """Load audio with librosa and split, compatible with Aero-1 official example"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=target_sr)
        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            audio = downsample_audio(audio, sample_rate, target_sr)
            sample_rate = target_sr
        if len(audio) > 480000:
            audio_chunks = split_audio(audio)
            return audio_chunks, sample_rate
        else:
            return [audio], sample_rate
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return [silence], target_sr

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    """Load dataset from concatenated_audio directory, based on gender_id_task_meta.json and balance male/female sample numbers"""
    meta_file = os.path.join(root_dir, "gender_id_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: metadata file not found: {meta_file}")
        return []
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        if not os.path.exists(wav_path):
            continue
        speaker_id = item["speaker_id_original"]
        gender = item["answer_gt"].lower().strip()
        sample = {
            "speaker_id": speaker_id,
            "gender": gender,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "answer_gt": gender,
            "task": "Speaker_Gender_Identification"
        }
        all_samples.append(sample)
    print(f"Total {len(all_samples)} valid audio samples loaded")
    male_samples = [sample for sample in all_samples if sample["gender"].lower() == "male"]
    female_samples = [sample for sample in all_samples if sample["gender"].lower() == "female"]
    print(f"Original sample count: male={len(male_samples)}, female={len(female_samples)}")
    min_samples_per_gender = min(len(male_samples), len(female_samples))
    if sample_limit > 0:
        max_per_gender = sample_limit // 2
        min_samples_per_gender = min(min_samples_per_gender, max_per_gender)
        print(f"Sample limit applied: max {min_samples_per_gender} per gender")
    if len(male_samples) > min_samples_per_gender:
        male_samples = random.sample(male_samples, min_samples_per_gender)
    if len(female_samples) > min_samples_per_gender:
        female_samples = random.sample(female_samples, min_samples_per_gender)
    balanced_samples = male_samples + female_samples
    random.shuffle(balanced_samples)
    print(f"Final sample count: male={len(male_samples)}, female={len(female_samples)}, total={len(balanced_samples)}")
    return balanced_samples

def extract_gender_answer(text, choice_a="male", choice_b="female"):
    """Extract gender answer from model output text, handle direct a/b replies"""
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

def cuda_timing_inference_aero1(model, processor, inputs, max_new_tokens=10):
    """
    Inference function using CUDA Event API for precise GPU timing measurement - Aero1 adapted version
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

def main():
    args = parse_arguments()
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in ["1", "true", "yes"]

    print(f"\n=== Vox DART Gender Identification Evaluation Config (Aero-1) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"DART sparse mode: {args.sparse}")
    print(f"Pruned layers: {args.pruned_layer}")
    print(f"Retained ratio: {args.reduction_ratio}")
    print(f"Prune ratio: {1.0 - args.reduction_ratio:.3f}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    print("=" * 50)

    data_path_root = os.environ.get("VOXCELEB_DATA_PATH", 
        '/data/to/your/voxceleb/path/concatenated_audio')
    result_dir = os.environ.get("RESULTS_DIR", '/data/to/your/results/path')
    os.makedirs(result_dir, exist_ok=True)

    sparse_suffix = "_sparse" if args.sparse else "_base"
    method_name = f"dart{sparse_suffix}_layer{args.pruned_layer}_ratio{args.reduction_ratio:.3f}"
    output_file = f'{result_dir}/VoxCeleb_Aero1_results_{method_name}.json'
    timing_output_file = f'{result_dir}/VoxCeleb_Aero1_timing_stats_{method_name}.json'
    print(f"Results will be saved to: {output_file}")
    print(f"Timing stats will be saved to: {timing_output_file}")

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

    configure_DART_aero1(model, args)
    timing_stats = VoxTimingStats()
    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    if not samples:
        print("Error: No samples loaded")
        return
    male_count = sum(1 for s in samples if s["gender"].lower() == "male")
    female_count = sum(1 for s in samples if s["gender"].lower() == "female")
    print(f"Gender stats: male samples={male_count}, female samples={female_count}")
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model load - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    all_predictions = []
    all_ground_truths = []
    all_speaker_ids = []
    all_sample_results = []
    _AUDIO_SPECIAL_TOKEN_ID = 151667
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
        tqdm.monitor_interval = 0
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    print(f"Start processing {len(samples)} samples...")
    with tqdm(total=len(samples), desc="Processing VoxCeleb gender identification samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, sample in enumerate(samples):
            wav_path = sample['wav_path']
            speaker_id = sample["speaker_id"]
            ground_truth = sample["gender"].lower().strip()
            prefill_time = 0
            decode_time = 0
            total_time = 0
            output_tokens = 0
            audio_token_length = 0
            predicted_gender = ""
            is_correct = False
            output = ""
            try:
                messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                audio_chunks, sample_rate = prepare_audio_for_processor(wav_path)
                for chunk in audio_chunks:
                    messages[0]["content"].append({
                        "type": "audio",
                        "audio": "placeholder",
                    })
                instruction = "Listen to this audio and identify the speaker's gender. Is this a male or female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'."
                messages[0]["content"].append({
                    "type": "text",
                    "text": instruction
                })
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    text=prompt,
                    audios=audio_chunks,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to("cuda")
                if _AUDIO_SPECIAL_TOKEN_ID in inputs.input_ids[0]:
                    token_ids = inputs.input_ids[0].tolist()
                    audio_token_start = token_ids.index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_end = len(token_ids) - 1 - token_ids[::-1].index(_AUDIO_SPECIAL_TOKEN_ID)
                    audio_token_length = audio_token_end - audio_token_start + 1
                    if args.sparse:
                        model.config.audio_token_num = audio_token_length
                        model.config.audio_token_start = audio_token_start
                if debug_mode:
                    print(f"Processing audio: {os.path.basename(wav_path)}")
                    print(f"Speaker ID: {speaker_id}")
                    print(f"Audio chunks: {len(audio_chunks)}")
                    print(f"Gender label: {ground_truth}")
                    print(f"Audio token length: {audio_token_length}")
                result = cuda_timing_inference_aero1(
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
                predicted_gender = extract_gender_answer(output)
                all_predictions.append(predicted_gender)
                all_ground_truths.append(ground_truth)
                all_speaker_ids.append(speaker_id)
                is_correct = predicted_gender == ground_truth
                if i > 0:
                    timing_stats.add_record(prefill_time, decode_time, output_tokens, inputs['input_ids'].shape[1], gender=ground_truth)
                if debug_mode:
                    print(f"Model output: '{output}'")
                    print(f"Extracted answer: '{predicted_gender}'")
                    print(f"Correct answer: '{ground_truth}'")
                    print(f"Answer correct: {is_correct}")
                    print(f"Inference time: total={total_time:.3f}s, Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s")
            except Exception as e:
                print(f"Inference error: {e}")
                if debug_mode:
                    traceback.print_exc()
                output = "ERROR"
                predicted_gender = "error"
                is_correct = False
                prefill_time = 0
                decode_time = 0
                total_time = 0
                output_tokens = 0
                all_predictions.append("")
                all_ground_truths.append(ground_truth)
                all_speaker_ids.append(speaker_id)
            sample_result = {
                "audio_file": os.path.basename(wav_path),
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_gender,
                "is_correct": is_correct,
                "audio_chunks": len(audio_chunks) if 'audio_chunks' in locals() else 1,
                "audio_tokens": audio_token_length,
                "output_tokens": output_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time
            }
            all_sample_results.append(sample_result)
            if 'inputs' in locals():
                del inputs
            if 'audio_chunks' in locals():
                del audio_chunks
            if 'result' in locals():
                del result
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            current_accuracy = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t and p in ['male', 'female'] and t in ['male', 'female']) / max(1, sum(1 for p, t in zip(all_predictions, all_ground_truths) if p in ['male', 'female'] and t in ['male', 'female']))
            if is_screen_env and (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(samples)} ({(i+1)/len(samples)*100:.1f}%), Accuracy: {current_accuracy:.2%}")
                sys.stdout.flush()
            pbar.set_postfix({
                'Sample': f'{i+1}/{len(samples)}',
                'Accuracy': f'{current_accuracy:.3f}',
                'Speaker': speaker_id[:8] + '...' if len(speaker_id) > 8 else speaker_id
            })
            pbar.update()
    metrics_result = calculate_metrics(all_predictions, all_ground_truths)
    final_stats = timing_stats.get_summary()
    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        print(f"\n=== Generating sklearn Vox DART evaluation report ===")
        print(f"Total samples: {len(all_predictions)} (predictions), {len(all_ground_truths)} (ground truth)")
        sklearn_evaluation = generate_sklearn_vox_dart_evaluation_report(
            y_true=all_ground_truths,
            y_pred=all_predictions,
            speaker_ids=all_speaker_ids,
            labels=['male', 'female']
        )
        print(f"sklearn Vox DART evaluation report generated:")
        print(f"  Overall Accuracy: {sklearn_evaluation['overall_metrics']['accuracy']:.4f}")
        print(f"  Precision (binary): {sklearn_evaluation['overall_metrics']['precision_binary']:.4f}")
        print(f"  Recall (binary): {sklearn_evaluation['overall_metrics']['recall_binary']:.4f}")
        print(f"  F1 Score (binary): {sklearn_evaluation['overall_metrics']['f1_binary']:.4f}")
        print(f"  Specificity (female): {sklearn_evaluation['overall_metrics']['specificity_female']:.4f}")
        print(f"  Valid samples: {sklearn_evaluation['sample_statistics']['valid_samples']}/{sklearn_evaluation['sample_statistics']['total_samples']}")
    else:
        print("Warning: No valid predictions, cannot generate sklearn evaluation report")
        sklearn_evaluation = {"error": "No valid predictions for evaluation"}
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
            "sklearn_evaluation": sklearn_evaluation,
            "timing": final_stats,
            "config": {
                "model_name": "Aero-1-Audio-1.5B",
                "gpu_id": gpu_id,
                "sparse": args.sparse,
                "pruned_layer": args.pruned_layer,
                "reduction_ratio": args.reduction_ratio,
                "prune_ratio": 1.0 - args.reduction_ratio,
                "sample_limit": sample_limit,
                "data_path": data_path_root
            }
        }
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    timing_stats.export_to_json(timing_output_file)
    print(f"Timing stats saved to: {timing_output_file}")
    print("\n=== Evaluation Summary (Aero-1 + DART) ===")
    print(f"Model: Aero-1-Audio-1.5B")
    print(f"DART Config: sparse={args.sparse}, layer={args.pruned_layer}, retained ratio={args.reduction_ratio:.3f}")
    print(f"Total samples: {total_samples}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Male Accuracy: {results['summary']['male_accuracy']:.2%} ({results['summary']['male_correct']}/{results['summary']['male_total']})")
    print(f"Female Accuracy: {results['summary']['female_accuracy']:.2%} ({results['summary']['female_correct']}/{results['summary']['female_total']})")
    print(f"F1 Score: {metrics_result['f1_score']:.4f}")
    print(f"Precision: {metrics_result['precision']:.4f}")  
    print(f"Recall: {metrics_result['recall']:.4f}")
    if "sklearn_evaluation" in results["summary"] and "error" not in results["summary"]["sklearn_evaluation"]:
        sklearn_metrics = results["summary"]["sklearn_evaluation"]["overall_metrics"]
        confusion_matrix = results["summary"]["sklearn_evaluation"]["confusion_matrix"]
        print(f"\n=== Sklearn Evaluation Metrics ===")
        print(f"Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision (Binary): {sklearn_metrics['precision_binary']:.4f}")
        print(f"Recall/Sensitivity (Binary): {sklearn_metrics['recall_binary']:.4f}")
        print(f"Specificity: {sklearn_metrics['specificity_female']:.4f}")
        print(f"F1 Score (Binary): {sklearn_metrics['f1_binary']:.4f}")
        print(f"F1 Score (Macro): {sklearn_metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {sklearn_metrics['f1_weighted']:.4f}")
        print(f"\nConfusion Matrix (female as positive):")
        print(f"  True Positive (TP): {confusion_matrix['true_positive_female']} (correct female)")
        print(f"  False Positive (FP): {confusion_matrix['false_positive_female']} (male predicted as female)")
        print(f"  False Negative (FN): {confusion_matrix['false_negative_female']} (female predicted as male)")
        print(f"  True Negative (TN): {confusion_matrix['true_negative_female']} (correct male)")
        print(f"\nGender detailed metrics:")
        per_gender_metrics = results["summary"]["sklearn_evaluation"]["per_gender_metrics"]
        for gender in sorted(per_gender_metrics.keys()):
            metrics_detail = per_gender_metrics[gender]
            print(f"  {gender}: Precision={metrics_detail['precision']:.4f}, "
                  f"Recall={metrics_detail['recall']:.4f}, F1={metrics_detail['f1_score']:.4f}, "
                  f"Support={metrics_detail['support']}")
        if "speaker_level_analysis" in results["summary"]["sklearn_evaluation"]:
            speaker_analysis = results["summary"]["sklearn_evaluation"]["speaker_level_analysis"]
            print(f"\nTop 5 Speaker-level Analysis:")
            sorted_speaker_analysis = sorted(speaker_analysis.items(), 
                                           key=lambda x: x[1]['accuracy'], reverse=True)
            for speaker_id, analysis in sorted_speaker_analysis[:5]:
                print(f"  {speaker_id}: Accuracy={analysis['accuracy']:.4f}, "
                      f"Sample count={analysis['sample_count']}, "
                      f"Correct count={analysis['correct_count']}, "
                      f"Gender={analysis['true_gender']}")
    print(f"Average inference time: {final_stats['avg_total_time']:.4f}s (excluding first sample)")
    print(f"Average Prefill time: {final_stats['avg_prefill_time']:.4f}s")
    print(f"Average Decode time: {final_stats['avg_decode_time']:.4f}s")
    print(f"Average tokens/sec: {final_stats['avg_tokens_per_sec']:.2f}")

if __name__ == "__main__":
    main()