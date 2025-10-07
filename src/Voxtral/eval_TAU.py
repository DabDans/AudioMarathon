import sys
import os
import numpy as np
import torch
import json
import time
import traceback
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import glob

data_path_root = '/data/to/your/dataset/root'
audio_dir = '/data/to/your/audio/dir'
result_dir = f'/data/to/your/result/dir'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_TAU_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/model/dir"

    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_tau_acoustic_scene_dataset(root_dir):
    """Load acoustic scene classification task from TAU dataset"""

    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")

    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        return [], {}

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")

    scene_counts = {}

    for item in metadata:

        rel_path = item["path"]
        wav_path = os.path.join(root_dir, rel_path)

        if not os.path.exists(wav_path):
            print(f"Warning: File does not exist {wav_path}")
            continue

        scene_label = item["scene_label"]
        answer_gt = item["answer_gt"] # A, B, C, D

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

    print(f"Total loaded {len(all_samples)} valid audio samples")

    print("Scene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count} samples ({count/len(all_samples)*100:.1f}%)")

    random.shuffle(all_samples)

    return all_samples, scene_counts

def create_tau_prompt(item):
    """Generate prompt for TAU acoustic scene classification task"""

    instruction = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.\n"
    instruction += f"A: {item['choice_a']}\nB: {item['choice_b']}\nC: {item['choice_c']}\nD: {item['choice_d']}\n"
    instruction += "Respond with only the letter of your answer (A, B, C, or D)."
    
    return instruction

def extract_acoustic_scene_answer(text, choices=None):
    """Extract acoustic scene answer option (A/B/C/D) from model output text"""
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
                return chr(65 + i)  # A, B, C, D

            keywords = choice_lower.split(' - ')[0].split()
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        
        if best_match and max_overlap > 1:
            return best_match

    return ""

def process_audio_with_voxtral(audio_path, question):
    """Process audio and question using Voxtral model"""
    try:

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to("cuda", dtype=torch.bfloat16)

        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=200)
        response_time = time.time() - start_time

        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return decoded_outputs[0], response_time
        
    except Exception as e:
        print(f"Voxtral model processing failed: {e}")
        return f"ERROR: {str(e)}", 0

def calculate_tau_metrics(predictions, ground_truths):
    """Calculate metrics for TAU acoustic scene classification"""
    try:

        valid_pairs = [(pred, gt) for pred, gt in zip(predictions, ground_truths) 
                       if pred and gt and pred in ['A', 'B', 'C', 'D'] and gt in ['A', 'B', 'C', 'D']]
        
        if not valid_pairs:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'macro_f1': 0.0,
                'valid_samples': 0
            }
        
        valid_preds, valid_gts = zip(*valid_pairs)

        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_gts, valid_preds, average='weighted', zero_division=0
        )

        _, _, macro_f1, _ = precision_recall_fscore_support(
            valid_gts, valid_preds, average='macro', zero_division=0
        )

        accuracy = accuracy_score(valid_gts, valid_preds)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'macro_f1': float(macro_f1),
            'valid_samples': len(valid_pairs)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'macro_f1': 0.0,
            'valid_samples': 0
        }

def main():
    print("Starting TAU acoustic scene classification Voxtral model evaluation...")

    samples, scene_counts = load_tau_acoustic_scene_dataset(audio_dir)
    
    if not samples:
        print("No valid sample data found, exiting...")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

        limited_scene_counts = {}
        for sample in samples:
            scene = sample["scene_label"]
            limited_scene_counts[scene] = limited_scene_counts.get(scene, 0) + 1
        scene_counts = limited_scene_counts
    
    print("Acoustic scene statistics:")
    for scene, count in scene_counts.items():
        print(f"  {scene}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
            "valid_samples": 0,
            "scene_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    for scene in scene_counts.keys():
        results["summary"]["scene_stats"][scene] = {
            "total": 0, 
            "correct": 0, 
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing TAU audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:

        for i, item in enumerate(samples):
            audio_path = item['wav_path']
            scene = item['scene_label']
            question = create_tau_prompt(item)
            choices = [item['choice_a'], item['choice_b'], item['choice_c'], item['choice_d']]
            ground_truth = item["answer_gt"]

            output = ""
            predicted_scene = ""
            is_correct = False
            response_time = 0
            
            try:

                output, response_time = process_audio_with_voxtral(audio_path, question)

                if output.startswith("ERROR"):
                    print(f"Voxtral model processing failed: {output}")
                    predicted_scene = "error"
                    is_correct = False
                else:

                    predicted_scene = extract_acoustic_scene_answer(output, choices)

                    is_correct = predicted_scene == ground_truth

                results["summary"]["total_samples"] += 1
                if scene in results["summary"]["scene_stats"]:
                    results["summary"]["scene_stats"][scene]["total"] += 1
                    if is_correct:
                        results["summary"]["scene_stats"][scene]["correct"] += 1
                        results["summary"]["correct_samples"] += 1

                results["summary"]["timing"]["total_response_time"] += response_time
                
            except Exception as e:
                print(f"Processing error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_scene = "error"
                is_correct = False
                response_time = 0

            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "scene": scene,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_scene,
                "is_correct": is_correct,
                "response_time": response_time
            }

            results["samples"].append(sample_result)

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

            time.sleep(0.1)

    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_response_time"] = results["summary"]["timing"]["total_response_time"] / total_samples

    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0

    all_predictions = [sample["extracted_answer"] for sample in results["samples"]]
    all_ground_truths = [sample["ground_truth"] for sample in results["samples"]]

    metrics = calculate_tau_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]

    for scene, stats in results["summary"]["scene_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        scene_samples = [sample for sample in results["samples"] if sample.get("scene") == scene]
        if scene_samples:
            scene_preds = [sample["extracted_answer"] for sample in scene_samples]
            scene_gts = [sample["ground_truth"] for sample in scene_samples]
            scene_metrics = calculate_tau_metrics(scene_preds, scene_gts)
            stats["f1_score"] = scene_metrics["f1_score"]
            stats["precision"] = scene_metrics["precision"]
            stats["recall"] = scene_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== TAU Acoustic Scene Classification Voxtral Model Evaluation Results Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")
    
    print("\nDetailed metrics by scene:")
    for scene, stats in results["summary"]["scene_stats"].items():
        print(f"  {scene}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")
    
    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()