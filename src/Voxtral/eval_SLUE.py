import sys
import os
import numpy as np
import torch
import json
import time
import traceback
import glob
import random
import re
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from collections import defaultdict

qa_json_file = "/data/to/your/SLUE/Dataset/merged_audio_data.json"
audio_base_dir = "/data/to/your/SLUE/Dataset"
result_dir = f'/data/to/your/Voxtral_SLUE_Results'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_SLUE_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/Model/Voxtral-Mini-3B-2507"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_slue_dataset(json_file, audio_base_dir):
    """
    Load SLUE task data from a JSON file

    Args:
        json_file: Path to SLUE-format JSON task file
        audio_base_dir: Base directory for audio files

    Returns:
        dataset: List containing task data
    """
    dataset = []

    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []

    print(f"Loading SLUE JSON file: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []

    if not isinstance(data, list):
        print(f"Error: JSON file format incorrect, expected list")
        return []

    print(f"Loaded {len(data)} tasks from JSON")

    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0

    for i, task in enumerate(data):
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task missing audio path: {task}")
            continue

        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue

        task_name = task.get("task_name", "unknown")
        dataset_name = task.get("dataset_name", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")

        choice_a = task.get("choice_a", "")
        choice_b = task.get("choice_b", "")
        choice_c = task.get("choice_c", "")
        choice_d = task.get("choice_d", "")

        try:
            try:
                import soundfile as sf
                audio_info = sf.info(full_audio_path)
                duration = audio_info.duration
                sample_rate = audio_info.samplerate
            except ImportError:
                file_size = os.path.getsize(full_audio_path)
                duration = file_size / (16000 * 2)
                sample_rate = 16000
        except Exception as e:
            print(f"Unable to read audio file info {full_audio_path}: {e}")
            continue

        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": sample_rate
            },
            "task_name": task_name,
            "dataset_name": dataset_name,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "entity_count": task.get("entity_count", 0),
            "entity_types": task.get("entity_types", []),
            "source_count": task.get("source_count", 0),
            "audio_duration_info": task.get("audio_duration_info", ""),
            "source_folder": task.get("source_folder", ""),
            "source_file": task.get("source_file", ""),
            "duration": duration,
            "uniq_id": task.get("uniq_id", i),
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }

        dataset.append(item)
        task_type_stats[task_name] += 1
        dataset_stats[dataset_name] += 1

    if missing_files > 5:
        print(f"Warning: A total of {missing_files} audio files do not exist")

    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")
    return dataset

def create_qa_prompt(doc):
    """Generate prompt for sound event detection task"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")

    prompt_text = f"""{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."""

    return prompt_text

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

def extract_answer_choice(response):
    """Extract answer choice (A, B, C, D) from model response"""
    if not response:
        return ""

    response = response.strip().upper()

    if response in ['A', 'B', 'C', 'D']:
        return response

    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)

    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)

    return ""

def calculate_slue_metrics(predictions, ground_truths):
    """Calculate classification metrics for SLUE task"""
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
            valid_gts, valid_preds, average='macro', zero_division=0
        )

        macro_f1 = f1

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
    print("Starting SLUE speech understanding Voxtral model evaluation...")

    samples = load_slue_dataset(qa_json_file, audio_base_dir)

    if not samples:
        print("No valid sample data found, exiting...")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

    task_type_counts = {}
    dataset_counts = {}
    for sample in samples:
        task_name = sample.get("task_name", "unknown")
        dataset_name = sample.get("dataset_name", "unknown")
        task_type_counts[task_name] = task_type_counts.get(task_name, 0) + 1
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

    print("Task type statistics:")
    for task_name, count in task_type_counts.items():
        print(f"  {task_name}: {count} samples")

    print("Dataset statistics:")
    for dataset_name, count in dataset_counts.items():
        print(f"  {dataset_name}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "task_type_stats": {},
            "dataset_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    for task_name in task_type_counts.keys():
        results["summary"]["task_type_stats"][task_name] = {"total": 0, "correct": 0}

    for dataset_name in dataset_counts.keys():
        results["summary"]["dataset_stats"][dataset_name] = {"total": 0, "correct": 0}

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing SLUE audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, item in enumerate(samples):
            audio_path = item['audio']['path']
            task_name = item.get('task_name', 'unknown')
            dataset_name = item.get('dataset_name', 'unknown')
            question = create_qa_prompt(item)
            ground_truth = item["answer_gt"]

            output = ""
            predicted_choice = ""
            is_correct = False
            response_time = 0

            try:
                output, response_time = process_audio_with_voxtral(audio_path, question)

                if output.startswith("ERROR"):
                    print(f"Voxtral model processing failed: {output}")
                    predicted_choice = "error"
                    is_correct = False
                else:
                    predicted_choice = extract_answer_choice(output)
                    is_correct = predicted_choice == ground_truth

                results["summary"]["total_samples"] += 1
                if task_name in results["summary"]["task_type_stats"]:
                    results["summary"]["task_type_stats"][task_name]["total"] += 1
                    if is_correct:
                        results["summary"]["task_type_stats"][task_name]["correct"] += 1
                        results["summary"]["correct_samples"] += 1

                if dataset_name in results["summary"]["dataset_stats"]:
                    results["summary"]["dataset_stats"][dataset_name]["total"] += 1
                    if is_correct:
                        results["summary"]["dataset_stats"][dataset_name]["correct"] += 1

                results["summary"]["timing"]["total_response_time"] += response_time

            except Exception as e:
                print(f"Processing error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_choice = "error"
                is_correct = False
                response_time = 0

            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "task_name": task_name,
                "dataset_name": dataset_name,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_choice,
                "is_correct": is_correct,
                "response_time": response_time
            }

            results["samples"].append(sample_result)

            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1

            if sample_count % update_interval == 0 or sample_count == len(samples):
                current_accuracy = results["summary"]["correct_samples"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 0

                pbar.set_postfix_str(
                    f"Accuracy: {current_accuracy:.2%} | Task: {task_name[:10]} | Dataset: {dataset_name[:10]}"
                )

                if is_screen_env:
                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Accuracy: {current_accuracy:.2%}, Task: {task_name[:15]}, Dataset: {dataset_name[:15]}")

            pbar.update()
            time.sleep(0.1)

    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_response_time"] = results["summary"]["timing"]["total_response_time"] / total_samples

    results["summary"]["accuracy"] = results["summary"]["correct_samples"] / total_samples if total_samples > 0 else 0

    all_predictions = [sample["extracted_answer"] for sample in results["samples"]]
    all_ground_truths = [sample["ground_truth"] for sample in results["samples"]]

    metrics = calculate_slue_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]

    for task_name, stats in results["summary"]["task_type_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        task_samples = [sample for sample in results["samples"] if sample.get("task_name") == task_name]
        if task_samples:
            task_preds = [sample["extracted_answer"] for sample in task_samples]
            task_gts = [sample["ground_truth"] for sample in task_samples]
            task_metrics = calculate_slue_metrics(task_preds, task_gts)
            stats["f1_score"] = task_metrics["f1_score"]
            stats["precision"] = task_metrics["precision"]
            stats["recall"] = task_metrics["recall"]

    for dataset_name, stats in results["summary"]["dataset_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        dataset_samples = [sample for sample in results["samples"] if sample.get("dataset_name") == dataset_name]
        if dataset_samples:
            dataset_preds = [sample["extracted_answer"] for sample in dataset_samples]
            dataset_gts = [sample["ground_truth"] for sample in dataset_samples]
            dataset_metrics = calculate_slue_metrics(dataset_preds, dataset_gts)
            stats["f1_score"] = dataset_metrics["f1_score"]
            stats["precision"] = dataset_metrics["precision"]
            stats["recall"] = dataset_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== SLUE Speech Understanding Voxtral Model Evaluation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")

    print("\nDetailed metrics by task type:")
    for task_name, stats in results["summary"]["task_type_stats"].items():
        print(f"  {task_name}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")

    print("\nDetailed metrics by dataset:")
    for dataset_name, stats in results["summary"]["dataset_stats"].items():
        print(f"  {dataset_name}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")

    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()