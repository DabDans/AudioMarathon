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

# Path definitions (updated to /data/to/your/xxx/path/ format)
data_path_root = "/data/to/your/had_dataset/path/"
result_dir = '/data/to/your/results/path/'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_HAD_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/model/path/"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_had_dataset(root_dir):
    """Load HAD dataset with balanced real/fake samples"""
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")

    all_samples = []

    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "real",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'.",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "a",
                "task": "Audio_Authenticity_Detection"
            })

    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for wav_path in fake_files:
            all_samples.append({
                "audio_path": wav_path,
                "label": "fake",
                "question": "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'.",
                "choice_a": "real",
                "choice_b": "fake",
                "answer_gt": "b",
                "task": "Audio_Authenticity_Detection"
            })

    print(f"Total loaded {len(all_samples)} audio samples")

    real_samples = [sample for sample in all_samples if sample["label"] == "real"]
    fake_samples = [sample for sample in all_samples if sample["label"] == "fake"]
    print(f"Original sample count: real={len(real_samples)}, fake={len(fake_samples)}")

    min_samples_per_category = min(len(real_samples), len(fake_samples))

    if len(real_samples) > min_samples_per_category:
        real_samples = random.sample(real_samples, min_samples_per_category)

    if len(fake_samples) > min_samples_per_category:
        fake_samples = random.sample(fake_samples, min_samples_per_category)

    balanced_samples = real_samples + fake_samples
    random.shuffle(balanced_samples)

    print(f"Balanced sample count: real={len(real_samples)}, fake={len(fake_samples)}, total={len(balanced_samples)}")

    return balanced_samples

def create_qa_prompt(doc):
    """Generate prompt for HAD audio authenticity detection task"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")

    prompt_text = f"""{question}

a. {choice_a}
b. {choice_b}

Please listen to the audio and select the correct answer. Reply with only the letter (a or b)."""
    return prompt_text

def extract_answer_choice(response):
    """Extract answer choice (a, b) from model response"""
    if not response:
        return ""

    response = response.strip().lower()

    if response in ['a', 'b']:
        return response

    match = re.search(r'\b([ab])\b', response)
    if match:
        return match.group(1)

    match = re.search(r'[(\[]?([ab])[)\].]?', response)
    if match:
        return match.group(1)

    response_lower = response.lower()
    if "real" in response_lower and "fake" not in response_lower:
        return "a"
    if "fake" in response_lower and "real" not in response_lower:
        return "b"

    return ""

def calculate_had_metrics(predictions, ground_truths):
    """Calculate metrics for HAD audio authenticity detection"""
    try:
        valid_pairs = [(pred, gt) for pred, gt in zip(predictions, ground_truths)
                       if pred and gt and pred in ['a', 'b'] and gt in ['a', 'b']]

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

def main():
    print("Start HAD audio authenticity detection Voxtral model evaluation...")

    samples = load_had_dataset(data_path_root)

    if not samples:
        print("No valid sample data found, exiting...")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

    task_type_counts = {}
    for sample in samples:
        task_type = sample.get("task", "unknown")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    print("Task type statistics:")
    for task_type, count in task_type_counts.items():
        print(f"  {task_type}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "real_total": 0,
            "real_correct": 0,
            "fake_total": 0,
            "fake_correct": 0,
            "task_type_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    for task_type in task_type_counts.keys():
        results["summary"]["task_type_stats"][task_type] = {"total": 0, "correct": 0}

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing HAD audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:

        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            label = item['label']
            task_type = item.get('task', 'unknown')
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
                if label in ["real", "fake"]:
                    results["summary"][f"{label}_total"] += 1
                    if is_correct:
                        results["summary"][f"{label}_correct"] += 1
                        results["summary"]["correct_samples"] += 1

                if task_type in results["summary"]["task_type_stats"]:
                    results["summary"]["task_type_stats"][task_type]["total"] += 1
                    if is_correct:
                        results["summary"]["task_type_stats"][task_type]["correct"] += 1

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
                "audio_label": label,
                "task_type": task_type,
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
    results["summary"]["real_accuracy"] = results["summary"]["real_correct"] / results["summary"]["real_total"] if results["summary"]["real_total"] > 0 else 0
    results["summary"]["fake_accuracy"] = results["summary"]["fake_correct"] / results["summary"]["fake_total"] if results["summary"]["fake_total"] > 0 else 0

    all_predictions = [sample["extracted_answer"] for sample in results["samples"]]
    all_ground_truths = [sample["ground_truth"] for sample in results["samples"]]

    metrics = calculate_had_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]

    for task_type, stats in results["summary"]["task_type_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        task_samples = [sample for sample in results["samples"] if sample.get("task_type") == task_type]
        if task_samples:
            task_preds = [sample["extracted_answer"] for sample in task_samples]
            task_gts = [sample["ground_truth"] for sample in task_samples]
            task_metrics = calculate_had_metrics(task_preds, task_gts)
            stats["f1_score"] = task_metrics["f1_score"]
            stats["precision"] = task_metrics["precision"]
            stats["recall"] = task_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== HAD Audio Authenticity Detection Voxtral Model Evaluation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"Real audio Accuracy: {results['summary']['real_accuracy']:.2%} ({results['summary']['real_correct']}/{results['summary']['real_total']})")
    print(f"Fake audio Accuracy: {results['summary']['fake_accuracy']:.2%} ({results['summary']['fake_correct']}/{results['summary']['fake_total']})")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")

    print("\nDetailed metrics by task type:")
    for task_type, stats in results["summary"]["task_type_stats"].items():
        print(f"  {task_type}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")

    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()