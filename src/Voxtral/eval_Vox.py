import sys
import os
import numpy as np
import torch
import json
import re
import time
import traceback
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import VoxtralForConditionalGeneration, AutoProcessor

qa_json_file = "/data/to/your/DESED_task_json/path/desed_sound_event_detection_task.json"
audio_base_dir = "/data/to/your/DESED_audio_base/path"
result_dir = '/data/to/your/model_results/path'

try:
    os.makedirs(result_dir, exist_ok=True)
    print(f"Result directory created/confirmed exists: {os.path.abspath(result_dir)}")
except Exception as e:
    print(f"Failed to create result directory: {e}")

    result_dir = os.path.join(os.getcwd(), 'model_results')
    os.makedirs(result_dir, exist_ok=True)
    print(f"Creating result directory using absolute path: {result_dir}")

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/model_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/model_repo/path"
    
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_desed_qa_dataset(json_file, audio_base_dir):
    """Load DESED QA dataset"""
    dataset = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []

    if not isinstance(data, dict) or 'tasks' not in data:
        print(f"Error: JSON file format incorrect, expected dict with 'tasks' field")
        return []
    
    tasks = data['tasks']
    print(f"Loaded {len(tasks)} tasks from JSON")
    
    task_type_stats = {}
    missing_files = 0
    
    for i, task in enumerate(tasks):

        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Missing audio path in task: {task}")
            continue

        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue

        task_type = task.get("task_type", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")

        choices = task.get("choices", {})
        choice_a = choices.get("A", "")
        choice_b = choices.get("B", "")
        choice_c = choices.get("C", "")
        choice_d = choices.get("D", "")

        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "audio": {
                "path": full_audio_path,
                "sampling_rate": 16000
            },
            "task_type": task_type,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "original_events": task.get("all_events", []),
            "all_events": task.get("all_events", []),
            "primary_event": task.get("primary_event", ""),
            "correct_event": task.get("correct_event", ""),
            "path_extracted_event": task.get("path_extracted_event", ""),
            "uniq_id": task.get("uniq_id", i),
            "id": f"qa_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type = item.get("task_type", "unknown")
        task_type_stats[task_type] = task_type_stats.get(task_type, 0) + 1
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files missing")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
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
        outputs = model.generate(**inputs, max_new_tokens=100)
        response_time = time.time() - start_time

        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return decoded_outputs[0], response_time
        
    except Exception as e:
        print(f"Voxtral model processing failed: {e}")
        return f"ERROR: {str(e)}", 0

def calculate_desed_metrics(predictions, ground_truths):
    """Calculate classification metrics for DESED task"""
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
    print("Starting Voxtral model evaluation for DESED sound event detection...")

    samples = load_desed_qa_dataset(qa_json_file, audio_base_dir)
    
    if not samples:
        print("No valid sample data found, exiting...")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

    task_type_counts = {}
    for sample in samples:
        task_type = sample.get("task_type", "unknown")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    print("Task type statistics:")
    for task_type, count in task_type_counts.items():
        print(f"  {task_type}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
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

    with tqdm(total=len(samples), desc="Processing DESED audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, item in enumerate(samples):
            audio_path = item['audio']['path']
            task_type = item.get('task_type', 'unknown')
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
                    predicted_choice = ""
                    is_correct = False
                else:
                    predicted_choice = extract_answer_choice(output)
                    is_correct = predicted_choice == ground_truth

                results["summary"]["total_samples"] += 1
                if task_type in results["summary"]["task_type_stats"]:
                    results["summary"]["task_type_stats"][task_type]["total"] += 1
                    if is_correct:
                        results["summary"]["task_type_stats"][task_type]["correct"] += 1
                        results["summary"]["correct_samples"] += 1

                results["summary"]["timing"]["total_response_time"] += response_time
                
            except Exception as e:
                print(f"Processing error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_choice = ""
                is_correct = False
                response_time = 0

            sample_result = {
                "audio_file": os.path.basename(audio_path),
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
                pbar.set_postfix_str(f"Accuracy:{current_accuracy:.2%}")
                
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
    
    metrics = calculate_desed_metrics(all_predictions, all_ground_truths)
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
            task_metrics = calculate_desed_metrics(task_preds, task_gts)
            stats["f1_score"] = task_metrics["f1_score"]
            stats["precision"] = task_metrics["precision"]
            stats["recall"] = task_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== DESED Sound Event Detection Voxtral Model Evaluation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
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