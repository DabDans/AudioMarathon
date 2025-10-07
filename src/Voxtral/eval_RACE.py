import sys
import os
import numpy as np
import torch
import json
import time
import traceback
import glob
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import re # Added for extract_race_answer

data_path_root = '/data/to/your/Dataset/path'
result_dir = '/data/to/your/Result/path'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_RACE_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/Model/path"
    
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_race_benchmark(root_dir, sample_limit=0):
    """Load RACE reading comprehension dataset from race_audio directory based on race_benchmark.json"""

    bench_path = os.path.join(root_dir, "race_benchmark.json")
    
    if not os.path.exists(bench_path):
        print(f"Error: Benchmark file does not exist: {bench_path}")
        return []
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    print(f"Loaded {len(benchmark)} samples metadata from {bench_path}")
    
    all_samples = []
    
    for item in benchmark:
        audio_rel = item["audio_path"]
        audio_full = os.path.join(root_dir, audio_rel)
        
        if not os.path.exists(audio_full):
            print(f"Warning: File does not exist {audio_full}")
            continue
        
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        
        all_samples.append({
            "audio_path": audio_full,
            "question": question,
            "options": options,
            "answer_gt": answer,
            "article_id": item.get("article_id", ""),
            "question_idx": item.get("question_idx", 0),
            "subset": "high" if "high" in audio_rel else "middle",
            "task": "Reading_Comprehension"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    
    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Applying sample limit: randomly selecting the first {sample_limit} out of {len(all_samples)} samples")
        all_samples = all_samples[:sample_limit]
        print(f"Sample count after limit: {len(all_samples)}")
    
    subset_counts = {}
    for sample in all_samples:
        subset = sample["subset"]
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    print("Subset distribution:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count} samples")
    
    return all_samples

def create_race_prompt(question, options):
    """Generate prompt for RACE reading comprehension task"""
    instruction = "Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."
    format_text = "Respond with only the letter of the correct option (A, B, C, or D)."
    
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options += f"{letter}. {opt}\n"
    
    prompt = f"{instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_options.strip()}\n\n{format_text}"
    
    return prompt

def extract_race_answer(text):
    """Extract RACE answer from model output text"""
    if not text:
        return ""
    
    text_upper = text.strip().upper()
    
    for ch in text_upper:
        if ch in ["A", "B", "C", "D"]:
            return ch
    
    patterns = [
        r'ANSWER\s*IS\s*([ABCD])',
        r'CHOOSE\s*([ABCD])',
        r'OPTION\s*([ABCD])',
        r'([ABCD])\s*OPTION',
        r'([ABCD])\s*IS\s*CORRECT',
        r'CORRECT\s*ANSWER\s*IS\s*([ABCD])',
        r'I\s*CHOOSE\s*([ABCD])',
        r'([ABCD])\s*CORRECT',
        r'([ABCD])\s*RIGHT',
        r'([ABCD])\s*✓',
        r'([ABCD])\s*√'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)
    
    words = text_upper.split()
    for word in words:
        if len(word) == 1 and word in ["A", "B", "C", "D"]:
            return word
        elif len(word) > 1 and any(ch in word for ch in ["A", "B", "C", "D"]):
            for ch in word:
                if ch in ["A", "B", "C", "D"]:
                    return ch
    
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

def calculate_race_metrics(predictions, ground_truths):
    """Calculate metrics for RACE reading comprehension"""
    try:
        valid_pairs = [(pred, gt) for pred, gt in zip(predictions, ground_truths) 
                       if pred and gt and pred in ['A', 'B', 'C', 'D'] and gt in ['A', 'B', 'C', 'D']]
        
        if not valid_pairs:
            print("Warning: No valid prediction-ground truth pairs")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'macro_f1': 0.0,
                'valid_samples': 0
            }
        
        valid_preds, valid_gts = zip(*valid_pairs)
        
        print(f"Valid samples: {len(valid_pairs)}")
        print(f"Predictions distribution: {dict(zip(*np.unique(valid_preds, return_counts=True)))}")
        print(f"Ground truth distribution: {dict(zip(*np.unique(valid_gts, return_counts=True)))}")
        
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            valid_gts, valid_preds, average=None, labels=['A', 'B', 'C', 'D'], zero_division=0
        )
        
        print("Class-wise metrics:")
        for i, label in enumerate(['A', 'B', 'C', 'D']):
            print(f"  {label}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, Support={support_per_class[i]}")
        
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
        traceback.print_exc()
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'macro_f1': 0.0,
            'valid_samples': 0
        }

def main():
    print("Starting RACE reading comprehension Voxtral model evaluation...")
    
    samples = load_race_benchmark(data_path_root, sample_limit)
    
    if not samples:
        print("No valid sample data found, exiting...")
        return
    
    subset_counts = {}
    for sample in samples:
        subset = sample.get("subset", "unknown")
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    print("Subset statistics:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count} samples")
    
    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "subset_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }
    
    for subset in subset_counts.keys():
        results["summary"]["subset_stats"][subset] = {"total": 0, "correct": 0}
    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }
    
    with tqdm(total=len(samples), desc="Processing RACE audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            subset = item.get('subset', 'unknown')
            question = item["question"]
            options = item["options"]
            ground_truth = item["answer_gt"]
            
            prompt_text = create_race_prompt(question, options)
            
            output = ""
            predicted_choice = ""
            is_correct = False
            response_time = 0
            
            try:
                output, response_time = process_audio_with_voxtral(audio_path, prompt_text)

                if output.startswith("ERROR"):
                    print(f"Voxtral model processing failed: {output}")
                    predicted_choice = "error"
                    is_correct = False
                else:
                    predicted_choice = extract_race_answer(output)
                    is_correct = predicted_choice == ground_truth

                    if not predicted_choice:
                        print(f"Warning: Could not extract answer from output: '{output[:100]}...'")
                    elif predicted_choice not in ['A', 'B', 'C', 'D']:
                        print(f"Warning: Invalid extracted answer: '{predicted_choice}', original output: '{output[:100]}...'")

                results["summary"]["total_samples"] += 1
                if subset in results["summary"]["subset_stats"]:
                    results["summary"]["subset_stats"][subset]["total"] += 1
                    if is_correct:
                        results["summary"]["subset_stats"][subset]["correct"] += 1
                        results["summary"]["correct_samples"] += 1

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
                "subset": subset,
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_answer": predicted_choice,
                "is_correct": is_correct,
                "response_time": response_time,
                "article_id": item.get("article_id", ""),
                "question_idx": item.get("question_idx", i)
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
    
    metrics = calculate_race_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]
    
    for subset, stats in results["summary"]["subset_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        subset_samples = [sample for sample in results["samples"] if sample.get("subset") == subset]
        if subset_samples:
            subset_preds = [sample["extracted_answer"] for sample in subset_samples]
            subset_gts = [sample["ground_truth"] for sample in subset_samples]
            subset_metrics = calculate_race_metrics(subset_preds, subset_gts)
            stats["f1_score"] = subset_metrics["f1_score"]
            stats["precision"] = subset_metrics["precision"]
            stats["recall"] = subset_metrics["recall"]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n=== RACE Reading Comprehension Voxtral Model Evaluation Results Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")
    
    print("\nDetailed metrics by subset:")
    for subset, stats in results["summary"]["subset_stats"].items():
        print(f"  {subset}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")
    
    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()