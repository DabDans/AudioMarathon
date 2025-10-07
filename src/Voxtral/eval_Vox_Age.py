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

# Update paths to standardized format
data_path_root = '/data/to/your/dataset/path'
result_dir = '/data/to/your/results/path'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/voxtral_vox_age_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = '/data/to/your/model/path'
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_concatenated_audio_dataset(root_dir, sample_limit=0):
    """Load dataset from concatenated_audio directory based on age_classification_task_meta.json"""

    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")

    for item in metadata:
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        if not os.path.exists(wav_path):
            print(f"Warning: File does not exist {wav_path}")
            continue

        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()
        speaker_age = item.get("speaker_age", 0)

        all_samples.append({
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "wav_path": wav_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "choice_c": item["choice_c"],
            "choice_d": item["choice_d"],
            "choice_e": item["choice_e"],
            "answer_gt": age_group,
            "task": "Speaker_Age_Classification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")

    if sample_limit > 0 and len(all_samples) > sample_limit:
        print(f"Applying sample limit: randomly selecting {sample_limit} from {len(all_samples)} samples")
        all_samples = random.sample(all_samples, sample_limit)
        print(f"Sample count after limit: {len(all_samples)}")

    age_group_counts = {}
    for sample in all_samples:
        group = sample["age_group"]
        age_group_counts[group] = age_group_counts.get(group, 0) + 1
    
    print("Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")

    random.shuffle(all_samples)
    return all_samples

def create_qa_prompt(doc):
    """Generate prompt for age detection task"""
    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")
    choice_e = doc.get("choice_e", "")

    prompt_text = f"""{question}

a. {choice_a}
b. {choice_b}
c. {choice_c}
d. {choice_d}
e. {choice_e}

Please listen to the audio and select the correct answer. Reply with only the letter (a, b, c, d, or e)."""
    return prompt_text

def extract_answer_choice(response, choices):
    """Extract answer choice from model response and return complete age group text"""
    if not response:
        return ""

    response = response.strip().lower()

    if response == 'a' or response.startswith('a.') or response.startswith('a)'):
        return choices.get("choice_a", "")
    if response == 'b' or response.startswith('b.') or response.startswith('b)'):
        return choices.get("choice_b", "")
    if response == 'c' or response.startswith('c.') or response.startswith('c)'):
        return choices.get("choice_c", "")
    if response == 'd' or response.startswith('d.') or response.startswith('d)'):
        return choices.get("choice_d", "")
    if response == 'e' or response.startswith('e.') or response.startswith('e)'):
        return choices.get("choice_e", "")

    import re
    match = re.search(r'\b([abcde])\b', response)
    if match:
        letter = match.group(1)
        if letter == 'a':
            return choices.get("choice_a", "")
        elif letter == 'b':
            return choices.get("choice_b", "")
        elif letter == 'c':
            return choices.get("choice_c", "")
        elif letter == 'd':
            return choices.get("choice_d", "")
        elif letter == 'e':
            return choices.get("choice_e", "")

    match = re.search(r'[(\[]?([abcde])[)\].]?', response)
    if match:
        letter = match.group(1)
        if letter == 'a':
            return choices.get("choice_a", "")
        elif letter == 'b':
            return choices.get("choice_b", "")
        elif letter == 'c':
            return choices.get("choice_c", "")
        elif letter == 'd':
            return choices.get("choice_d", "")
        elif letter == 'e':
            return choices.get("choice_e", "")

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

def calculate_vox_age_metrics(predictions, ground_truths):
    """Calculate classification metrics for Vox_Age task"""
    try:
        age_groups = ['Young Adult (18-30)', 'Early Career (31-40)', 'Mid Career (41-50)', 'Senior (51-70)', 'Elderly (71+)']
        valid_pairs = [(pred, gt) for pred, gt in zip(predictions, ground_truths) 
                       if pred and gt and pred in age_groups and gt in age_groups]
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
    print("Starting VoxCeleb age classification Voxtral model evaluation...")

    samples = load_concatenated_audio_dataset(data_path_root, sample_limit)
    if not samples:
        print("No valid sample data found, exiting...")
        return

    age_counts = {}
    for sample in samples:
        age_group = sample["age_group"]
        age_counts[age_group] = age_counts.get(age_group, 0) + 1
    
    print("Age group statistics:")
    for age_group, count in age_counts.items():
        print(f"  {age_group}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "age_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    for age_group in age_counts.keys():
        results["summary"]["age_stats"][age_group] = {"total": 0, "correct": 0}

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing Vox age audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, item in enumerate(samples):
            audio_path = item['wav_path']
            age_group = item['age_group']
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
                    choices = {
                        "choice_a": item["choice_a"],
                        "choice_b": item["choice_b"],
                        "choice_c": item["choice_c"],
                        "choice_d": item["choice_d"],
                        "choice_e": item["choice_e"]
                    }
                    predicted_choice = extract_answer_choice(output, choices)
                    is_correct = predicted_choice == ground_truth

                results["summary"]["total_samples"] += 1
                if age_group in results["summary"]["age_stats"]:
                    results["summary"]["age_stats"][age_group]["total"] += 1
                    if is_correct:
                        results["summary"]["age_stats"][age_group]["correct"] += 1
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
                "age_group": age_group,
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
    
    metrics = calculate_vox_age_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]

    for age_group, stats in results["summary"]["age_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        age_samples = [sample for sample in results["samples"] if sample.get("age_group") == age_group]
        if age_samples:
            age_preds = [sample["extracted_answer"] for sample in age_samples]
            age_gts = [sample["ground_truth"] for sample in age_samples]
            age_metrics = calculate_vox_age_metrics(age_preds, age_gts)
            stats["f1_score"] = age_metrics["f1_score"]
            stats["precision"] = age_metrics["precision"]
            stats["recall"] = age_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== VoxCeleb Age Classification Voxtral Model Evaluation Results Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")
    
    print("\nDetailed metrics by age group:")
    for age_group, stats in results["summary"]["age_stats"].items():
        print(f"  {age_group}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")
    
    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()