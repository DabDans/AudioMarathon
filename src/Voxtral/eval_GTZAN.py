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

# Path replacements
data_path_root = '/data/to/your/GTZAN/Dataset/path/concatenated_audio'
result_dir = '/data/to/your/GTZAN/Result/path'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_GTZAN_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/Voxtral/Model/path"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_gtzan_dataset(root_dir):
    """Load GTZAN music genre classification dataset from concatenated_audio directory"""

    meta_file = os.path.join(root_dir, "music_genre_classification_meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")

    for item in metadata:
        if not all(key in item for key in ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]):
            continue

        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)

        if not os.path.exists(wav_path):
            print(f"Warning: File does not exist {wav_path}")
            continue

        genre_label = item.get("genre_label", "unknown")
        answer_letter = item["answer_gt"].strip()

        all_samples.append({
            "genre": genre_label,
            "audio_path": wav_path,
            "question": item["question"],
            "choices": [item["choice_a"], item["choice_b"], item["choice_c"], item["choice_d"]],
            "answer_gt": answer_letter,
            "task": "Music_Genre_Classification"
        })
    
    print(f"Total loaded {len(all_samples)} valid audio samples")

    genre_counts = {}
    for sample in all_samples:
        group = sample["genre"]
        genre_counts[group] = genre_counts.get(group, 0) + 1
    
    print("Music genre distribution:")
    for group, count in genre_counts.items():
        print(f"  {group}: {count}")

    random.shuffle(all_samples)
    return all_samples

def create_qa_prompt(doc):
    """Generate prompt for GTZAN music genre classification task"""
    question = doc.get("question", "")
    choices = doc.get("choices", [])

    formatted_options = ""
    for i, opt in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D...
        formatted_options += f"{letter}. {opt}\n"
    
    prompt_text = f"""{question}

{formatted_options.strip()}

Please listen to the audio and identify the music genre. Reply with only the letter (A, B, C, or D)."""
    
    return prompt_text

def extract_genre_answer(text, choices):
    """Extract music genre answer from model output text"""
    if not text:
        return ""
    
    text_lower = text.lower().strip()

    genre_map = {
        'a': choices[0], 'b': choices[1], 'c': choices[2], 'd': choices[3]
    }

    import re
    answer_match = re.search(r'the answer is ([abcd])', text_lower)
    if answer_match:
        return answer_match.group(1).upper()

    answer_match = re.search(r'answer:\s*([abcd])', text_lower)
    if answer_match:
        return answer_match.group(1).upper()

    for letter, genre in genre_map.items():
        if text_lower == letter or text_lower.startswith(f'{letter}.') or text_lower.startswith(f'{letter})'):
            return letter.upper()

    for letter, genre in genre_map.items():
        if f"option {letter}" in text_lower or f"choice {letter}" in text_lower or f"{letter})" in text_lower:
            return letter.upper()

    for letter, genre in genre_map.items():
        if genre.lower() in text_lower:
            return letter.upper()

    for letter, genre in genre_map.items():
        genre_name = genre.split(' - ')[0].lower() if ' - ' in genre else genre.lower()
        if genre_name in text_lower:
            return letter.upper()

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

def calculate_gtzan_metrics(predictions, ground_truths):
    """Calculate metrics for GTZAN music genre classification"""
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
    print("Start GTZAN music genre classification Voxtral model evaluation...")

    samples = load_gtzan_dataset(data_path_root)
    
    if not samples:
        print("No valid sample data found, exiting...")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

    genre_counts = {}
    for sample in samples:
        genre = sample.get("genre", "unknown")
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    print("Music genre statistics:")
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "genre_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    for genre in genre_counts.keys():
        results["summary"]["genre_stats"][genre] = {"total": 0, "correct": 0}

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing GTZAN audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            genre = item.get('genre', 'unknown')
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
                    predicted_choice = extract_genre_answer(output, item['choices'])
                    is_correct = predicted_choice == ground_truth

                results["summary"]["total_samples"] += 1
                if genre in results["summary"]["genre_stats"]:
                    results["summary"]["genre_stats"][genre]["total"] += 1
                    if is_correct:
                        results["summary"]["genre_stats"][genre]["correct"] += 1
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
                "genre_label": genre,
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

    all_predictions = [sample["extracted_answer"] for sample in results["samples"]]
    all_ground_truths = [sample["ground_truth"] for sample in results["samples"]]
    
    metrics = calculate_gtzan_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]

    for genre, stats in results["summary"]["genre_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        genre_samples = [sample for sample in results["samples"] if sample.get("genre_label") == genre]
        if genre_samples:
            genre_preds = [sample["extracted_answer"] for sample in genre_samples]
            genre_gts = [sample["ground_truth"] for sample in genre_samples]
            genre_metrics = calculate_gtzan_metrics(genre_preds, genre_gts)
            stats["f1_score"] = genre_metrics["f1_score"]
            stats["precision"] = genre_metrics["precision"]
            stats["recall"] = genre_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== GTZAN Music Genre Classification Voxtral Model Evaluation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall Accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")
    
    print("\nDetailed metrics by music genre:")
    for genre, stats in results["summary"]["genre_stats"].items():
        print(f"  {genre}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")
    
    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()