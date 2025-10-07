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

data_path_root = '/data/to/your/dataset/path'
result_dir = '/data/to/your/results/path'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_VESUS_results.json'
print(f"Results will be saved to: {output_file}")

def init_model():
    """Initialize Voxtral model"""
    device = "cuda"
    repo_id = "/data/to/your/model/path"
    
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, processor

model, processor = init_model()

def load_vesus_dataset(data_path_root):
    """Load VESUS speech emotion analysis dataset from concatenated_audio directory"""

    meta_file = os.path.join(data_path_root, "audio_emotion_dataset.json")
    
    all_samples = []
    missing_files = 0
    total_metadata = 0

    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        total_metadata = len(metadata)
        print(f"Loaded {total_metadata} sample metadata from {meta_file}")

        for item in metadata:

            rel_path = item["path"]
            wav_path = os.path.join(data_path_root, rel_path)

            if not os.path.exists(wav_path):
                missing_files += 1

                if missing_files <= 10:
                    print(f"Warning: File does not exist {wav_path}")
                elif missing_files == 11:
                    print(f"... More missing files, detailed display omitted ...")
                continue

            answer_gt = item["answer_gt"].strip()
            emotion_label = item.get("emotion_label", "unknown")

            all_samples.append({
                "emotion": emotion_label,
                "audio_path": wav_path,
                "question": item["question"],
                "choices": [item[f"choice_{chr(ord('a') + i)}"] for i in range(4)],
                "answer_gt": answer_gt,
                "task": "Speech_Emotion_Recognition"
            })
    else:
        print(f"Warning: Metadata file does not exist {meta_file}")
        return []
    
    print(f"Total loaded {len(all_samples)} valid audio samples")
    if missing_files > 0:
        print(f"Found {missing_files} missing audio files (total metadata: {total_metadata})")
        print(f"Available sample ratio: {len(all_samples)/total_metadata*100:.1f}%")

    emotion_counts = {}
    for sample in all_samples:
        group = sample["emotion"]
        emotion_counts[group] = emotion_counts.get(group, 0) + 1
    
    print("Emotion distribution:")
    for group, count in emotion_counts.items():
        print(f"  {group}: {count} samples")

    random.shuffle(all_samples)
    
    return all_samples

def create_qa_prompt(doc):
    """Generate prompt for VESUS speech emotion recognition task"""
    question = doc.get("question", "")
    choices = doc.get("choices", [])

    prompt_text = f"""{question}

"""

    for i, choice in enumerate(choices):
        if i < 4:
            letter = chr(ord('A') + i)
            prompt_text += f"{letter}. {choice}\n"
    
    prompt_text += "\nPlease listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."
    
    return prompt_text

def extract_answer_choice(response):
    """Extract answer choice (A-D) from model response"""
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

def calculate_vesus_metrics(predictions, ground_truths):
    """Calculate metrics for VESUS emotion recognition"""
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
    print("Starting VESUS speech emotion recognition Voxtral model evaluation...")

    samples = load_vesus_dataset(data_path_root)
    
    if not samples:
        print("No valid sample data found, exiting...")
        return

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

    emotion_counts = {}
    for sample in samples:
        emotion = sample["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("Emotion statistics:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} samples")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "correct_samples": 0,
            "emotion_stats": {},
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    for emotion in emotion_counts.keys():
        results["summary"]["emotion_stats"][emotion] = {"total": 0, "correct": 0}

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing VESUS audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:

        for i, item in enumerate(samples):
            audio_path = item['audio_path']
            emotion = item["emotion"]
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
                if emotion in results["summary"]["emotion_stats"]:
                    results["summary"]["emotion_stats"][emotion]["total"] += 1
                    if is_correct:
                        results["summary"]["emotion_stats"][emotion]["correct"] += 1
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
                "emotion": emotion,
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
    
    metrics = calculate_vesus_metrics(all_predictions, all_ground_truths)
    results["summary"]["f1_score"] = metrics["f1_score"]
    results["summary"]["precision"] = metrics["precision"]
    results["summary"]["recall"] = metrics["recall"]
    results["summary"]["macro_f1"] = metrics["macro_f1"]
    results["summary"]["valid_samples"] = metrics["valid_samples"]

    for emotion, stats in results["summary"]["emotion_stats"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        emotion_samples = [sample for sample in results["samples"] if sample.get("emotion") == emotion]
        if emotion_samples:
            emotion_preds = [sample["extracted_answer"] for sample in emotion_samples]
            emotion_gts = [sample["ground_truth"] for sample in emotion_samples]
            emotion_metrics = calculate_vesus_metrics(emotion_preds, emotion_gts)
            stats["f1_score"] = emotion_metrics["f1_score"]
            stats["precision"] = emotion_metrics["precision"]
            stats["recall"] = emotion_metrics["recall"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== VESUS Speech Emotion Recognition Voxtral Model Evaluation Results Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {results['summary']['valid_samples']}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.2%}")
    print(f"F1 score: {results['summary']['f1_score']:.4f}")
    print(f"Precision: {results['summary']['precision']:.4f}")
    print(f"Recall: {results['summary']['recall']:.4f}")
    print(f"Macro F1: {results['summary']['macro_f1']:.4f}")
    
    print("\nDetailed metrics by emotion type:")
    for emotion, stats in results["summary"]["emotion_stats"].items():
        print(f"  {emotion}:")
        print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if 'f1_score' in stats:
            print(f"    F1 score: {stats['f1_score']:.4f}")
            print(f"    Precision: {stats['precision']:.4f}")
            print(f"    Recall: {stats['recall']:.4f}")
    
    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()