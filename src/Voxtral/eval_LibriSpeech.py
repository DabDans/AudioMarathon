import sys
import os
import numpy as np
import torch
import json
import time
import traceback
import glob
import random
import soundfile as sf
import librosa
import tempfile
import re
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import VoxtralForConditionalGeneration, AutoProcessor

model_name = "Voxtral-Mini-3B-2507"

data_path_root = '/data/to/your/librispeech_long/path'
result_dir = f'./Voxtral_LibriSpeech_Results_{model_name.replace("/", "_").replace("-", "_")}'
os.makedirs(result_dir, exist_ok=True)

sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
if sample_limit > 0:
    print(f"Sample limit set to: {sample_limit}")

output_file = f'{result_dir}/Voxtral_LibriSpeech_results_{model_name.replace("/", "_").replace("-", "_")}.json'
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

def load_librispeech_long_dataset(base_dir, split="test-clean"):
    """Load local LibriSpeech-Long dataset - based on Baichuan's correct implementation"""
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Error: Dataset path does not exist: {split_dir}")
        return []
    
    print(f"Scanning dataset directory: {split_dir}")

    speaker_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
    
    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)

        chapter_dirs = sorted([d for d in glob.glob(os.path.join(speaker_dir, "*")) if os.path.isdir(d)])
        
        for chapter_dir in chapter_dirs:
            chapter_id = os.path.basename(chapter_dir)

            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))
            
            for flac_file in flac_files:

                base_name = os.path.splitext(os.path.basename(flac_file))[0]

                txt_file = os.path.join(chapter_dir, f"{base_name}.txt")
                trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                
                transcription = None

                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()

                elif os.path.exists(trans_file):
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(base_name):

                                parts = line.strip().split(' ', 1)
                                if len(parts) > 1:
                                    transcription = parts[1]
                                break
                
                if transcription:
                    try:

                        try:

                            audio_data, sample_rate = librosa.load(flac_file, sr=None)
                            duration = len(audio_data) / sample_rate
                        except Exception:

                            file_size = os.path.getsize(flac_file)
                            duration = file_size / (16000 * 2)

                        item = {
                            "path": flac_file,
                            "audio": {
                                "path": flac_file,
                                "sampling_rate": 16000
                            },
                            "transcription": transcription,
                            "duration": duration,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "language": "en",
                            "id": f"{speaker_id}_{chapter_id}_{base_name}"
                        }
                        
                        dataset.append(item)
                        
                    except Exception as e:
                        print(f"Unable to process audio file {flac_file}: {e}")
                        continue
    
    print(f"Loaded {len(dataset)} audio samples")
    return dataset

def create_librispeech_asr_prompt():
    """Generate prompt for LibriSpeech ASR task - based on Baichuan's implementation"""

    instruction = "Transcribe this audio accurately. Remove hesitation words like 'um', 'uh'."
    format_text = "Your response should be formatted as follows: Spoken Content:"

    return f"{instruction} {format_text}"

def clean_asr_response(response):
    """Clean ASR response - based on Baichuan's implementation"""
    if not response or response.strip() == "":
        return ""

    for marker in ["spoken content:", "content:", "transcription:", "transcript:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break

    response = re.sub(r'<transcribed text here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    response = re.sub(r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription):\s*', '', response.strip())
    
    return response.strip()

def standardize_text(text):
    """Standardize text for fair comparison and WER calculation - consistent with API version"""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER) - fully consistent with API version"""
    try:

        ref_standardized = standardize_text(reference)
        hyp_standardized = standardize_text(hypothesis)
        
        if not ref_standardized or not hyp_standardized:
            return 1.0

        ref_words = ref_standardized.split()
        hyp_words = hyp_standardized.split()

        d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]
        
        for i in range(len(hyp_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(hyp_words) + 1):
            for j in range(1, len(ref_words) + 1):
                if hyp_words[i-1] == ref_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(hyp_words)][len(ref_words)] / len(ref_words) if len(ref_words) > 0 else 1.0
    except Exception as e:
        print(f"Error when calculating WER: {e}")
        return 1.0

def calculate_asr_metrics(predictions, ground_truths):
    """Calculate metrics for ASR task - based on Baichuan's implementation"""
    try:

        valid_pairs = [(pred, gt) for pred, gt in zip(predictions, ground_truths) 
                       if pred and gt and pred != "error"]
        
        if not valid_pairs:
            return {
                'total_wer': 0.0,
                'avg_wer': 1.0,
                'valid_samples': 0
            }
        
        valid_preds, valid_gts = zip(*valid_pairs)

        total_wer = 0.0
        for pred, gt in valid_pairs:
            wer = calculate_wer(gt, pred)
            total_wer += wer
        
        avg_wer = total_wer / len(valid_pairs)
        
        return {
            'total_wer': float(total_wer),
            'avg_wer': float(avg_wer),
            'valid_samples': len(valid_pairs)
        }
    except Exception as e:
        print(f"Error when calculating ASR metrics: {e}")
        return {
            'total_wer': 0.0,
            'avg_wer': 1.0,
            'valid_samples': 0
        }

def process_audio_with_voxtral(audio_path, question):
    """Process audio and question using Voxtral model"""
    try:

        inputs = processor.apply_transcription_request(
            language="en", 
            audio=audio_path, 
            model_id="/data/to/your/model/path"
        )
        inputs = inputs.to("cuda", dtype=torch.bfloat16)

        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False, temperature=0.0)
        response_time = time.time() - start_time

        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        output_text = decoded_outputs[0]

        truncation_indicators = [
            ' and ', ' but ', ' or ', ' the ', ' a ', ' an ',
            ' in ', ' on ', ' at ', ' to ', ' for ', ' of ', ' with '
        ]
        
        is_truncated = any(output_text.rstrip().endswith(indicator.strip()) for indicator in truncation_indicators)
        
        if is_truncated:
            print(f"Warning: Possible truncation detected, output length: {len(output_text.split())} words")
        
        return output_text, response_time
        
    except Exception as e:
        print(f"Voxtral model processing failed: {e}")
        return f"ERROR: {str(e)}", 0

def main():
    print("Starting LibriSpeech speech recognition Voxtral model evaluation...")

    print(f"Loading LibriSpeech dataset: {data_path_root}")
    samples = load_librispeech_long_dataset(data_path_root, "test-clean")
    
    if not samples:
        print("No valid sample data found, exiting...")
        return

    random.shuffle(samples)

    if sample_limit > 0 and len(samples) > sample_limit:
        samples = samples[:sample_limit]
        print(f"Applying sample limit, will process {len(samples)} samples")

    from collections import defaultdict
    speaker_stats = defaultdict(int)
    for sample in samples:
        speaker_id = sample.get("speaker_id", "unknown")
        speaker_stats[speaker_id] += 1
    
    print(f"Speaker statistics: {len(speaker_stats)} speakers")
    print(f"Sample distribution: {dict(list(speaker_stats.items())[:5])}...")

    results = {
        "samples": [],
        "summary": {
            "total_samples": 0,
            "total_wer": 0,
            "avg_wer": 0,
            "speaker_count": len(speaker_stats),
            "timing": {
                "avg_response_time": 0,
                "total_response_time": 0,
            }
        }
    }

    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")

    tqdm_kwargs = {
        'ascii': True,
        'dynamic_ncols': True,
        'file': sys.stdout
    }

    with tqdm(total=len(samples), desc="Processing LibriSpeech audio samples", position=0, leave=True, **tqdm_kwargs) as pbar:

        for i, item in enumerate(samples):
            audio_path = item['path']
            transcription = item['transcription']
            speaker_id = item.get('speaker_id', 'unknown')
            ground_truth = transcription

            output = ""
            predicted_transcription = ""
            wer = 1.0
            response_time = 0
            
            try:

                question = create_librispeech_asr_prompt()
                output, response_time = process_audio_with_voxtral(audio_path, question)

                if output.startswith("ERROR"):
                    print(f"Voxtral model processing failed: {output}")
                    predicted_transcription = "error"
                    wer = 1.0
                else:

                    predicted_transcription = clean_asr_response(output)

                    wer = calculate_wer(ground_truth, predicted_transcription)

                    if results["summary"]["total_samples"] < 3:
                        print(f"\nDebug info - sample {results['summary']['total_samples'] + 1}:")
                        print(f"  Raw output: {output}")
                        print(f"  Cleaned: {predicted_transcription}")
                        print(f"  Reference text: {ground_truth}")

                        ref_words = len(ground_truth.split())
                        hyp_words = len(predicted_transcription.split())
                        print(f"  Text length comparison - Reference: {ref_words} words, Prediction: {hyp_words} words")

                        ref_std = standardize_text(ground_truth)
                        hyp_std = standardize_text(predicted_transcription)
                        print(f"  Standardized reference: {ref_std[:100]}...")
                        print(f"  Standardized prediction: {hyp_std[:100]}...")
                        print(f"  WER: {wer:.4f}")

                results["summary"]["total_samples"] += 1
                results["summary"]["total_wer"] += wer

                results["summary"]["timing"]["total_response_time"] += response_time
                
            except Exception as e:
                print(f"Processing error: {e}")
                traceback.print_exc()
                output = "ERROR"
                predicted_transcription = "error"
                wer = 1.0
                response_time = 0

                results["summary"]["total_samples"] += 1
                results["summary"]["total_wer"] += wer
                results["summary"]["timing"]["total_response_time"] += response_time

            sample_result = {
                "audio_file": os.path.basename(audio_path),
                "speaker_id": speaker_id,
                "duration": item.get('duration', 0),
                "ground_truth": ground_truth,
                "model_output": output,
                "extracted_transcription": predicted_transcription,
                "wer": wer,
                "response_time": response_time
            }

            results["samples"].append(sample_result)

            update_interval = 10 if is_screen_env else 1
            sample_count = i + 1
            
            if sample_count % update_interval == 0 or sample_count == len(samples):

                current_avg_wer = results["summary"]["total_wer"] / results["summary"]["total_samples"] if results["summary"]["total_samples"] > 0 else 1.0

                pbar.set_postfix_str(
                    f"Average WER:{current_avg_wer:.3f}"
                )
                
                if is_screen_env:

                    print(f"  Progress: {sample_count}/{len(samples)} ({sample_count/len(samples)*100:.1f}%), "
                          f"Average WER: {current_avg_wer:.3f}")

            pbar.update()

            time.sleep(0.1)

    total_samples = results["summary"]["total_samples"]
    if total_samples > 0:
        results["summary"]["timing"]["avg_response_time"] = results["summary"]["timing"]["total_response_time"] / total_samples
        results["summary"]["avg_wer"] = results["summary"]["total_wer"] / total_samples

        failed_samples = sum(1 for result in results["samples"] if result["wer"] == 1.0)
        success_samples = total_samples - failed_samples
        results["summary"]["failed_samples"] = failed_samples
        results["summary"]["success_samples"] = success_samples
        results["summary"]["failure_rate"] = failed_samples / total_samples if total_samples > 0 else 0.0

    speaker_wer = defaultdict(list)
    for result in results["samples"]:
        speaker_id = result.get("speaker_id", "unknown")
        speaker_wer[speaker_id].append(result["wer"])
    
    speaker_avg_wer = {}
    for speaker_id, wers in speaker_wer.items():
        speaker_avg_wer[speaker_id] = sum(wers) / len(wers)

    results["summary"]["speaker_wer"] = speaker_avg_wer

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== LibriSpeech Speech Recognition Voxtral Model Evaluation Results Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Successful samples: {results['summary'].get('success_samples', 0)}")
    print(f"Failed samples: {results['summary'].get('failed_samples', 0)}")
    print(f"Failure rate: {results['summary'].get('failure_rate', 0):.2%}")
    print(f"Speaker count: {len(speaker_stats)}")
    print(f"Average Word Error Rate (WER): {results['summary']['avg_wer']:.4f}")
    print(f"Average response time: {results['summary']['timing']['avg_response_time']:.4f} seconds")
    
    print(f"\nTop 5 speakers WER:")
    for speaker_id, avg_wer in list(speaker_avg_wer.items())[:5]:
        sample_count = speaker_stats[speaker_id]
        print(f"  {speaker_id}: {avg_wer:.4f} ({sample_count} samples)")
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()