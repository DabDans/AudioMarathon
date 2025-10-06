import os
import json
import glob
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import sys
from collections import defaultdict
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def load_librispeech_for_analysis(base_dir, split="test-clean"):
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Error: Dataset path does not exist: {split_dir}")
        return [], {}
    
    print(f"Scanning LibriSpeech dataset directory: {split_dir}")
    
    speaker_stats = defaultdict(int)
    chapter_stats = defaultdict(int)
    missing_files = 0
    
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
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            transcription = f.read().strip()
                    except Exception as e:
                        print(f"Failed to read txt file {txt_file}: {e}")
                        continue
                elif os.path.exists(trans_file):
                    try:
                        with open(trans_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.startswith(base_name):
                                    parts = line.strip().split(' ', 1)
                                    if len(parts) > 1:
                                        transcription = parts[1]
                                    break
                    except Exception as e:
                        print(f"Failed to read trans file {trans_file}: {e}")
                        continue
                
                if transcription:
                    try:
                        audio_info = sf.info(flac_file)
                        
                        item = {
                            "path": flac_file,
                            "filename": os.path.basename(flac_file),
                            "audio": {
                                "path": flac_file,
                                "sampling_rate": audio_info.samplerate
                            },
                            "transcription": transcription,
                            "duration": audio_info.duration,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "language": "en",
                            "id": f"{speaker_id}_{chapter_id}_{base_name}",
                            "frames": audio_info.frames,
                            "channels": audio_info.channels
                        }
                        
                        dataset.append(item)
                        speaker_stats[speaker_id] += 1
                        chapter_stats[f"{speaker_id}-{chapter_id}"] += 1
                        
                    except Exception as e:
                        missing_files += 1
                        if missing_files <= 5:
                            print(f"Cannot process audio file {flac_file}: {e}")
                        continue
                else:
                    missing_files += 1
                    if missing_files <= 5:
                        print(f"Transcription file not found: {base_name}")
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} files cannot be processed")
    
    metadata_info = {
        "split": split,
        "total_samples": len(dataset),
        "speaker_count": len(speaker_stats),
        "chapter_count": len(chapter_stats),
        "speaker_distribution": dict(speaker_stats),
        "missing_files": missing_files
    }
    
    print(f"Loaded {len(dataset)} valid audio samples from {split}")
    print(f"Speaker count: {len(speaker_stats)}")
    print(f"Chapter count: {len(chapter_stats)}")
    
    return dataset, metadata_info

def analyze_audio_length(audio_path):
    try:
        info = sf.info(audio_path)
        duration = info.duration
        return duration
    except Exception as e1:
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e2:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                return duration
            except Exception as e3:
                print(f"Error processing audio file {audio_path}:")
                print(f"  Method 1 (soundfile.info): {e1}")
                print(f"  Method 2 (librosa.get_duration): {e2}")
                print(f"  Method 3 (librosa.load): {e3}")
                return None

def calculate_statistics(durations, name=""):
    if not durations:
        return {}
    
    durations_array = np.array(durations)
    
    stats = {
        "sample_count": len(durations),
        "mean_duration": float(np.mean(durations_array)),
        "median_duration": float(np.median(durations_array)),
        "min_duration": float(np.min(durations_array)),
        "max_duration": float(np.max(durations_array)),
        "std_duration": float(np.std(durations_array)),
        "total_duration": float(np.sum(durations_array)),
        "percentile_25": float(np.percentile(durations_array, 25)),
        "percentile_75": float(np.percentile(durations_array, 75)),
        "percentile_90": float(np.percentile(durations_array, 90)),
        "percentile_95": float(np.percentile(durations_array, 95)),
        "percentile_99": float(np.percentile(durations_array, 99))
    }
    
    return stats

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours ({seconds:.2f} seconds)"

def get_duration_distribution(durations):
    if not durations:
        return {}
    
    bins = [0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, float('inf')]
    bin_labels = [
        '0-2s', '2-5s', '5-10s', '10-15s', '15-20s', '20-30s',
        '30-45s', '45-60s', '1-1.5min', '1.5-2min', '2-3min', 
        '3-5min', '5min+'
    ]
    
    distribution = {}
    total_count = len(durations)
    
    for i, (start, end, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
        if end == float('inf'):
            count = sum(1 for d in durations if d >= start)
        else:
            count = sum(1 for d in durations if start <= d < end)
        
        percentage = count / total_count * 100 if total_count > 0 else 0
        distribution[label] = {
            "count": count,
            "percentage": percentage,
            "range": f"{start}-{end if end != float('inf') else '∞'} seconds"
        }
    
    return distribution

def detect_outliers(durations, method='iqr'):
    if not durations:
        return [], {}
    
    durations_array = np.array(durations)
    
    if method == 'iqr':
        Q1 = np.percentile(durations_array, 25)
        Q3 = np.percentile(durations_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = durations_array[(durations_array < lower_bound) | (durations_array > upper_bound)]
        outlier_info = {
            "method": "IQR",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR
        }
    
    return outliers.tolist(), outlier_info

def analyze_speaker_patterns(audio_files):
    speaker_durations = defaultdict(list)
    
    for audio_info in audio_files:
        if 'duration' in audio_info and audio_info['duration'] is not None:
            speaker_id = audio_info.get('speaker_id', 'unknown')
            speaker_durations[speaker_id].append(audio_info['duration'])
    
    speaker_stats = {}
    for speaker_id, durations in speaker_durations.items():
        if durations:
            speaker_stats[speaker_id] = {
                "sample_count": len(durations),
                "mean_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "std_duration": np.std(durations),
                "total_duration": np.sum(durations)
            }
    
    return speaker_stats

def main():
    librispeech_path = "/data/to/your/dataset/path//librispeech-long"
    split = os.environ.get("LIBRISPEECH_SPLIT", "test-clean")
    
    result_dir = './LibriSpeech_Analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== LibriSpeech Dataset Audio Length Analysis ===")
    print(f"Dataset path: {librispeech_path}")
    print(f"Analysis split: {split}")
    print(f"Result save path: {result_dir}")
    print()
    
    if not os.path.exists(librispeech_path):
        print(f"Error: LibriSpeech dataset path does not exist: {librispeech_path}")
        return
    
    print("Loading audio information from LibriSpeech dataset...")
    audio_files, metadata_info = load_librispeech_for_analysis(librispeech_path, split)
    
    if not audio_files:
        print("Error: No valid audio files found")
        return
    
    print()
    
    print("Analyzing audio length...")
    
    all_durations = []
    speaker_durations = defaultdict(list)
    chapter_durations = defaultdict(list)
    failed_files = []
    
    for audio_info in tqdm(audio_files, desc="Analyzing audio length", ascii=True):
        if 'duration' in audio_info and audio_info['duration'] is not None:
            duration = audio_info['duration']
        else:
            duration = analyze_audio_length(audio_info["path"])
        
        if duration is not None:
            all_durations.append(duration)
            
            speaker_id = audio_info.get("speaker_id", "unknown")
            speaker_durations[speaker_id].append(duration)
            
            chapter_id = f"{speaker_id}-{audio_info.get('chapter_id', 'unknown')}"
            chapter_durations[chapter_id].append(duration)
            
            audio_info["duration"] = duration
        else:
            failed_files.append({
                "filename": audio_info["filename"],
                "path": audio_info["path"],
                "speaker_id": audio_info.get("speaker_id", "unknown"),
                "reason": "Audio processing failed"
            })
    
    print()
    
    print("Calculating statistics...")
    
    overall_stats = calculate_statistics(all_durations, "overall")
    
    speaker_stats = {}
    for speaker_id, durations in speaker_durations.items():
        if durations:
            speaker_stats[speaker_id] = calculate_statistics(durations, speaker_id)
    
    duration_distribution = get_duration_distribution(all_durations)
    
    outliers, outlier_info = detect_outliers(all_durations)
    
    speaker_patterns = analyze_speaker_patterns(audio_files)
    
    analysis_results = {
        "dataset_info": {
            "librispeech_path": librispeech_path,
            "split": split,
            "metadata_info": metadata_info,
            "total_files_found": len(audio_files),
            "successfully_analyzed": len(all_durations),
            "failed_files": len(failed_files),
            "speakers": list(speaker_durations.keys()),
            "speaker_counts": {k: len(v) for k, v in speaker_durations.items()},
            "chapter_counts": {k: len(v) for k, v in chapter_durations.items()}
        },
        "overall_statistics": overall_stats,
        "speaker_statistics": speaker_stats,
        "speaker_patterns": speaker_patterns,
        "duration_distribution": duration_distribution,
        "outlier_analysis": {
            "outliers": outliers,
            "outlier_info": outlier_info,
            "outlier_count": len(outliers)
        },
        "failed_files": failed_files
    }
    
    output_file = os.path.join(result_dir, f'librispeech_{split}_audio_length_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("🎙️ LibriSpeech Dataset Audio Length Statistical Report")
    print("="*80)
    
    print(f"\n📁 Dataset Information:")
    print(f"   Dataset path: {librispeech_path}")
    print(f"   Analysis split: {split}")
    print(f"   Samples found: {len(audio_files)}")
    print(f"   Successfully analyzed: {len(all_durations)}")
    print(f"   Analysis failed: {len(failed_files)}")
    print(f"   Speaker count: {len(speaker_durations)}")
    print(f"   Chapter count: {len(chapter_durations)}")
    
    if overall_stats:
        print(f"\n🎙️ Overall Audio Length Statistics:")
        print(f"   Sample count: {overall_stats['sample_count']}")
        print(f"   Average length: {format_duration(overall_stats['mean_duration'])}")
        print(f"   Median length: {format_duration(overall_stats['median_duration'])}")
        print(f"   Shortest length: {format_duration(overall_stats['min_duration'])}")
        print(f"   Longest length: {format_duration(overall_stats['max_duration'])}")
        print(f"   Standard deviation: {overall_stats['std_duration']:.2f} seconds")
        print(f"   Total duration: {format_duration(overall_stats['total_duration'])}")
        print(f"   25th percentile: {format_duration(overall_stats['percentile_25'])}")
        print(f"   75th percentile: {format_duration(overall_stats['percentile_75'])}")
        print(f"   90th percentile: {format_duration(overall_stats['percentile_90'])}")
        print(f"   95th percentile: {format_duration(overall_stats['percentile_95'])}")
        print(f"   99th percentile: {format_duration(overall_stats['percentile_99'])}")
    
    if speaker_stats:
        print(f"\n👥 Speaker Audio Length Statistics (Top 10):")
        sorted_speakers = sorted(speaker_stats.items(), 
                               key=lambda x: x[1]['sample_count'], reverse=True)
        
        for i, (speaker_id, stats) in enumerate(sorted_speakers[:10]):
            print(f"\n  🗣️ Speaker {speaker_id}:")
            print(f"     Sample count: {stats['sample_count']}")
            print(f"     Average length: {format_duration(stats['mean_duration'])}")
            print(f"     Median length: {format_duration(stats['median_duration'])}")
            print(f"     Length range: {format_duration(stats['min_duration'])} - {format_duration(stats['max_duration'])}")
            print(f"     Total duration: {format_duration(stats['total_duration'])}")
    
    if duration_distribution:
        print(f"\n📈 Audio Length Distribution:")
        for range_label, dist_info in duration_distribution.items():
            count = dist_info['count']
            percentage = dist_info['percentage']
            if count > 0:
                print(f"   {range_label}: {count} files ({percentage:.1f}%)")
    
    if outliers:
        print(f"\n⚠️  Audio Length Outlier Analysis:")
        print(f"   Detected {len(outliers)} outliers")
        print(f"   Outlier range: {format_duration(min(outliers))} - {format_duration(max(outliers))}")
        print(f"   IQR bounds: {format_duration(outlier_info['lower_bound'])} - {format_duration(outlier_info['upper_bound'])}")
        
        outlier_samples = []
        for audio_info in audio_files:
            if audio_info.get('duration') in outliers:
                outlier_samples.append(audio_info)
        
        if outlier_samples:
            print(f"   Outlier samples (first 5):")
            for sample in outlier_samples[:5]:
                duration = sample.get('duration', 0)
                speaker_id = sample.get('speaker_id', 'unknown')
                filename = sample.get('filename', 'unknown')
                print(f"     - {filename} (Speaker: {speaker_id}): {format_duration(duration)}")
    
    if failed_files:
        print(f"\n❌ Failed Processing Files:")
        for failed_file in failed_files[:10]:
            print(f"   - {failed_file['filename']} (Speaker: {failed_file['speaker_id']}) - {failed_file['reason']}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more files")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*80)
    
    if all_durations:
        short_audios = [d for d in all_durations if d < 2]
        medium_audios = [d for d in all_durations if 2 <= d <= 20]
        long_audios = [d for d in all_durations if d > 60]
        
        print(f"\n📏 LibriSpeech Audio Length Analysis:")
        print(f"   Short audio (<2s): {len(short_audios)} files ({len(short_audios)/len(all_durations)*100:.1f}%)")
        if short_audios:
            print(f"     Average length: {format_duration(np.mean(short_audios))}")
        
        print(f"   Medium audio (2-20s): {len(medium_audios)} files ({len(medium_audios)/len(all_durations)*100:.1f}%)")
        if medium_audios:
            print(f"     Average length: {format_duration(np.mean(medium_audios))}")
            
        print(f"   Long audio (>60s): {len(long_audios)} files ({len(long_audios)/len(all_durations)*100:.1f}%)")
        if long_audios:
            print(f"     Average length: {format_duration(np.mean(long_audios))}")
    
    if overall_stats:
        quality_issues = []
        
        if overall_stats['min_duration'] < 0.5:
            quality_issues.append("Audio too short (<0.5s) exists")
        
        if overall_stats['max_duration'] > 600:
            quality_issues.append("Audio too long (>10min) exists")
        
        if overall_stats['std_duration'] > overall_stats['mean_duration']:
            quality_issues.append("Audio length distribution has high variance")
        
        if quality_issues:
            print(f"\n⚠️  Data Quality Warnings:")
            for issue in quality_issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ Good data quality, suitable for ASR model training and evaluation")

if __name__ == "__main__":
    main()