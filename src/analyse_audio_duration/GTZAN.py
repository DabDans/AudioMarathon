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

def load_gtzan_metadata_for_analysis(metadata_path):
    dataset = []
    metadata_info = {}
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file does not exist: {metadata_path}")
        return [], {}
    
    print(f"Loading GTZAN metadata file: {metadata_path}")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read metadata file: {e}")
        return [], {}
    
    if not isinstance(data, list):
        print(f"Error: Metadata file format is incorrect, expected list format")
        return [], {}
    
    print(f"Loaded {len(data)} music samples from metadata")
    
    genre_stats = defaultdict(int)
    missing_files = 0
    
    for i, sample in enumerate(data):
        required_fields = ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]
        if not all(field in sample for field in required_fields):
            print(f"Warning: Sample {i} missing required fields, skipping")
            continue
        
        genre_label = sample.get("genre_label", "unknown")
        
        item = {
            "path": sample["path"],
            "filename": os.path.basename(sample["path"]),
            "genre_label": genre_label,
            "uniq_id": sample.get("uniq_id", i),
            "question": sample.get("question", ""),
            "choice_a": sample.get("choice_a", ""),
            "choice_b": sample.get("choice_b", ""),
            "choice_c": sample.get("choice_c", ""),
            "choice_d": sample.get("choice_d", ""),
            "answer_gt": sample.get("answer_gt", "")
        }
        
        dataset.append(item)
        genre_stats[genre_label] += 1
    
    metadata_info = {
        "total_samples": len(dataset),
        "genre_distribution": dict(genre_stats),
        "genres": list(genre_stats.keys())
    }
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Genre statistics: {dict(genre_stats)}")
    return dataset, metadata_info

def analyze_audio_length(audio_path):
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e1:
        try:
            info = sf.info(audio_path)
            duration = info.duration
            return duration
        except Exception as e2:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                return duration
            except Exception as e3:
                print(f"Error processing audio file {audio_path}:")
                print(f"  Method 1 (librosa.get_duration): {e1}")
                print(f"  Method 2 (soundfile.info): {e2}")
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
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min ({seconds:.2f}s)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h ({seconds:.2f}s)"

def get_duration_distribution(durations):
    if not durations:
        return {}
    
    bins = [0, 10, 20, 25, 28, 30, 32, 35, 40, 45, 60, 90, 120, 180, float('inf')]
    bin_labels = [
        '0-10s', '10-20s', '20-25s', '25-28s', '28-30s',
        '30-32s', '32-35s', '35-40s', '40-45s', '45-60s',
        '1-1.5min', '1.5-2min', '2-3min', '3min+'
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
            "range": f"{start}-{end if end != float('inf') else '∞'}s"
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

def main():
    data_path_root = "/data/to/your/dataset/path//GTZAN/concatenated_audio/wav"
    metadata_file = "/data/to/your/dataset/path//GTZAN/concatenated_audio/music_genre_classification_meta.json"
    result_dir = './GTZAN_Analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== GTZAN Dataset Audio Length Analysis ===")
    print(f"Data directory: {data_path_root}")
    print(f"Metadata file: {metadata_file}")
    print(f"Results save path: {result_dir}")
    print()
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file does not exist: {metadata_file}")
        return
    
    if not os.path.exists(data_path_root):
        print(f"Error: Data directory does not exist: {data_path_root}")
        return
    
    print("Loading audio information from metadata file...")
    audio_files, metadata_info = load_gtzan_metadata_for_analysis(metadata_file)
    
    if not audio_files:
        print("Error: No valid audio files found")
        return
    
    print()
    
    print("Analyzing audio lengths...")
    
    all_durations = []
    genre_durations = defaultdict(list)
    failed_files = []
    
    for audio_info in tqdm(audio_files, desc="Analyzing audio lengths", ascii=True):
        audio_rel_path = audio_info["path"]
        audio_full_path = os.path.join(data_path_root, audio_rel_path)
        
        if not os.path.exists(audio_full_path):
            failed_files.append({
                "filename": audio_info["filename"],
                "path": audio_full_path,
                "genre": audio_info.get("genre_label", "unknown"),
                "reason": "File does not exist"
            })
            continue
        
        duration = analyze_audio_length(audio_full_path)
        
        if duration is not None:
            all_durations.append(duration)
            genre = audio_info.get("genre_label", "unknown")
            genre_durations[genre].append(duration)
            
            audio_info["duration"] = duration
            audio_info["full_path"] = audio_full_path
        else:
            failed_files.append({
                "filename": audio_info["filename"],
                "path": audio_full_path,
                "genre": audio_info.get("genre_label", "unknown"),
                "reason": "Audio processing failed"
            })
    
    print()
    
    print("Calculating statistics...")
    
    overall_stats = calculate_statistics(all_durations, "overall")
    
    genre_stats = {}
    for genre, durations in genre_durations.items():
        if durations:
            genre_stats[genre] = calculate_statistics(durations, genre)
    
    duration_distribution = get_duration_distribution(all_durations)
    
    outliers, outlier_info = detect_outliers(all_durations)
    
    analysis_results = {
        "dataset_info": {
            "data_path": data_path_root,
            "metadata_file": metadata_file,
            "metadata_info": metadata_info,
            "total_files_from_metadata": len(audio_files),
            "successfully_analyzed": len(all_durations),
            "failed_files": len(failed_files),
            "genres": list(genre_durations.keys()),
            "genre_counts": {k: len(v) for k, v in genre_durations.items()}
        },
        "overall_statistics": overall_stats,
        "genre_statistics": genre_stats,
        "duration_distribution": duration_distribution,
        "outlier_analysis": {
            "outliers": outliers,
            "outlier_info": outlier_info,
            "outlier_count": len(outliers)
        },
        "failed_files": failed_files
    }
    
    output_file = os.path.join(result_dir, 'gtzan_audio_length_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("🎵 GTZAN Dataset Audio Length Statistics Report")
    print("="*70)
    
    print(f"\n📁 Dataset Information:")
    print(f"   Data directory: {data_path_root}")
    print(f"   Metadata file: {metadata_file}")
    print(f"   Samples in metadata: {len(audio_files)}")
    print(f"   Successfully analyzed: {len(all_durations)}")
    print(f"   Analysis failed: {len(failed_files)}")
    print(f"   Number of genres: {len(genre_durations)}")
    
    if overall_stats:
        print(f"\n🎵 Overall Audio Length Statistics:")
        print(f"   Sample count: {overall_stats['sample_count']}")
        print(f"   Average length: {format_duration(overall_stats['mean_duration'])}")
        print(f"   Median length: {format_duration(overall_stats['median_duration'])}")
        print(f"   Shortest length: {format_duration(overall_stats['min_duration'])}")
        print(f"   Longest length: {format_duration(overall_stats['max_duration'])}")
        print(f"   Standard deviation: {overall_stats['std_duration']:.2f}s")
        print(f"   Total duration: {format_duration(overall_stats['total_duration'])}")
        print(f"   25th percentile: {format_duration(overall_stats['percentile_25'])}")
        print(f"   75th percentile: {format_duration(overall_stats['percentile_75'])}")
        print(f"   90th percentile: {format_duration(overall_stats['percentile_90'])}")
        print(f"   95th percentile: {format_duration(overall_stats['percentile_95'])}")
        print(f"   99th percentile: {format_duration(overall_stats['percentile_99'])}")
    
    if genre_stats:
        print(f"\n🎼 Audio Length Statistics by Genre:")
        for genre, stats in genre_stats.items():
            print(f"\n  🎶 {genre.upper()}:")
            print(f"     Sample count: {stats['sample_count']}")
            print(f"     Average length: {format_duration(stats['mean_duration'])}")
            print(f"     Median length: {format_duration(stats['median_duration'])}")
            print(f"     Shortest length: {format_duration(stats['min_duration'])}")
            print(f"     Longest length: {format_duration(stats['max_duration'])}")
            print(f"     Total duration: {format_duration(stats['total_duration'])}")
    
    if duration_distribution:
        print(f"\n📈 Audio Length Distribution:")
        for range_label, dist_info in duration_distribution.items():
            count = dist_info['count']
            percentage = dist_info['percentage']
            if count > 0:
                print(f"   {range_label}: {count} files ({percentage:.1f}%)")
    
    print(f"\n🏷️  Genre Distribution:")
    genre_counts = analysis_results["dataset_info"]["genre_counts"]
    total_samples = sum(genre_counts.values())
    for genre, count in genre_counts.items():
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"   {genre}: {count} samples ({percentage:.1f}%)")
    
    if outliers:
        print(f"\n⚠️  Audio Length Outlier Analysis:")
        print(f"   Detected {len(outliers)} outliers")
        print(f"   Outlier range: {min(outliers):.2f}s - {max(outliers):.2f}s")
        print(f"   IQR bounds: {outlier_info['lower_bound']:.2f}s - {outlier_info['upper_bound']:.2f}s")
        
        outlier_samples = []
        for audio_info in audio_files:
            if hasattr(audio_info, 'duration') and audio_info.get('duration') in outliers:
                outlier_samples.append(audio_info)
        
        if outlier_samples:
            print(f"   Outlier samples (first 5):")
            for sample in outlier_samples[:5]:
                duration = sample.get('duration', 0)
                genre = sample.get('genre_label', 'unknown')
                filename = sample.get('filename', 'unknown')
                print(f"     - {filename} ({genre}): {format_duration(duration)}")
    
    if failed_files:
        print(f"\n❌ Failed Files:")
        for failed_file in failed_files[:10]:
            print(f"   - {failed_file['filename']} ({failed_file['genre']}) - {failed_file['reason']}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more files")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*70)
    
    if all_durations:
        target_duration = 30.0
        tolerance = 0.5
        
        standard_length = [d for d in all_durations if abs(d - target_duration) <= tolerance]
        non_standard = [d for d in all_durations if abs(d - target_duration) > tolerance]
        
        print(f"\n📏 GTZAN Standard Length Analysis (30s±{tolerance}s):")
        print(f"   Standard length audio: {len(standard_length)} files ({len(standard_length)/len(all_durations)*100:.1f}%)")
        print(f"   Non-standard length audio: {len(non_standard)} files ({len(non_standard)/len(all_durations)*100:.1f}%)")
        
        if standard_length:
            print(f"   Standard length average: {format_duration(np.mean(standard_length))}")
        if non_standard:
            print(f"   Non-standard length range: {format_duration(min(non_standard))} - {format_duration(max(non_standard))}")

if __name__ == "__main__":
    main()