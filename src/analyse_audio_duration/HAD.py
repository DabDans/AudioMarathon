import os
import glob
import librosa
import numpy as np
import json
from tqdm import tqdm
import sys

def analyze_audio_length(audio_path):
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def load_had_dataset_for_analysis(root_dir):
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    
    all_audio_files = []
    
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            all_audio_files.append({
                "path": wav_path,
                "label": "real",
                "filename": os.path.basename(wav_path)
            })
        print(f"Found {len(real_files)} real audio files")
    else:
        print(f"Warning: Real audio directory does not exist: {real_dir}")
    
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for wav_path in fake_files:
            all_audio_files.append({
                "path": wav_path,
                "label": "fake", 
                "filename": os.path.basename(wav_path)
            })
        print(f"Found {len(fake_files)} fake audio files")
    else:
        print(f"Warning: Fake audio directory does not exist: {fake_dir}")
    
    print(f"Total found {len(all_audio_files)} audio files")
    return all_audio_files

def calculate_statistics(durations):
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
        "percentile_75": float(np.percentile(durations_array, 75))
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

def main():
    data_path_root = '/data/to/your/dataset/path//HAD/concatenated_audio'
    
    result_dir = './HAD_Analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== HAD Dataset Audio Length Analysis ===")
    print(f"Data path: {data_path_root}")
    print(f"Result save path: {result_dir}")
    print()
    
    if not os.path.exists(data_path_root):
        print(f"Error: Data path does not exist: {data_path_root}")
        return
    
    print("Scanning audio files...")
    audio_files = load_had_dataset_for_analysis(data_path_root)
    
    if not audio_files:
        print("Error: No audio files found")
        return
    
    print()
    
    print("Analyzing audio length...")
    
    all_durations = []
    real_durations = []
    fake_durations = []
    failed_files = []
    
    for audio_info in tqdm(audio_files, desc="Analyzing audio length", ascii=True):
        duration = analyze_audio_length(audio_info["path"])
        
        if duration is not None:
            all_durations.append(duration)
            
            if audio_info["label"] == "real":
                real_durations.append(duration)
            else:
                fake_durations.append(duration)
        else:
            failed_files.append(audio_info)
    
    print()
    
    print("Calculating statistics...")
    
    overall_stats = calculate_statistics(all_durations)
    
    real_stats = calculate_statistics(real_durations)
    
    fake_stats = calculate_statistics(fake_durations)
    
    analysis_results = {
        "dataset_info": {
            "data_path": data_path_root,
            "total_files_found": len(audio_files),
            "successfully_analyzed": len(all_durations),
            "failed_files": len(failed_files),
            "real_files": len(real_durations),
            "fake_files": len(fake_durations)
        },
        "overall_statistics": overall_stats,
        "real_audio_statistics": real_stats,
        "fake_audio_statistics": fake_stats,
        "failed_files": [f["filename"] for f in failed_files]
    }
    
    output_file = os.path.join(result_dir, 'had_audio_length_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("📊 HAD Dataset Audio Length Statistics Report")
    print("="*60)
    
    print(f"\n📁 Dataset Information:")
    print(f"   Data path: {data_path_root}")
    print(f"   Files found: {len(audio_files)}")
    print(f"   Successfully analyzed: {len(all_durations)}")
    print(f"   Analysis failed: {len(failed_files)}")
    print(f"   Real audio: {len(real_durations)}")
    print(f"   Fake audio: {len(fake_durations)}")
    
    if overall_stats:
        print(f"\n🎵 Overall Audio Length Statistics:")
        print(f"   Sample count: {overall_stats['sample_count']}")
        print(f"   Mean length: {format_duration(overall_stats['mean_duration'])}")
        print(f"   Median length: {format_duration(overall_stats['median_duration'])}")
        print(f"   Min length: {format_duration(overall_stats['min_duration'])}")
        print(f"   Max length: {format_duration(overall_stats['max_duration'])}")
        print(f"   Standard deviation: {overall_stats['std_duration']:.2f} seconds")
        print(f"   Total duration: {format_duration(overall_stats['total_duration'])}")
        print(f"   25th percentile: {format_duration(overall_stats['percentile_25'])}")
        print(f"   75th percentile: {format_duration(overall_stats['percentile_75'])}")
    
    if real_stats:
        print(f"\n✅ Real Audio Length Statistics:")
        print(f"   Sample count: {real_stats['sample_count']}")
        print(f"   Mean length: {format_duration(real_stats['mean_duration'])}")
        print(f"   Median length: {format_duration(real_stats['median_duration'])}")
        print(f"   Min length: {format_duration(real_stats['min_duration'])}")
        print(f"   Max length: {format_duration(real_stats['max_duration'])}")
        print(f"   Total duration: {format_duration(real_stats['total_duration'])}")
    
    if fake_stats:
        print(f"\n❌ Fake Audio Length Statistics:")
        print(f"   Sample count: {fake_stats['sample_count']}")
        print(f"   Mean length: {format_duration(fake_stats['mean_duration'])}")
        print(f"   Median length: {format_duration(fake_stats['median_duration'])}")
        print(f"   Min length: {format_duration(fake_stats['min_duration'])}")
        print(f"   Max length: {format_duration(fake_stats['max_duration'])}")
        print(f"   Total duration: {format_duration(fake_stats['total_duration'])}")
    
    if all_durations:
        print(f"\n📈 Audio Length Distribution:")
        
        bins = [0, 10, 30, 60, 120, 300, 600, float('inf')]
        bin_labels = ['0-10s', '10-30s', '30-60s', '1-2min', '2-5min', '5-10min', '10min+']
        
        for i, (start, end, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
            if end == float('inf'):
                count = sum(1 for d in all_durations if d >= start)
            else:
                count = sum(1 for d in all_durations if start <= d < end)
            
            percentage = count / len(all_durations) * 100
            print(f"   {label}: {count} files ({percentage:.1f}%)")
    
    if failed_files:
        print(f"\n⚠️  Failed processing files:")
        for failed_file in failed_files[:10]:
            print(f"   - {failed_file['filename']}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more files")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()