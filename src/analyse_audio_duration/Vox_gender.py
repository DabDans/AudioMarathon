import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import random

warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze VoxCeleb dataset audio statistics")
    parser.add_argument("--data-path", type=str, 
                       default='/data/to/your/dataset/path/VoxCeleb/concatenated_audio',
                       help="VoxCeleb dataset root directory path")
    parser.add_argument("--audio-dir", type=str, default='wav',
                       help="Audio files directory name")
    parser.add_argument("--output-dir", type=str, default='./vox_audio_analysis_results',
                       help="Results output directory")
    parser.add_argument("--save-plots", action='store_true',
                       help="Whether to save statistical plots")
    parser.add_argument("--target-sr", type=int, default=16000,
                       help="Target sample rate")
    parser.add_argument("--sample-limit", type=int, default=0,
                       help="Limit number of samples to analyze, 0 means analyze all samples")
    return parser.parse_args()

def load_vox_metadata(root_dir):
    meta_file = os.path.join(root_dir, "gender_id_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Cannot find metadata file: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} sample metadata from {meta_file}")
    return metadata

def get_audio_duration_librosa(audio_path, target_sr=16000):
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Warning: Cannot read audio file {audio_path}: {e}")
        return None

def get_audio_duration_soundfile(audio_path):
    try:
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        return duration
    except Exception as e:
        print(f"Warning: Cannot read audio file {audio_path}: {e}")
        return None

def analyze_vox_audio_statistics(metadata, audio_root_dir, target_sr=16000, sample_limit=0):
    
    audio_stats = []
    gender_stats = defaultdict(list)
    speaker_stats = defaultdict(list)
    failed_files = []
    
    print("Starting VoxCeleb audio file analysis...")
    
    if sample_limit > 0 and sample_limit < len(metadata):
        print(f"Randomly sampling {sample_limit} samples for analysis")
        metadata = random.sample(metadata, sample_limit)
    
    for item in tqdm(metadata, desc="Processing audio files", ascii=True):
        rel_path = item["path"]
        audio_path = os.path.join(audio_root_dir, "wav", rel_path)
        
        if not os.path.exists(audio_path):
            failed_files.append(audio_path)
            continue
        
        speaker_id = item["speaker_id_original"]
        gender = item["answer_gt"].lower().strip()
        
        duration = get_audio_duration_librosa(audio_path, target_sr)
        
        if duration is None:
            duration = get_audio_duration_soundfile(audio_path)
        
        if duration is not None:
            audio_info = {
                "file_path": audio_path,
                "file_name": os.path.basename(audio_path),
                "relative_path": rel_path,
                "speaker_id": speaker_id,
                "gender": gender,
                "duration_seconds": duration,
                "duration_minutes": duration / 60.0,
                "sample_id": item.get("id", f"vox_{len(audio_stats)}"),
                "question": item.get("question", ""),
                "choice_a": item.get("choice_a", ""),
                "choice_b": item.get("choice_b", "")
            }
            
            audio_stats.append(audio_info)
            gender_stats[gender].append(duration)
            speaker_stats[speaker_id].append(duration)
        else:
            failed_files.append(audio_path)
    
    return audio_stats, gender_stats, speaker_stats, failed_files

def calculate_statistics(durations):
    if not durations:
        return {}
    
    durations_array = np.array(durations)
    
    stats = {
        "count": len(durations),
        "mean": np.mean(durations_array),
        "median": np.median(durations_array),
        "std": np.std(durations_array),
        "min": np.min(durations_array),
        "max": np.max(durations_array),
        "q25": np.percentile(durations_array, 25),
        "q75": np.percentile(durations_array, 75),
        "total_duration": np.sum(durations_array)
    }
    
    return stats

def print_vox_statistics_report(audio_stats, gender_stats, speaker_stats, failed_files):
    
    if not audio_stats:
        print("Error: No successfully analyzed audio files")
        return
    
    all_durations = [item["duration_seconds"] for item in audio_stats]
    overall_stats = calculate_statistics(all_durations)
    
    print("\n" + "="*70)
    print("VoxCeleb Dataset Audio Statistical Analysis Report")
    print("="*70)
    
    print(f"\n[Overall Statistics]")
    print(f"Total sample count: {overall_stats['count']:,}")
    print(f"Successfully analyzed: {len(audio_stats):,}")
    print(f"Analysis failed: {len(failed_files):,}")
    print(f"Success rate: {len(audio_stats)/(len(audio_stats)+len(failed_files))*100:.1f}%")
    
    print(f"\n[Audio Duration Statistics (seconds)]")
    print(f"Average duration: {overall_stats['mean']:.2f} seconds")
    print(f"Median duration: {overall_stats['median']:.2f} seconds")
    print(f"Shortest duration: {overall_stats['min']:.2f} seconds")
    print(f"Longest duration: {overall_stats['max']:.2f} seconds")
    print(f"Standard deviation: {overall_stats['std']:.2f} seconds")
    print(f"25% percentile: {overall_stats['q25']:.2f} seconds")
    print(f"75% percentile: {overall_stats['q75']:.2f} seconds")
    
    print(f"\n[Audio Duration Statistics (minutes)]")
    print(f"Average duration: {overall_stats['mean']/60:.2f} minutes")
    print(f"Median duration: {overall_stats['median']/60:.2f} minutes")
    print(f"Shortest duration: {overall_stats['min']/60:.2f} minutes")
    print(f"Longest duration: {overall_stats['max']/60:.2f} minutes")
    print(f"Total duration: {overall_stats['total_duration']/3600:.1f} hours")
    
    print(f"\n[Statistics by Gender]")
    gender_summary = []
    for gender, durations in gender_stats.items():
        gender_stat = calculate_statistics(durations)
        gender_summary.append({
            "gender": gender,
            "count": gender_stat["count"],
            "mean_duration": gender_stat["mean"],
            "median_duration": gender_stat["median"],
            "min_duration": gender_stat["min"],
            "max_duration": gender_stat["max"],
            "total_duration": gender_stat["total_duration"]
        })
    
    gender_summary.sort(key=lambda x: x["count"], reverse=True)
    
    print(f"{'Gender':<8} {'Samples':<8} {'Avg Duration(s)':<15} {'Median(s)':<10} {'Min(s)':<8} {'Max(s)':<8} {'Total(h)':<10}")
    print("-" * 80)
    for item in gender_summary:
        print(f"{item['gender']:<8} {item['count']:<8} {item['mean_duration']:<15.2f} "
              f"{item['median_duration']:<10.2f} {item['min_duration']:<8.2f} "
              f"{item['max_duration']:<8.2f} {item['total_duration']/3600:<10.2f}")
    
    print(f"\n[Speaker Statistics (Top 10)]")
    speaker_summary = []
    for speaker_id, durations in speaker_stats.items():
        speaker_stat = calculate_statistics(durations)
        speaker_summary.append({
            "speaker_id": speaker_id,
            "count": speaker_stat["count"],
            "mean_duration": speaker_stat["mean"],
            "total_duration": speaker_stat["total_duration"]
        })
    
    speaker_summary.sort(key=lambda x: x["count"], reverse=True)
    
    print(f"{'Speaker ID':<15} {'Samples':<8} {'Avg Duration(s)':<15} {'Total(h)':<10}")
    print("-" * 55)
    for item in speaker_summary[:10]:
        print(f"{item['speaker_id']:<15} {item['count']:<8} {item['mean_duration']:<15.2f} "
              f"{item['total_duration']/3600:<10.2f}")
    
    print(f"Total speakers: {len(speaker_stats)}")
    
    print(f"\n[Audio Duration Distribution]")
    duration_ranges = [
        (0, 5, "0-5 seconds"),
        (5, 10, "5-10 seconds"),
        (10, 30, "10-30 seconds"),
        (30, 60, "30-60 seconds"),
        (60, 120, "1-2 minutes"),
        (120, 300, "2-5 minutes"),
        (300, 600, "5-10 minutes"),
        (600, float('inf'), "10+ minutes")
    ]
    
    for min_dur, max_dur, label in duration_ranges:
        count = sum(1 for d in all_durations if min_dur <= d < max_dur)
        percentage = count / len(all_durations) * 100
        print(f"{label:<15}: {count:>6} samples ({percentage:>5.1f}%)")
    
    if failed_files:
        print(f"\n[Failed Analysis Files]")
        for i, failed_file in enumerate(failed_files[:10]):
            print(f"  {i+1}. {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more files")

def create_vox_visualization_plots(audio_stats, gender_stats, speaker_stats, output_dir):
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    all_durations = [item["duration_seconds"] for item in audio_stats]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].hist(all_durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Audio Duration (seconds)')
    axes[0, 0].set_ylabel('Sample Count')
    axes[0, 0].set_title('Audio Duration Distribution Histogram')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot(all_durations, vert=True)
    axes[0, 1].set_ylabel('Audio Duration (seconds)')
    axes[0, 1].set_title('Audio Duration Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    gender_names = list(gender_stats.keys())
    gender_durations = [gender_stats[gender] for gender in gender_names]
    
    axes[0, 2].boxplot(gender_durations, labels=gender_names)
    axes[0, 2].set_ylabel('Audio Duration (seconds)')
    axes[0, 2].set_title('Audio Duration Distribution by Gender')
    axes[0, 2].grid(True, alpha=0.3)
    
    gender_means = [np.mean(gender_stats[gender]) for gender in gender_names]
    
    axes[1, 0].bar(gender_names, gender_means, color=['lightcoral', 'lightblue'])
    axes[1, 0].set_ylabel('Average Audio Duration (seconds)')
    axes[1, 0].set_title('Average Audio Duration by Gender')
    axes[1, 0].grid(True, alpha=0.3)
    
    gender_counts = [len(gender_stats[gender]) for gender in gender_names]
    
    axes[1, 1].pie(gender_counts, labels=gender_names, autopct='%1.1f%%', 
                   colors=['lightcoral', 'lightblue'])
    axes[1, 1].set_title('Sample Count Distribution by Gender')
    
    speaker_counts = [(speaker, len(durations)) for speaker, durations in speaker_stats.items()]
    speaker_counts.sort(key=lambda x: x[1], reverse=True)
    
    top_speakers = speaker_counts[:20]
    speaker_names = [item[0][:8] + '...' if len(item[0]) > 8 else item[0] for item in top_speakers]
    speaker_sample_counts = [item[1] for item in top_speakers]
    
    axes[1, 2].bar(range(len(speaker_names)), speaker_sample_counts, color='lightgreen')
    axes[1, 2].set_xticks(range(len(speaker_names)))
    axes[1, 2].set_xticklabels(speaker_names, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Sample Count')
    axes[1, 2].set_title('Speaker Sample Count Distribution (Top 20)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'voxceleb_audio_statistics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Statistical plots saved to: {plot_file}")
    
    plt.show()

def save_vox_detailed_results(audio_stats, gender_stats, speaker_stats, failed_files, output_dir):
    df_audio = pd.DataFrame(audio_stats)
    csv_file = os.path.join(output_dir, 'voxceleb_audio_detailed_stats.csv')
    df_audio.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Detailed audio information saved to: {csv_file}")
    
    all_durations = [item["duration_seconds"] for item in audio_stats]
    overall_stats = calculate_statistics(all_durations)
    
    gender_summary = {}
    for gender, durations in gender_stats.items():
        gender_summary[gender] = calculate_statistics(durations)
    
    speaker_summary = {}
    for speaker_id, durations in speaker_stats.items():
        speaker_summary[speaker_id] = calculate_statistics(durations)
    
    summary_data = {
        "overall_statistics": overall_stats,
        "gender_statistics": gender_summary,
        "speaker_statistics": speaker_summary,
        "failed_files": failed_files,
        "analysis_summary": {
            "total_files_attempted": len(audio_stats) + len(failed_files),
            "successful_analysis": len(audio_stats),
            "failed_analysis": len(failed_files),
            "success_rate": len(audio_stats) / (len(audio_stats) + len(failed_files)) * 100 if (len(audio_stats) + len(failed_files)) > 0 else 0,
            "total_speakers": len(speaker_stats),
            "gender_distribution": {gender: len(durations) for gender, durations in gender_stats.items()}
        }
    }
    
    json_file = os.path.join(output_dir, 'voxceleb_audio_summary_stats.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Summary statistics saved to: {json_file}")

def main():
    args = parse_arguments()
    
    random.seed(42)
    
    data_root = "/data/to/your/dataset/path//VoxCeleb/concatenated_audio"
    audio_dir = "/data/to/your/dataset/path//VoxCeleb/concatenated_audio/wav"
    output_dir = "./Vox_gender_audio_analysis"

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"VoxCeleb Dataset Audio Statistical Analysis")
    print(f"Data root directory: {data_root}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    if args.sample_limit > 0:
        print(f"Sample limit: {args.sample_limit}")
    print("-" * 50)
    
    if not os.path.exists(data_root):
        print(f"Error: Data root directory does not exist: {data_root}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory does not exist: {audio_dir}")
        return
    
    metadata = load_vox_metadata(data_root)
    if not metadata:
        print("Error: Cannot load metadata")
        return
    
    audio_stats, gender_stats, speaker_stats, failed_files = analyze_vox_audio_statistics(
        metadata, data_root, args.target_sr, args.sample_limit
    )
    
    print_vox_statistics_report(audio_stats, gender_stats, speaker_stats, failed_files)
    
    save_vox_detailed_results(audio_stats, gender_stats, speaker_stats, failed_files, output_dir)
    
    if args.save_plots and audio_stats:
        try:
            create_vox_visualization_plots(audio_stats, gender_stats, speaker_stats, output_dir)
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Skipping plot generation...")

if __name__ == "__main__":
    main()