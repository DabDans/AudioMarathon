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

warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze TAU dataset audio statistics")
    parser.add_argument("--data-path", type=str, 
                       default='/data/to/your/dataset/path/TAU',
                       help="TAU dataset root directory path")
    parser.add_argument("--audio-dir", type=str, default='concatenated_resampled',
                       help="Audio files directory name")
    parser.add_argument("--output-dir", type=str, default='./audio_analysis_results',
                       help="Results output directory")
    parser.add_argument("--save-plots", action='store_true',
                       help="Whether to save statistical plots")
    parser.add_argument("--target-sr", type=int, default=16000,
                       help="Target sample rate")
    return parser.parse_args()

def load_audio_metadata(root_dir):
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
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

def analyze_audio_statistics(metadata, audio_root_dir, target_sr=16000):
    
    audio_stats = []
    scene_stats = defaultdict(list)
    failed_files = []
    
    print("Starting audio file analysis...")
    
    for item in tqdm(metadata, desc="Processing audio files", ascii=True):
        if "path" in item:
            rel_path = item["path"]
            audio_path = os.path.join(audio_root_dir, rel_path)
        elif "audio_file" in item:
            audio_path = os.path.join(audio_root_dir, item["audio_file"])
        else:
            continue
        
        if not os.path.exists(audio_path):
            failed_files.append(audio_path)
            continue
        
        scene_label = item.get("scene_label", "unknown")
        
        duration = get_audio_duration_librosa(audio_path, target_sr)
        
        if duration is None:
            duration = get_audio_duration_soundfile(audio_path)
        
        if duration is not None:
            audio_info = {
                "file_path": audio_path,
                "file_name": os.path.basename(audio_path),
                "scene_label": scene_label,
                "duration_seconds": duration,
                "duration_minutes": duration / 60.0,
                "sample_id": item.get("id", f"sample_{len(audio_stats)}")
            }
            
            audio_stats.append(audio_info)
            scene_stats[scene_label].append(duration)
        else:
            failed_files.append(audio_path)
    
    return audio_stats, scene_stats, failed_files

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

def print_statistics_report(audio_stats, scene_stats, failed_files):
    
    if not audio_stats:
        print("Error: No successfully analyzed audio files")
        return
    
    all_durations = [item["duration_seconds"] for item in audio_stats]
    overall_stats = calculate_statistics(all_durations)
    
    print("\n" + "="*60)
    print("TAU Dataset Audio Statistical Analysis Report")
    print("="*60)
    
    print(f"\n[Overall Statistics]")
    print(f"Total samples: {overall_stats['count']:,}")
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
    
    print(f"\n[Statistics by Scene Category]")
    scene_summary = []
    for scene, durations in scene_stats.items():
        scene_stat = calculate_statistics(durations)
        scene_summary.append({
            "scene": scene,
            "count": scene_stat["count"],
            "mean_duration": scene_stat["mean"],
            "median_duration": scene_stat["median"],
            "min_duration": scene_stat["min"],
            "max_duration": scene_stat["max"]
        })
    
    scene_summary.sort(key=lambda x: x["count"], reverse=True)
    
    print(f"{'Scene Category':<20} {'Samples':<8} {'Avg Duration(s)':<15} {'Median(s)':<10} {'Min(s)':<8} {'Max(s)':<8}")
    print("-" * 80)
    for item in scene_summary:
        print(f"{item['scene']:<20} {item['count']:<8} {item['mean_duration']:<15.2f} "
              f"{item['median_duration']:<10.2f} {item['min_duration']:<8.2f} {item['max_duration']:<8.2f}")
    
    print(f"\n[Audio Duration Distribution]")
    duration_ranges = [
        (0, 10, "0-10 seconds"),
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
        print(f"\n[Files Failed to Analyze]")
        for i, failed_file in enumerate(failed_files[:10]):
            print(f"  {i+1}. {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more files")

def create_visualization_plots(audio_stats, scene_stats, output_dir):
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    all_durations = [item["duration_seconds"] for item in audio_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(all_durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Audio Duration (seconds)')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Audio Duration Distribution Histogram')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot(all_durations, vert=True)
    axes[0, 1].set_ylabel('Audio Duration (seconds)')
    axes[0, 1].set_title('Audio Duration Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    scene_names = list(scene_stats.keys())
    scene_means = [np.mean(scene_stats[scene]) for scene in scene_names]
    
    axes[1, 0].bar(range(len(scene_names)), scene_means, color='lightcoral')
    axes[1, 0].set_xticks(range(len(scene_names)))
    axes[1, 0].set_xticklabels(scene_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Average Audio Duration (seconds)')
    axes[1, 0].set_title('Average Audio Duration by Scene')
    axes[1, 0].grid(True, alpha=0.3)
    
    scene_counts = [len(scene_stats[scene]) for scene in scene_names]
    
    axes[1, 1].bar(range(len(scene_names)), scene_counts, color='lightgreen')
    axes[1, 1].set_xticks(range(len(scene_names)))
    axes[1, 1].set_xticklabels(scene_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Sample Count Distribution by Scene')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'tau_audio_statistics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Statistical plots saved to: {plot_file}")
    
    plt.show()

def save_detailed_results(audio_stats, scene_stats, failed_files, output_dir):
    
    df_audio = pd.DataFrame(audio_stats)
    csv_file = os.path.join(output_dir, 'tau_audio_detailed_stats.csv')
    df_audio.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Detailed audio information saved to: {csv_file}")
    
    all_durations = [item["duration_seconds"] for item in audio_stats]
    overall_stats = calculate_statistics(all_durations)
    
    scene_summary = {}
    for scene, durations in scene_stats.items():
        scene_summary[scene] = calculate_statistics(durations)
    
    summary_data = {
        "overall_statistics": overall_stats,
        "scene_statistics": scene_summary,
        "failed_files": failed_files,
        "analysis_summary": {
            "total_files_attempted": len(audio_stats) + len(failed_files),
            "successful_analysis": len(audio_stats),
            "failed_analysis": len(failed_files),
            "success_rate": len(audio_stats) / (len(audio_stats) + len(failed_files)) * 100 if (len(audio_stats) + len(failed_files)) > 0 else 0
        }
    }
    
    json_file = os.path.join(output_dir, 'tau_audio_summary_stats.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Summary statistics saved to: {json_file}")

def main():
    args = parse_arguments()
    
    data_root = "/data/to/your/dataset/path//TAU"
    audio_dir = os.path.join(data_root, 'concatenated_resampled')
    output_dir = "./TAU_audio_analysis"

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"TAU Dataset Audio Statistical Analysis")
    print(f"Data root directory: {data_root}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    if not os.path.exists(data_root):
        print(f"Error: Data root directory does not exist: {data_root}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory does not exist: {audio_dir}")
        return
    
    metadata = load_audio_metadata(data_root)
    if not metadata:
        print("Error: Cannot load metadata")
        return
    
    audio_stats, scene_stats, failed_files = analyze_audio_statistics(
        metadata, audio_dir, args.target_sr
    )
    
    print_statistics_report(audio_stats, scene_stats, failed_files)
    
    save_detailed_results(audio_stats, scene_stats, failed_files, output_dir)
    
    if args.save_plots and audio_stats:
        try:
            create_visualization_plots(audio_stats, scene_stats, output_dir)
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Skipping plot generation...")

if __name__ == "__main__":
    main()