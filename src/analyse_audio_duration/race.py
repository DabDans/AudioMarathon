import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def get_audio_duration(audio_path, target_sr=16000):
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Failed to get audio duration: {audio_path}, error: {e}")
        return 0

def load_race_dataset_for_analysis(data_path_root):
    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: Cannot find benchmark file: {bench_path}")
        return []
    
    print(f"Loading RACE benchmark file: {bench_path}")
    
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    
    all_samples = []
    missing_files = 0
    audio_info_errors = 0
    
    print(f"Loaded {len(benchmark_data)} sample information from benchmark")
    
    for i, sample in tqdm(enumerate(benchmark_data), total=len(benchmark_data), desc="Loading audio file information"):
        audio_rel = sample["audio_path"]
        audio_full = os.path.join(data_path_root, audio_rel)
        
        if not os.path.exists(audio_full):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: File does not exist {audio_full}")
            continue
        
        try:
            audio_info = sf.info(audio_full)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
            frames = audio_info.frames
            channels = audio_info.channels
            
            file_size = os.path.getsize(audio_full)
            
            librosa_duration = get_audio_duration(audio_full)
            
        except Exception as e:
            audio_info_errors += 1
            if audio_info_errors <= 5:
                print(f"Warning: Cannot get audio information {audio_full}: {e}")
            continue
        
        difficulty = "unknown"
        if "high" in audio_rel.lower():
            difficulty = "high"
        elif "middle" in audio_rel.lower():
            difficulty = "middle"
        
        filename = os.path.basename(audio_full)
        relative_dir = os.path.dirname(audio_rel)
        
        sample_data = {
            "id": sample.get("id", f"race_sample_{i}"),
            "audio_path": audio_full,
            "relative_path": audio_rel,
            "relative_dir": relative_dir,
            "filename": filename,
            "difficulty": difficulty,
            "duration": duration,
            "librosa_duration": librosa_duration,
            "sample_rate": sample_rate,
            "frames": frames,
            "channels": channels,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "question": sample.get("question", ""),
            "options": sample.get("options", []),
            "answer": sample.get("answer", ""),
            "task": "Reading_Comprehension"
        }
        
        all_samples.append(sample_data)
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    if audio_info_errors > 5:
        print(f"Warning: Total {audio_info_errors} audio files failed to get information")
    
    print(f"Successfully loaded {len(all_samples)} valid audio samples")
    
    difficulty_counts = defaultdict(int)
    for sample in all_samples:
        difficulty_counts[sample["difficulty"]] += 1
    
    print(f"Difficulty level distribution:")
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty}: {count} samples")
    
    return all_samples

def calculate_audio_statistics(dataset):
    if not dataset:
        print("Error: No valid audio data")
        return None
    
    durations = [item["duration"] for item in dataset]
    librosa_durations = [item["librosa_duration"] for item in dataset if item["librosa_duration"] > 0]
    sample_rates = [item["sample_rate"] for item in dataset]
    file_sizes_mb = [item["file_size_mb"] for item in dataset]
    channels_list = [item["channels"] for item in dataset]
    
    stats = {
        "sample_count": len(dataset),
        "average_audio_length_seconds": np.mean(durations),
        "shortest_audio_length_seconds": np.min(durations),
        "longest_audio_length_seconds": np.max(durations),
        "audio_length_median_seconds": np.median(durations),
        "audio_length_std_seconds": np.std(durations),
        "audio_length_25th_percentile_seconds": np.percentile(durations, 25),
        "audio_length_75th_percentile_seconds": np.percentile(durations, 75),
        "audio_length_95th_percentile_seconds": np.percentile(durations, 95),
        "audio_length_99th_percentile_seconds": np.percentile(durations, 99),
        "total_audio_duration_hours": np.sum(durations) / 3600,
        "average_sample_rate_hz": np.mean(sample_rates),
        "sample_rate_types": list(set(sample_rates)),
        "average_file_size_mb": np.mean(file_sizes_mb),
        "total_file_size_gb": np.sum(file_sizes_mb) / 1024,
        "channel_types": list(set(channels_list)),
        "duration_validation_difference": {
            "librosa_sample_count": len(librosa_durations),
            "average_difference_seconds": np.mean([abs(d1 - d2) for d1, d2 in zip(durations, librosa_durations)]) if librosa_durations else 0
        }
    }
    
    return stats, durations, sample_rates, file_sizes_mb

def analyze_by_difficulty(dataset):
    difficulty_stats = defaultdict(list)
    
    for item in dataset:
        difficulty = item["difficulty"]
        difficulty_stats[difficulty].append(item["duration"])
    
    difficulty_analysis = {}
    for difficulty, durations in difficulty_stats.items():
        difficulty_analysis[difficulty] = {
            "sample_count": len(durations),
            "average_length_seconds": np.mean(durations),
            "shortest_length_seconds": np.min(durations),
            "longest_length_seconds": np.max(durations),
            "median_seconds": np.median(durations),
            "std_seconds": np.std(durations),
            "total_duration_hours": np.sum(durations) / 3600,
            "average_length_minutes": np.mean(durations) / 60
        }
    
    return difficulty_analysis

def analyze_duration_distribution(durations):
    distribution = {
        "< 30 seconds": sum(1 for d in durations if d < 30),
        "30 seconds - 1 minute": sum(1 for d in durations if 30 <= d < 60),
        "1-2 minutes": sum(1 for d in durations if 60 <= d < 120),
        "2-3 minutes": sum(1 for d in durations if 120 <= d < 180),
        "3-5 minutes": sum(1 for d in durations if 180 <= d < 300),
        "5-10 minutes": sum(1 for d in durations if 300 <= d < 600),
        "> 10 minutes": sum(1 for d in durations if d >= 600)
    }
    
    total_samples = len(durations)
    distribution_percentage = {
        range_name: {
            "count": count,
            "percentage": count / total_samples * 100 if total_samples > 0 else 0
        }
        for range_name, count in distribution.items()
    }
    
    return distribution_percentage

def plot_race_audio_distribution(durations, difficulty_analysis, output_dir="./race_audio_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RACE Dataset Audio Statistics Analysis', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Audio Duration Distribution Histogram')
    axes[0, 0].set_xlabel('Audio Duration (seconds)')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].grid(True, alpha=0.3)
    
    sorted_durations = np.sort(durations)
    y_vals = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    axes[0, 1].plot(sorted_durations, y_vals, linewidth=2, color='coral')
    axes[0, 1].set_title('Audio Duration Cumulative Distribution Function')
    axes[0, 1].set_xlabel('Audio Duration (seconds)')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].boxplot(durations, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[0, 2].set_title('Audio Duration Box Plot')
    axes[0, 2].set_ylabel('Audio Duration (seconds)')
    axes[0, 2].grid(True, alpha=0.3)
    
    difficulties = list(difficulty_analysis.keys())
    sample_counts = [difficulty_analysis[diff]["sample_count"] for diff in difficulties]
    
    colors = ['lightcoral', 'gold', 'lightblue', 'lightpink']
    axes[1, 0].bar(range(len(difficulties)), sample_counts, color=colors[:len(difficulties)])
    axes[1, 0].set_title('Sample Count by Difficulty Level')
    axes[1, 0].set_xlabel('Difficulty Level')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_xticks(range(len(difficulties)))
    axes[1, 0].set_xticklabels(difficulties)
    axes[1, 0].grid(True, alpha=0.3)
    
    avg_durations = [difficulty_analysis[diff]["average_length_minutes"] for diff in difficulties]
    
    axes[1, 1].bar(range(len(difficulties)), avg_durations, color=colors[:len(difficulties)])
    axes[1, 1].set_title('Average Audio Duration by Difficulty Level')
    axes[1, 1].set_xlabel('Difficulty Level')
    axes[1, 1].set_ylabel('Average Duration (minutes)')
    axes[1, 1].set_xticks(range(len(difficulties)))
    axes[1, 1].set_xticklabels(difficulties)
    axes[1, 1].grid(True, alpha=0.3)
    
    duration_dist = analyze_duration_distribution(durations)
    labels = [f"{k}\n({v['count']} samples)" for k, v in duration_dist.items() if v['count'] > 0]
    sizes = [v['percentage'] for v in duration_dist.values() if v['count'] > 0]
    
    axes[1, 2].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Audio Duration Range Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/race_audio_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Audio distribution plot saved to: {output_dir}/race_audio_distribution.png")

def export_detailed_analysis(stats, difficulty_analysis, duration_distribution, durations, 
                           output_dir="./race_audio_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_results = {
        "overall_statistics": stats,
        "difficulty_analysis": difficulty_analysis,
        "duration_distribution": duration_distribution,
        "duration_percentiles": {
            f"{p}%": float(np.percentile(durations, p)) 
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
    }
    
    json_file = f'{output_dir}/race_audio_analysis.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    df_overall = pd.DataFrame([stats])
    df_overall.to_csv(f'{output_dir}/overall_statistics.csv', index=False)
    
    df_difficulty = pd.DataFrame.from_dict(difficulty_analysis, orient='index')
    df_difficulty.to_csv(f'{output_dir}/difficulty_analysis.csv')
    
    df_duration_dist = pd.DataFrame.from_dict(
        {k: [v['count'], v['percentage']] for k, v in duration_distribution.items()},
        orient='index',
        columns=['Sample Count', 'Percentage']
    )
    df_duration_dist.to_csv(f'{output_dir}/duration_distribution.csv')
    
    print(f"Detailed analysis results saved to: {output_dir}/")
    print(f"  - JSON format: {json_file}")
    print(f"  - Overall statistics: {output_dir}/overall_statistics.csv")
    print(f"  - Difficulty analysis: {output_dir}/difficulty_analysis.csv")
    print(f"  - Duration distribution: {output_dir}/duration_distribution.csv")

def print_formatted_statistics(stats, difficulty_analysis, duration_distribution):
    print("\n" + "="*60)
    print("📚 RACE Dataset Audio Statistics Analysis Results")
    print("="*60)
    
    print(f"📈 Basic Statistics:")
    print(f"  Sample count: {stats['sample_count']:,}")
    print(f"  Total audio duration: {stats['total_audio_duration_hours']:.2f} hours")
    print(f"  Total file size: {stats['total_file_size_gb']:.2f} GB")
    
    print(f"\n⏱️  Audio Duration Statistics (seconds):")
    print(f"  Average duration: {stats['average_audio_length_seconds']:.2f}")
    print(f"  Shortest duration: {stats['shortest_audio_length_seconds']:.2f}")
    print(f"  Longest duration: {stats['longest_audio_length_seconds']:.2f}")
    print(f"  Median: {stats['audio_length_median_seconds']:.2f}")
    print(f"  Standard deviation: {stats['audio_length_std_seconds']:.2f}")
    
    print(f"\n📊 Percentile Statistics (seconds):")
    print(f"  25th percentile: {stats['audio_length_25th_percentile_seconds']:.2f}")
    print(f"  75th percentile: {stats['audio_length_75th_percentile_seconds']:.2f}")
    print(f"  95th percentile: {stats['audio_length_95th_percentile_seconds']:.2f}")
    print(f"  99th percentile: {stats['audio_length_99th_percentile_seconds']:.2f}")
    
    print(f"\n🎵 Audio Technical Parameters:")
    print(f"  Average sample rate: {stats['average_sample_rate_hz']:,.0f} Hz")
    print(f"  Sample rate types: {stats['sample_rate_types']}")
    print(f"  Channel types: {stats['channel_types']}")
    print(f"  Average file size: {stats['average_file_size_mb']:.2f} MB")
    
    print(f"\n🎯 Statistics by Difficulty Level:")
    for difficulty, diff_stats in difficulty_analysis.items():
        print(f"  {difficulty.upper()}:")
        print(f"    Sample count: {diff_stats['sample_count']:,}")
        print(f"    Average duration: {diff_stats['average_length_seconds']:.2f}s ({diff_stats['average_length_minutes']:.2f}min)")
        print(f"    Duration range: {diff_stats['shortest_length_seconds']:.2f}s - {diff_stats['longest_length_seconds']:.2f}s")
        print(f"    Total duration: {diff_stats['total_duration_hours']:.2f}h")
    
    print(f"\n⏰ Duration Range Distribution:")
    for range_name, dist_info in duration_distribution.items():
        if dist_info['count'] > 0:
            print(f"  {range_name}: {dist_info['count']:,} samples ({dist_info['percentage']:.1f}%)")

def analyze_long_audio_filtering(durations, min_duration_minutes=2):
    min_duration_seconds = min_duration_minutes * 60
    
    total_samples = len(durations)
    long_audio_samples = sum(1 for d in durations if d >= min_duration_seconds)
    filtering_ratio = long_audio_samples / total_samples * 100
    
    long_audio_durations = [d for d in durations if d >= min_duration_seconds]
    
    filtering_analysis = {
        "filtering_threshold": f"{min_duration_minutes} minutes ({min_duration_seconds} seconds)",
        "original_sample_count": total_samples,
        "filtered_sample_count": long_audio_samples,
        "retention_ratio": filtering_ratio,
        "removed_sample_count": total_samples - long_audio_samples,
        "removed_ratio": 100 - filtering_ratio
    }
    
    if long_audio_durations:
        filtering_analysis.update({
            "long_audio_average_duration_seconds": np.mean(long_audio_durations),
            "long_audio_shortest_duration_seconds": np.min(long_audio_durations),
            "long_audio_longest_duration_seconds": np.max(long_audio_durations),
            "long_audio_median_seconds": np.median(long_audio_durations),
            "long_audio_total_duration_hours": np.sum(long_audio_durations) / 3600
        })
    
    return filtering_analysis

def main():
    output_dir = "./race_audio_analysis"
    
    data_path_root = "/data/to/your/dataset/path//race_audio"

    print("📚 RACE Dataset Audio Statistics Analysis Tool")
    print(f"Data root directory: {data_path_root}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(data_path_root):
        print(f"❌ Error: Data directory does not exist - {data_path_root}")
        return
    
    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"❌ Error: Benchmark file does not exist - {bench_path}")
        return
    
    print("\n📁 Loading RACE dataset...")
    dataset = load_race_dataset_for_analysis(data_path_root)
    
    if not dataset:
        print("❌ No valid audio data found")
        return
    
    print("\n📊 Calculating audio statistics...")
    stats, durations, sample_rates, file_sizes_mb = calculate_audio_statistics(dataset)
    
    print("\n🔍 Analyzing by difficulty level...")
    difficulty_analysis = analyze_by_difficulty(dataset)
    
    print("\n⏰ Analyzing duration distribution...")
    duration_distribution = analyze_duration_distribution(durations)
    
    print("\n🎯 Analyzing long audio filtering impact (>=2 minutes)...")
    long_audio_filtering = analyze_long_audio_filtering(durations, min_duration_minutes=2)
    
    print_formatted_statistics(stats, difficulty_analysis, duration_distribution)
    
    print(f"\n🔍 Long Audio Filtering Analysis (corresponding to race_aero1.py filtering functionality):")
    for key, value in long_audio_filtering.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📈 Generating audio distribution plots...")
    plot_race_audio_distribution(durations, difficulty_analysis, output_dir)
    
    print(f"\n💾 Exporting detailed analysis results...")
    export_data = {
        **{
            "overall_statistics": stats,
            "difficulty_analysis": difficulty_analysis,
            "duration_distribution": duration_distribution,
            "long_audio_filtering_analysis": long_audio_filtering
        }
    }
    
    export_detailed_analysis(stats, difficulty_analysis, duration_distribution, durations, output_dir)
    
    long_audio_file = f'{output_dir}/long_audio_filtering_analysis.json'
    with open(long_audio_file, 'w', encoding='utf-8') as f:
        json.dump(long_audio_filtering, f, indent=2, ensure_ascii=False)
    print(f"  - Long audio filtering analysis: {long_audio_file}")
    
    print(f"\n✅ Analysis complete! All results saved to: {output_dir}")
    
    print(f"\n📋 Additional Statistics:")
    print(f"  Samples over 1 minute: {sum(1 for d in durations if d >= 60):,} ({sum(1 for d in durations if d >= 60)/len(durations)*100:.1f}%)")
    print(f"  Samples over 2 minutes: {sum(1 for d in durations if d >= 120):,} ({sum(1 for d in durations if d >= 120)/len(durations)*100:.1f}%)")
    print(f"  Samples over 5 minutes: {sum(1 for d in durations if d >= 300):,} ({sum(1 for d in durations if d >= 300)/len(durations)*100:.1f}%)")
    
    print(f"\n🔧 Validation with race_aero1.py filtering logic:")
    print(f"  Default filtering threshold (120s): retains {long_audio_filtering['filtered_sample_count']} samples")
    print(f"  Retention ratio: {long_audio_filtering['retention_ratio']:.2f}%")

if __name__ == "__main__":
    main()