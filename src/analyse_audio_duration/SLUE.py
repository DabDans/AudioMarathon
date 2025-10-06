import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_slue_dataset_for_analysis(json_file, audio_base_dir):
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []
    
    print(f"Loading SLUE JSON file: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return []
    
    if not isinstance(data, list):
        print(f"Error: Incorrect JSON file format, expected list format")
        return []
    
    print(f"Loaded {len(data)} tasks from JSON")
    
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0
    audio_info_errors = 0
    
    for i, task in tqdm(enumerate(data), total=len(data), desc="Loading audio information"):
        relative_path = task.get("path", "")
        if not relative_path:
            print(f"Warning: Task missing 'path' key, skipped: {task}")
            continue

        full_audio_path = os.path.join(audio_base_dir, relative_path)
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue
        
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
            frames = audio_info.frames
            channels = audio_info.channels
            
            file_size = os.path.getsize(full_audio_path)
            
        except Exception as e:
            audio_info_errors += 1
            if audio_info_errors <= 5:
                print(f"Warning: Cannot get audio info {full_audio_path}: {e}")
            continue

        item = {
            "path": full_audio_path,
            "relative_path": relative_path,
            "filename": os.path.basename(full_audio_path),
            "task_name": task.get("task_name", "unknown"),
            "dataset_name": task.get("dataset_name", "unknown"),
            "duration": duration,
            "sample_rate": sample_rate,
            "frames": frames,
            "channels": channels,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[item["task_name"]] += 1
        dataset_stats[item["dataset_name"]] += 1
    
    if missing_files > 5:
        print(f"Warning: Total of {missing_files} audio files do not exist")
    if audio_info_errors > 5:
        print(f"Warning: Total of {audio_info_errors} audio files failed to get info")
    
    print(f"Successfully loaded {len(dataset)} valid audio samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    print(f"Dataset statistics: {dict(dataset_stats)}")
    return dataset

def categorize_duration(duration):
    if duration < 1:
        return "0-1s"
    elif duration < 5:
        return "1-5s"
    elif duration < 10:
        return "5-10s"
    elif duration < 30:
        return "10-30s"
    elif duration < 60:
        return "30-60s"
    elif duration < 120:
        return "1-2min"
    elif duration < 300:
        return "2-5min"
    elif duration < 600:
        return "5-10min"
    elif duration < 1800:
        return "10-30min"
    else:
        return "30min+"

def analyze_duration_distribution(dataset):
    if not dataset:
        print("Error: No valid audio data")
        return None
    
    duration_categories = defaultdict(int)
    duration_details = defaultdict(list)
    
    for item in dataset:
        duration = item["duration"]
        category = categorize_duration(duration)
        duration_categories[category] += 1
        duration_details[category].append({
            'filename': item['filename'],
            'duration': duration,
            'task_name': item['task_name'],
            'dataset_name': item['dataset_name']
        })
    
    ordered_categories = [
        "0-1s", "1-5s", "5-10s", "10-30s", "30-60s",
        "1-2min", "2-5min", "5-10min", "10-30min", "30min+"
    ]
    
    ordered_distribution = {}
    for category in ordered_categories:
        if category in duration_categories:
            ordered_distribution[category] = duration_categories[category]
    
    total_count = len(dataset)
    distribution_with_percentage = {}
    for category, count in ordered_distribution.items():
        percentage = (count / total_count) * 100
        distribution_with_percentage[category] = {
            'count': count,
            'percentage': percentage
        }
    
    return ordered_distribution, distribution_with_percentage, duration_details

def analyze_detailed_duration_bins(dataset, bin_size=10):
    if not dataset:
        return None
    
    durations = [item["duration"] for item in dataset]
    max_duration = max(durations)
    
    bins = list(range(0, int(max_duration) + bin_size, bin_size))
    bin_counts = defaultdict(int)
    
    for duration in durations:
        bin_index = int(duration // bin_size) * bin_size
        bin_label = f"{bin_index}-{bin_index + bin_size}s"
        bin_counts[bin_label] += 1
    
    return dict(bin_counts)

def analyze_duration_by_percentiles(dataset):
    if not dataset:
        return None
    
    durations = [item["duration"] for item in dataset]
    durations_sorted = sorted(durations)
    
    percentile_ranges = [
        (0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99), (99, 100)
    ]
    
    percentile_analysis = {}
    total_count = len(durations)
    
    for start_p, end_p in percentile_ranges:
        start_idx = int(total_count * start_p / 100)
        end_idx = int(total_count * end_p / 100)
        
        if end_idx >= total_count:
            end_idx = total_count - 1
        
        range_durations = durations_sorted[start_idx:end_idx + 1] if start_idx <= end_idx else []
        
        if range_durations:
            percentile_analysis[f"{start_p}-{end_p}%"] = {
                'count': len(range_durations),
                'min_duration': min(range_durations),
                'max_duration': max(range_durations),
                'avg_duration': np.mean(range_durations)
            }
    
    return percentile_analysis

def calculate_audio_statistics(dataset):
    if not dataset:
        print("Error: No valid audio data")
        return None
    
    durations = [item["duration"] for item in dataset]
    sample_rates = [item["sample_rate"] for item in dataset]
    file_sizes_mb = [item["file_size_mb"] for item in dataset]
    channels_list = [item["channels"] for item in dataset]
    
    stats = {
        "sample_count": len(dataset),
        "mean_duration_seconds": np.mean(durations),
        "min_duration_seconds": np.min(durations),
        "max_duration_seconds": np.max(durations),
        "median_duration_seconds": np.median(durations),
        "std_duration_seconds": np.std(durations),
        "percentile_25_seconds": np.percentile(durations, 25),
        "percentile_75_seconds": np.percentile(durations, 75),
        "percentile_95_seconds": np.percentile(durations, 95),
        "percentile_99_seconds": np.percentile(durations, 99),
        "total_duration_hours": np.sum(durations) / 3600,
        "mean_sample_rate_hz": np.mean(sample_rates),
        "sample_rate_types": list(set(sample_rates)),
        "mean_file_size_mb": np.mean(file_sizes_mb),
        "total_file_size_gb": np.sum(file_sizes_mb) / 1024,
        "channel_types": list(set(channels_list))
    }
    
    return stats, durations, sample_rates, file_sizes_mb

def analyze_by_task_and_dataset(dataset):
    task_stats = defaultdict(list)
    dataset_stats = defaultdict(list)
    
    for item in dataset:
        task_stats[item["task_name"]].append(item["duration"])
        dataset_stats[item["dataset_name"]].append(item["duration"])
    
    task_analysis = {}
    for task_name, durations in task_stats.items():
        task_analysis[task_name] = {
            "sample_count": len(durations),
            "mean_duration_seconds": np.mean(durations),
            "min_duration_seconds": np.min(durations),
            "max_duration_seconds": np.max(durations),
            "median_seconds": np.median(durations),
            "std_seconds": np.std(durations)
        }
    
    dataset_analysis = {}
    for dataset_name, durations in dataset_stats.items():
        dataset_analysis[dataset_name] = {
            "sample_count": len(durations),
            "mean_duration_seconds": np.mean(durations),
            "min_duration_seconds": np.min(durations),
            "max_duration_seconds": np.max(durations),
            "median_seconds": np.median(durations),
            "std_seconds": np.std(durations)
        }
    
    return task_analysis, dataset_analysis

def plot_duration_distribution_charts(durations, duration_distribution, output_dir="./audio_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('SLUE Dataset Audio Duration Distribution Analysis', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Audio Duration Distribution Histogram')
    axes[0, 0].set_xlabel('Audio Duration (seconds)')
    axes[0, 0].set_ylabel('Sample Count')
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
    
    categories = list(duration_distribution.keys())
    counts = list(duration_distribution.values())
    
    bars = axes[1, 0].bar(range(len(categories)), counts, alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Audio Duration Interval Distribution')
    axes[1, 0].set_xlabel('Duration Interval')
    axes[1, 0].set_ylabel('Audio Count')
    axes[1, 0].set_xticks(range(len(categories)))
    axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom', fontsize=9)
    
    axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Audio Duration Interval Distribution Percentage')
    
    axes[1, 2].hist(durations, bins=50, alpha=0.7, color='plum', edgecolor='black')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title('Audio Duration Distribution Histogram (Log Scale)')
    axes[1, 2].set_xlabel('Audio Duration (seconds)')
    axes[1, 2].set_ylabel('Sample Count (Log Scale)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/audio_duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Audio duration distribution chart saved to: {output_dir}/audio_duration_distribution.png")

def print_duration_distribution_analysis(duration_distribution, distribution_with_percentage, 
                                        detailed_bins, percentile_analysis):
    print("\n" + "="*80)
    print("📊 Audio Duration Distribution Detailed Analysis")
    print("="*80)
    
    print(f"\n🔢 Statistics by Duration Interval:")
    print(f"{'Interval':<15} {'Count':<10} {'Percentage':<10} {'Cumulative %':<12}")
    print("-" * 50)
    
    cumulative_percentage = 0
    for category, count in duration_distribution.items():
        percentage = distribution_with_percentage[category]['percentage']
        cumulative_percentage += percentage
        print(f"{category:<15} {count:<10} {percentage:>6.2f}% {cumulative_percentage:>9.2f}%")
    
    print(f"\n📈 Analysis by Percentile Intervals:")
    print(f"{'Percentile Range':<15} {'Count':<8} {'Min(s)':<10} {'Max(s)':<10} {'Avg(s)':<10}")
    print("-" * 65)
    
    for range_name, stats in percentile_analysis.items():
        print(f"{range_name:<15} {stats['count']:<8} {stats['min_duration']:<10.2f} "
              f"{stats['max_duration']:<10.2f} {stats['avg_duration']:<10.2f}")
    
    print(f"\n🎯 Key Findings:")
    total_samples = sum(duration_distribution.values())
    
    max_category = max(duration_distribution.items(), key=lambda x: x[1])
    min_category = min(duration_distribution.items(), key=lambda x: x[1])
    
    print(f"  • Most common duration interval: {max_category[0]} ({max_category[1]} samples, "
          f"{(max_category[1]/total_samples)*100:.1f}%)")
    print(f"  • Least common duration interval: {min_category[0]} ({min_category[1]} samples, "
          f"{(min_category[1]/total_samples)*100:.1f}%)")
    
    short_audio_count = sum(count for category, count in duration_distribution.items() 
                           if "s" in category and "min" not in category)
    long_audio_count = total_samples - short_audio_count
    
    print(f"  • Short audio (<60s): {short_audio_count} samples ({(short_audio_count/total_samples)*100:.1f}%)")
    print(f"  • Long audio (≥60s): {long_audio_count} samples ({(long_audio_count/total_samples)*100:.1f}%)")

def export_detailed_analysis(stats, task_analysis, dataset_analysis, durations, 
                           duration_distribution, distribution_with_percentage, 
                           detailed_bins, percentile_analysis, duration_details,
                           output_dir="./audio_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_results = {
        "overall_statistics": stats,
        "task_analysis": task_analysis,
        "dataset_analysis": dataset_analysis,
        "duration_percentiles": {
            f"{p}%": float(np.percentile(durations, p)) 
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        "duration_distribution_by_intervals": duration_distribution,
        "duration_distribution_with_percentage": distribution_with_percentage,
        "detailed_duration_bins": detailed_bins,
        "percentile_analysis": percentile_analysis
    }
    
    json_file = f'{output_dir}/slue_audio_analysis.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    df_overall = pd.DataFrame([stats])
    df_overall.to_csv(f'{output_dir}/overall_statistics.csv', index=False)
    
    df_tasks = pd.DataFrame.from_dict(task_analysis, orient='index')
    df_tasks.to_csv(f'{output_dir}/task_analysis.csv')
    
    df_datasets = pd.DataFrame.from_dict(dataset_analysis, orient='index')
    df_datasets.to_csv(f'{output_dir}/dataset_analysis.csv')
    
    duration_dist_data = []
    for category, count in duration_distribution.items():
        percentage = distribution_with_percentage[category]['percentage']
        duration_dist_data.append({
            'duration_interval': category,
            'count': count,
            'percentage': percentage
        })
    
    df_duration_dist = pd.DataFrame(duration_dist_data)
    df_duration_dist.to_csv(f'{output_dir}/duration_distribution.csv', index=False)
    
    df_percentiles = pd.DataFrame.from_dict(percentile_analysis, orient='index')
    df_percentiles.to_csv(f'{output_dir}/percentile_analysis.csv')
    
    detailed_audio_data = []
    for item in duration_details.values():
        detailed_audio_data.extend(item)
    
    df_detailed = pd.DataFrame(detailed_audio_data)
    df_detailed.to_csv(f'{output_dir}/detailed_audio_files.csv', index=False)
    
    print(f"Detailed analysis results saved to: {output_dir}/")
    print(f"  - JSON format: {json_file}")
    print(f"  - CSV format: {output_dir}/overall_statistics.csv")
    print(f"  - Task analysis: {output_dir}/task_analysis.csv")
    print(f"  - Dataset analysis: {output_dir}/dataset_analysis.csv")
    print(f"  - Duration distribution analysis: {output_dir}/duration_distribution.csv")
    print(f"  - Percentile analysis: {output_dir}/percentile_analysis.csv")
    print(f"  - Detailed audio file info: {output_dir}/detailed_audio_files.csv")

def print_formatted_statistics(stats):
    print("\n" + "="*60)
    print("📊 SLUE Dataset Audio Statistics Analysis Results")
    print("="*60)
    
    print(f"📈 Basic Statistics:")
    print(f"  Sample count: {stats['sample_count']:,}")
    print(f"  Total audio duration: {stats['total_duration_hours']:.2f} hours")
    print(f"  Total file size: {stats['total_file_size_gb']:.2f} GB")
    
    print(f"\n⏱️  Audio Duration Statistics (seconds):")
    print(f"  Mean duration: {stats['mean_duration_seconds']:.2f}")
    print(f"  Min duration: {stats['min_duration_seconds']:.2f}")
    print(f"  Max duration: {stats['max_duration_seconds']:.2f}")
    print(f"  Median: {stats['median_duration_seconds']:.2f}")
    print(f"  Standard deviation: {stats['std_duration_seconds']:.2f}")
    
    print(f"\n📊 Percentile Statistics (seconds):")
    print(f"  25th percentile: {stats['percentile_25_seconds']:.2f}")
    print(f"  75th percentile: {stats['percentile_75_seconds']:.2f}")
    print(f"  95th percentile: {stats['percentile_95_seconds']:.2f}")
    print(f"  99th percentile: {stats['percentile_99_seconds']:.2f}")
    
    print(f"\n🎵 Audio Technical Parameters:")
    print(f"  Mean sample rate: {stats['mean_sample_rate_hz']:,.0f} Hz")
    print(f"  Sample rate types: {stats['sample_rate_types']}")
    print(f"  Channel types: {stats['channel_types']}")
    print(f"  Mean file size: {stats['mean_file_size_mb']:.2f} MB")

def main():
    slue_json_file = "/data/to/your/dataset/path//SLUE/merged_audio_data.json"
    audio_base_dir = "/data/to/your/dataset/path//SLUE"
    output_dir = "./slue_audio_analysis"
    
    print("🎵 SLUE Dataset Audio Statistics Analysis Tool (Enhanced Version)")
    print(f"JSON file path: {slue_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(slue_json_file):
        print(f"❌ Error: JSON file does not exist - {slue_json_file}")
        return
    
    if not os.path.exists(audio_base_dir):
        print(f"❌ Error: Audio directory does not exist - {audio_base_dir}")
        return
    
    print("\n📁 Loading SLUE dataset...")
    dataset = load_slue_dataset_for_analysis(slue_json_file, audio_base_dir)
    
    if not dataset:
        print("❌ No valid audio data found")
        return
    
    print("\n📊 Calculating audio statistics...")
    stats, durations, sample_rates, file_sizes_mb = calculate_audio_statistics(dataset)
    
    print("\n🔍 Analyzing audio duration distribution...")
    duration_distribution, distribution_with_percentage, duration_details = analyze_duration_distribution(dataset)
    
    print("\n📏 Calculating detailed time interval distribution...")
    detailed_bins = analyze_detailed_duration_bins(dataset, bin_size=10)
    
    print("\n📈 Performing percentile analysis...")
    percentile_analysis = analyze_duration_by_percentiles(dataset)
    
    print("\n🔍 Analyzing by task type and dataset...")
    task_analysis, dataset_analysis = analyze_by_task_and_dataset(dataset)
    
    print_formatted_statistics(stats)
    
    print_duration_distribution_analysis(duration_distribution, distribution_with_percentage, 
                                       detailed_bins, percentile_analysis)
    
    print(f"\n📋 Analysis by Task Type:")
    for task_name, task_stats in task_analysis.items():
        print(f"  {task_name}:")
        print(f"    Sample count: {task_stats['sample_count']:,}")
        print(f"    Mean duration: {task_stats['mean_duration_seconds']:.2f}s")
        print(f"    Median: {task_stats['median_seconds']:.2f}s")
    
    print(f"\n🗂️  Analysis by Dataset:")
    for dataset_name, dataset_stats in dataset_analysis.items():
        print(f"  {dataset_name}:")
        print(f"    Sample count: {dataset_stats['sample_count']:,}")
        print(f"    Mean duration: {dataset_stats['mean_duration_seconds']:.2f}s")
        print(f"    Median: {dataset_stats['median_seconds']:.2f}s")
    
    print(f"\n💾 Exporting detailed analysis results...")
    export_detailed_analysis(stats, task_analysis, dataset_analysis, durations, 
                            duration_distribution, distribution_with_percentage, 
                            detailed_bins, percentile_analysis, duration_details, output_dir)
    
    print(f"\n✅ Analysis complete! All results saved to: {output_dir}")
    print(f"\n🎯 New Features Summary:")
    print(f"  • Statistics by duration intervals with counts and percentages")
    print(f"  • Analysis by percentile intervals for audio distribution")
    print(f"  • Generation of detailed duration distribution visualization charts")
    print(f"  • Export of complete detailed audio file information")

if __name__ == "__main__":
    main()