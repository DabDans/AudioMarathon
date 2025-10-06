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

def load_vox_age_dataset_for_analysis(root_dir):
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Cannot find metadata file: {meta_file}")
        return []
    
    print(f"Loading VoxCeleb age classification metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    all_samples = []
    missing_files = 0
    audio_info_errors = 0
    
    print(f"Loaded {len(metadata)} sample information from metadata")
    
    for i, item in tqdm(enumerate(metadata), total=len(metadata), desc="Loading audio file information"):
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(wav_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: File does not exist {wav_path}")
            continue
        
        try:
            audio_info = sf.info(wav_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
            frames = audio_info.frames
            channels = audio_info.channels
            
            file_size = os.path.getsize(wav_path)
            
        except Exception as e:
            audio_info_errors += 1
            if audio_info_errors <= 5:
                print(f"Warning: Cannot get audio information {wav_path}: {e}")
            continue
        
        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()
        speaker_age = item.get("speaker_age", 0)
        
        sample_data = {
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "wav_path": wav_path,
            "relative_path": rel_path,
            "filename": os.path.basename(wav_path),
            "duration": duration,
            "sample_rate": sample_rate,
            "frames": frames,
            "channels": channels,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "task": "Speaker_Age_Classification",
            "question": item.get("question", ""),
            "choice_a": item.get("choice_a", ""),
            "choice_b": item.get("choice_b", ""),
            "choice_c": item.get("choice_c", ""),
            "choice_d": item.get("choice_d", ""),
            "choice_e": item.get("choice_e", ""),
            "id": f"vox_age_{item.get('uniq_id', i)}"
        }
        
        all_samples.append(sample_data)
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    if audio_info_errors > 5:
        print(f"Warning: Total {audio_info_errors} audio files failed to get information")
    
    print(f"Successfully loaded {len(all_samples)} valid audio samples")
    
    age_group_counts = defaultdict(int)
    speaker_counts = defaultdict(int)
    
    for sample in all_samples:
        age_group_counts[sample["age_group"]] += 1
        speaker_counts[sample["speaker_id"]] += 1
    
    print(f"Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} samples")
    
    print(f"Speaker statistics:")
    print(f"  Unique speakers: {len(speaker_counts)}")
    print(f"  Average samples per speaker: {np.mean(list(speaker_counts.values())):.2f}")
    
    return all_samples

def calculate_audio_statistics(dataset):
    if not dataset:
        print("Error: No valid audio data")
        return None
    
    durations = [item["duration"] for item in dataset]
    sample_rates = [item["sample_rate"] for item in dataset]
    file_sizes_mb = [item["file_size_mb"] for item in dataset]
    channels_list = [item["channels"] for item in dataset]
    speaker_ages = [item["speaker_age"] for item in dataset if item["speaker_age"] > 0]
    
    stats = {
        "sample_count": len(dataset),
        "unique_speakers": len(set(item["speaker_id"] for item in dataset)),
        "avg_audio_duration_sec": np.mean(durations),
        "min_audio_duration_sec": np.min(durations),
        "max_audio_duration_sec": np.max(durations),
        "median_audio_duration_sec": np.median(durations),
        "std_audio_duration_sec": np.std(durations),
        "duration_25_percentile_sec": np.percentile(durations, 25),
        "duration_75_percentile_sec": np.percentile(durations, 75),
        "duration_95_percentile_sec": np.percentile(durations, 95),
        "duration_99_percentile_sec": np.percentile(durations, 99),
        "total_duration_hours": np.sum(durations) / 3600,
        "avg_sample_rate_hz": np.mean(sample_rates),
        "sample_rate_types": list(set(sample_rates)),
        "avg_file_size_mb": np.mean(file_sizes_mb),
        "total_file_size_gb": np.sum(file_sizes_mb) / 1024,
        "channel_types": list(set(channels_list)),
        "avg_speaker_age": np.mean(speaker_ages) if speaker_ages else 0,
        "speaker_age_range": f"{min(speaker_ages)}-{max(speaker_ages)}" if speaker_ages else "N/A"
    }
    
    return stats, durations, sample_rates, file_sizes_mb

def analyze_by_age_group(dataset):
    age_group_stats = defaultdict(list)
    age_group_speakers = defaultdict(set)
    
    for item in dataset:
        age_group = item["age_group"]
        age_group_stats[age_group].append(item["duration"])
        age_group_speakers[age_group].add(item["speaker_id"])
    
    age_group_analysis = {}
    for age_group, durations in age_group_stats.items():
        age_group_analysis[age_group] = {
            "sample_count": len(durations),
            "unique_speakers": len(age_group_speakers[age_group]),
            "avg_duration_sec": np.mean(durations),
            "min_duration_sec": np.min(durations),
            "max_duration_sec": np.max(durations),
            "median_sec": np.median(durations),
            "std_sec": np.std(durations),
            "total_duration_hours": np.sum(durations) / 3600
        }
    
    return age_group_analysis

def analyze_by_speaker(dataset):
    speaker_stats = defaultdict(list)
    speaker_age_groups = {}
    
    for item in dataset:
        speaker_id = item["speaker_id"]
        speaker_stats[speaker_id].append(item["duration"])
        speaker_age_groups[speaker_id] = item["age_group"]
    
    speaker_analysis = []
    for speaker_id, durations in speaker_stats.items():
        speaker_analysis.append({
            "speaker_id": speaker_id,
            "age_group": speaker_age_groups[speaker_id],
            "sample_count": len(durations),
            "total_duration_sec": np.sum(durations),
            "avg_duration_sec": np.mean(durations),
            "min_duration_sec": np.min(durations),
            "max_duration_sec": np.max(durations),
            "median_sec": np.median(durations)
        })
    
    speaker_analysis.sort(key=lambda x: x["total_duration_sec"], reverse=True)
    
    return speaker_analysis

def plot_vox_age_distribution(durations, age_group_analysis, output_dir="./vox_age_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VoxCeleb Age Classification Dataset Audio Statistical Analysis', fontsize=16, fontweight='bold')
    
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
    
    age_groups = list(age_group_analysis.keys())
    sample_counts = [age_group_analysis[group]["sample_count"] for group in age_groups]
    
    axes[1, 0].bar(range(len(age_groups)), sample_counts, color='lightcoral')
    axes[1, 0].set_title('Sample Count by Age Group')
    axes[1, 0].set_xlabel('Age Group')
    axes[1, 0].set_ylabel('Sample Count')
    axes[1, 0].set_xticks(range(len(age_groups)))
    axes[1, 0].set_xticklabels([group.replace(' ', '\n') for group in age_groups], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    avg_durations = [age_group_analysis[group]["avg_duration_sec"] for group in age_groups]
    
    axes[1, 1].bar(range(len(age_groups)), avg_durations, color='plum')
    axes[1, 1].set_title('Average Audio Duration by Age Group')
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Average Duration (seconds)')
    axes[1, 1].set_xticks(range(len(age_groups)))
    axes[1, 1].set_xticklabels([group.replace(' ', '\n') for group in age_groups], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(durations, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title('Audio Duration Distribution Histogram (Log Scale)')
    axes[1, 2].set_xlabel('Audio Duration (seconds)')
    axes[1, 2].set_ylabel('Sample Count (Log Scale)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vox_age_audio_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Audio distribution plot saved to: {output_dir}/vox_age_audio_distribution.png")

def export_detailed_analysis(stats, age_group_analysis, speaker_analysis, durations, 
                           output_dir="./vox_age_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_results = {
        "overall_statistics": stats,
        "age_group_analysis": age_group_analysis,
        "speaker_top_20": speaker_analysis[:20],
        "duration_percentiles": {
            f"{p}%": float(np.percentile(durations, p)) 
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
    }
    
    json_file = f'{output_dir}/vox_age_audio_analysis.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    df_overall = pd.DataFrame([stats])
    df_overall.to_csv(f'{output_dir}/overall_statistics.csv', index=False)
    
    df_age_groups = pd.DataFrame.from_dict(age_group_analysis, orient='index')
    df_age_groups.to_csv(f'{output_dir}/age_group_analysis.csv')
    
    df_speakers = pd.DataFrame(speaker_analysis)
    df_speakers.to_csv(f'{output_dir}/speaker_analysis.csv', index=False)
    
    print(f"Detailed analysis results saved to: {output_dir}/")
    print(f"  - JSON format: {json_file}")
    print(f"  - Overall statistics: {output_dir}/overall_statistics.csv")
    print(f"  - Age group analysis: {output_dir}/age_group_analysis.csv")
    print(f"  - Speaker analysis: {output_dir}/speaker_analysis.csv")

def print_formatted_statistics(stats, age_group_analysis):
    print("\n" + "="*60)
    print("🎙️  VoxCeleb Age Classification Dataset Audio Statistical Analysis Results")
    print("="*60)
    
    print(f"📈 Basic Statistics:")
    print(f"  Sample count: {stats['sample_count']:,}")
    print(f"  Unique speakers: {stats['unique_speakers']:,}")
    print(f"  Total audio duration: {stats['total_duration_hours']:.2f} hours")
    print(f"  Total file size: {stats['total_file_size_gb']:.2f} GB")
    print(f"  Average speaker age: {stats['avg_speaker_age']:.1f} years")
    print(f"  Speaker age range: {stats['speaker_age_range']}")
    
    print(f"\n⏱️  Audio Duration Statistics (seconds):")
    print(f"  Average duration: {stats['avg_audio_duration_sec']:.2f}")
    print(f"  Minimum duration: {stats['min_audio_duration_sec']:.2f}")
    print(f"  Maximum duration: {stats['max_audio_duration_sec']:.2f}")
    print(f"  Median: {stats['median_audio_duration_sec']:.2f}")
    print(f"  Standard deviation: {stats['std_audio_duration_sec']:.2f}")
    
    print(f"\n📊 Percentile Statistics (seconds):")
    print(f"  25th percentile: {stats['duration_25_percentile_sec']:.2f}")
    print(f"  75th percentile: {stats['duration_75_percentile_sec']:.2f}")
    print(f"  95th percentile: {stats['duration_95_percentile_sec']:.2f}")
    print(f"  99th percentile: {stats['duration_99_percentile_sec']:.2f}")
    
    print(f"\n🎵 Audio Technical Parameters:")
    print(f"  Average sample rate: {stats['avg_sample_rate_hz']:,.0f} Hz")
    print(f"  Sample rate types: {stats['sample_rate_types']}")
    print(f"  Channel types: {stats['channel_types']}")
    print(f"  Average file size: {stats['avg_file_size_mb']:.2f} MB")
    
    print(f"\n👥 Statistics by Age Group:")
    for age_group, group_stats in age_group_analysis.items():
        print(f"  {age_group}:")
        print(f"    Sample count: {group_stats['sample_count']:,}")
        print(f"    Speaker count: {group_stats['unique_speakers']:,}")
        print(f"    Average duration: {group_stats['avg_duration_sec']:.2f} seconds")
        print(f"    Total duration: {group_stats['total_duration_hours']:.2f} hours")

def main():
    
    data_path_root = "/data/to/your/dataset/path//VoxCeleb/concatenated_audio"
    output_dir = "./vox_age_audio_analysis"
    
    print("🎙️  VoxCeleb Age Classification Dataset Audio Statistical Analysis Tool")
    print(f"Data root directory: {data_path_root}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(data_path_root):
        print(f"❌ Error: Data directory does not exist - {data_path_root}")
        return
    
    meta_file = os.path.join(data_path_root, "age_classification_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"❌ Error: Metadata file does not exist - {meta_file}")
        return
    
    print("\n📁 Loading VoxCeleb age classification dataset...")
    dataset = load_vox_age_dataset_for_analysis(data_path_root)
    
    if not dataset:
        print("❌ No valid audio data found")
        return
    
    print("\n📊 Calculating audio statistics...")
    stats, durations, sample_rates, file_sizes_mb = calculate_audio_statistics(dataset)
    
    print("\n🔍 Analyzing by age group...")
    age_group_analysis = analyze_by_age_group(dataset)
    
    print("\n👤 Analyzing by speaker...")
    speaker_analysis = analyze_by_speaker(dataset)
    
    print_formatted_statistics(stats, age_group_analysis)
    
    print(f"\n🏆 Top 10 Speakers (by total duration):")
    for i, speaker in enumerate(speaker_analysis[:10]):
        print(f"  {i+1}. {speaker['speaker_id']} ({speaker['age_group']}):")
        print(f"     Sample count: {speaker['sample_count']}, Total duration: {speaker['total_duration_sec']:.1f} seconds")
    
    print(f"\n📈 Generating audio distribution plots...")
    plot_vox_age_distribution(durations, age_group_analysis, output_dir)
    
    print(f"\n💾 Exporting detailed analysis results...")
    export_detailed_analysis(stats, age_group_analysis, speaker_analysis, durations, output_dir)
    
    print(f"\n✅ Analysis completed! All results saved to: {output_dir}")
    
    print(f"\n📋 Additional Statistics:")
    print(f"  Average samples per speaker: {len(dataset) / len(set(item['speaker_id'] for item in dataset)):.2f}")
    print(f"  Speaker with most samples: {speaker_analysis[0]['speaker_id']} ({speaker_analysis[0]['sample_count']} samples)")
    print(f"  Age group with most balanced distribution: {min(age_group_analysis.keys(), key=lambda k: age_group_analysis[k]['sample_count'])}")
    print(f"  Age group with most samples: {max(age_group_analysis.keys(), key=lambda k: age_group_analysis[k]['sample_count'])}")

if __name__ == "__main__":
    main()