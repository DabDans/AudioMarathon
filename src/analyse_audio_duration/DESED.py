import os
import json
import glob
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import sys
from collections import defaultdict

def load_desed_qa_dataset_for_analysis(json_file, audio_base_dir):
    dataset = []
    metadata = {}
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return [], {}
    
    print(f"Loading DESED QA format JSON: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return [], {}
    
    if isinstance(data, dict) and "tasks" in data:
        print("Detected new format JSON file: {'task_info':{}, 'missing_files_info': {}, 'tasks': []}")
        
        metadata = {
            "task_info": data.get("task_info", {}),
            "missing_files_info": data.get("missing_files_info", {}),
            "json_format": "structured"
        }
        
        tasks = data.get("tasks", [])
        
        if metadata["task_info"]:
            print(f"Task info: {metadata['task_info']}")
        if metadata["missing_files_info"]:
            print(f"Missing files info: {metadata['missing_files_info']}")
        
    elif isinstance(data, list):
        print("Detected old format JSON file: direct list format")
        tasks = data
        metadata = {"json_format": "list"}
    else:
        print(f"Error: JSON file format is incorrect, expected dictionary with 'tasks' field or direct list format")
        return [], {}
    
    print(f"Loaded {len(tasks)} tasks from JSON")
    
    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task missing audio path: {task}")
            continue
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue
        
        task_type = task.get("task_type", "unknown")
        
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "task_type": task_type,
            "uniq_id": task.get("uniq_id", i),
            "original_events": task.get("original_events", []),
            "all_events": task.get("all_events", []),
            "primary_event": task.get("primary_event", ""),
            "dominant_event": task.get("dominant_event", ""),
            "question": task.get("question", ""),
            "answer_gt": task.get("answer_gt", ""),
            "choice_a": task.get("choice_a", ""),
            "choice_b": task.get("choice_b", ""),
            "choice_c": task.get("choice_c", ""),
            "choice_d": task.get("choice_d", "")
        }
        
        dataset.append(item)
        task_type_stats[task_type] += 1
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    return dataset, metadata

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
    
    bins = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, float('inf')]
    bin_labels = [
        '0-5s', '5-10s', '10-15s', '15-20s', '20-30s', 
        '30-45s', '45-60s', '1-1.5min', '1.5-2min', 
        '2-3min', '3-5min', '5-10min', '10+min'
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

def main():
    qa_json_file = "/data/to/your/dataset/path//DESED/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json"
    audio_base_dir = "/data/to/your/dataset/path//DESED/DESED_dataset/concatenated_audio"    
    result_dir = './DESED_Analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== DESED Dataset Audio Length Analysis ===")
    print(f"QA JSON file: {qa_json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    print(f"Results save path: {result_dir}")
    print()
    
    if not os.path.exists(qa_json_file):
        print(f"Error: QA JSON file does not exist: {qa_json_file}")
        return
    
    if not os.path.exists(audio_base_dir):
        print(f"Error: Audio base directory does not exist: {audio_base_dir}")
        return
    
    print("Loading audio information from QA JSON file...")
    audio_files, json_metadata = load_desed_qa_dataset_for_analysis(qa_json_file, audio_base_dir)
    
    if not audio_files:
        print("Error: No valid audio files found")
        return
    
    print()
    
    print("Analyzing audio lengths...")
    
    all_durations = []
    task_type_durations = defaultdict(list)
    failed_files = []
    
    for audio_info in tqdm(audio_files, desc="Analyzing audio lengths", ascii=True):
        duration = analyze_audio_length(audio_info["path"])
        
        if duration is not None:
            all_durations.append(duration)
            task_type = audio_info.get("task_type", "unknown")
            task_type_durations[task_type].append(duration)
        else:
            failed_files.append(audio_info)
    
    print()
    
    print("Calculating statistics...")
    
    overall_stats = calculate_statistics(all_durations, "overall")
    
    task_type_stats = {}
    for task_type, durations in task_type_durations.items():
        if durations:
            task_type_stats[task_type] = calculate_statistics(durations, task_type)
    
    duration_distribution = get_duration_distribution(all_durations)
    
    analysis_results = {
        "dataset_info": {
            "qa_json_file": qa_json_file,
            "audio_base_dir": audio_base_dir,
            "json_format": json_metadata.get("json_format", "unknown"),
            "json_metadata": json_metadata,
            "total_files_from_json": len(audio_files),
            "successfully_analyzed": len(all_durations),
            "failed_files": len(failed_files),
            "task_types": list(task_type_durations.keys()),
            "task_type_counts": {k: len(v) for k, v in task_type_durations.items()}
        },
        "overall_statistics": overall_stats,
        "task_type_statistics": task_type_stats,
        "duration_distribution": duration_distribution,
        "failed_files": [
            {
                "filename": f["filename"],
                "path": f["path"],
                "task_type": f.get("task_type", "unknown")
            } for f in failed_files
        ]
    }
    
    output_file = os.path.join(result_dir, 'desed_audio_length_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("📊 DESED Dataset Audio Length Statistics Report")
    print("="*70)
    
    print(f"\n📁 Dataset Information:")
    print(f"   QA JSON file: {qa_json_file}")
    print(f"   Audio base directory: {audio_base_dir}")
    print(f"   JSON format: {json_metadata.get('json_format', 'unknown')}")
    
    if json_metadata.get("task_info"):
        print(f"   Task info: {json_metadata['task_info']}")
    if json_metadata.get("missing_files_info"):
        print(f"   Missing files info: {json_metadata['missing_files_info']}")
    
    print(f"   Tasks in JSON: {len(audio_files)}")
    print(f"   Successfully analyzed: {len(all_durations)}")
    print(f"   Analysis failed: {len(failed_files)}")
    print(f"   Number of task types: {len(task_type_durations)}")
    
    if overall_stats:
        print(f"\n🎵 Overall Audio Length Statistics:")
        print(f"   Sample count: {overall_stats['sample_count']}")
        print(f"   Mean duration: {format_duration(overall_stats['mean_duration'])}")
        print(f"   Median duration: {format_duration(overall_stats['median_duration'])}")
        print(f"   Min duration: {format_duration(overall_stats['min_duration'])}")
        print(f"   Max duration: {format_duration(overall_stats['max_duration'])}")
        print(f"   Standard deviation: {overall_stats['std_duration']:.2f} seconds")
        print(f"   Total duration: {format_duration(overall_stats['total_duration'])}")
        print(f"   25th percentile: {format_duration(overall_stats['percentile_25'])}")
        print(f"   75th percentile: {format_duration(overall_stats['percentile_75'])}")
        print(f"   90th percentile: {format_duration(overall_stats['percentile_90'])}")
        print(f"   95th percentile: {format_duration(overall_stats['percentile_95'])}")
        print(f"   99th percentile: {format_duration(overall_stats['percentile_99'])}")
    
    if task_type_stats:
        print(f"\n📋 Audio Length Statistics by Task Type:")
        for task_type, stats in task_type_stats.items():
            print(f"\n  📌 {task_type}:")
            print(f"     Sample count: {stats['sample_count']}")
            print(f"     Mean duration: {format_duration(stats['mean_duration'])}")
            print(f"     Median duration: {format_duration(stats['median_duration'])}")
            print(f"     Min duration: {format_duration(stats['min_duration'])}")
            print(f"     Max duration: {format_duration(stats['max_duration'])}")
            print(f"     Total duration: {format_duration(stats['total_duration'])}")
    
    if duration_distribution:
        print(f"\n📈 Audio Length Distribution:")
        for range_label, dist_info in duration_distribution.items():
            count = dist_info['count']
            percentage = dist_info['percentage']
            if count > 0:
                print(f"   {range_label}: {count} files ({percentage:.1f}%)")
    
    print(f"\n🏷️  Task Type Distribution:")
    task_type_counts = analysis_results["dataset_info"]["task_type_counts"]
    total_tasks = sum(task_type_counts.values())
    for task_type, count in task_type_counts.items():
        percentage = count / total_tasks * 100 if total_tasks > 0 else 0
        print(f"   {task_type}: {count} tasks ({percentage:.1f}%)")
    
    if failed_files:
        print(f"\n⚠️  Failed Processing Files:")
        for failed_file in failed_files[:10]:
            print(f"   - {failed_file['filename']} (task type: {failed_file.get('task_type', 'unknown')})")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more files")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*70)
    
    if all_durations:
        short_audios = [d for d in all_durations if d < 10]
        long_audios = [d for d in all_durations if d > 60]
        
        print(f"\n📊 Special Length Audio Statistics:")
        print(f"   Short audio (<10s): {len(short_audios)} files ({len(short_audios)/len(all_durations)*100:.1f}%)")
        if short_audios:
            print(f"     Mean duration: {format_duration(np.mean(short_audios))}")
        
        print(f"   Long audio (>1min): {len(long_audios)} files ({len(long_audios)/len(all_durations)*100:.1f}%)")
        if long_audios:
            print(f"     Mean duration: {format_duration(np.mean(long_audios))}")

if __name__ == "__main__":
    main()