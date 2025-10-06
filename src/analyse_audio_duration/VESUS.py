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
import traceback

warnings.filterwarnings("ignore")

def load_vesus_for_analysis(json_file_path, data_path):
    if not os.path.exists(json_file_path):
        print(f"Error: Dataset file does not exist: {json_file_path}")
        return [], {}
    
    print(f"Loading VESUS emotion dataset: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        emotion_stats = defaultdict(int)
        person_stats = defaultdict(int)
        person_emotion_stats = defaultdict(lambda: defaultdict(int))
        
        valid_samples = []
        filtered_count = 0
        missing_files = 0
        
        for item in data:
            if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
                person_id = item.get('person_id', '')
                emotion_label = item.get('emotion_label', '').lower()
                
                if (person_id in ['person2', 'person10'] and emotion_label == 'happy'):
                    filtered_count += 1
                    print(f"Filtered sample: {person_id} - {emotion_label} - {item.get('path', '')}")
                    continue
                
                audio_path = item.get("path", "")
                full_audio_path = os.path.join(data_path, audio_path)
                
                if not os.path.exists(full_audio_path):
                    missing_files += 1
                    if missing_files <= 5:
                        print(f"Warning: Audio file does not exist: {full_audio_path}")
                    continue
                
                enhanced_item = item.copy()
                enhanced_item.update({
                    "filename": os.path.basename(audio_path),
                    "full_path": full_audio_path,
                    "emotion_label": emotion_label,
                    "person_id": person_id
                })
                
                valid_samples.append(enhanced_item)
                
                emotion_stats[emotion_label] += 1
                person_stats[person_id] += 1
                person_emotion_stats[person_id][emotion_label] += 1
        
        if missing_files > 5:
            print(f"Warning: Total of {missing_files} audio files do not exist")
        
        metadata_info = {
            "original_samples": len(data),
            "filtered_samples": filtered_count,
            "missing_files": missing_files,
            "valid_samples": len(valid_samples),
            "emotion_distribution": dict(emotion_stats),
            "person_distribution": dict(person_stats),
            "person_emotion_distribution": {person: dict(emotions) for person, emotions in person_emotion_stats.items()},
            "emotions": list(emotion_stats.keys()),
            "persons": list(person_stats.keys())
        }
        
        print(f"Filtered {filtered_count} samples (happy emotion from person2 and person10)")
        print(f"Loaded {len(valid_samples)} valid samples")
        print(f"Emotion distribution: {dict(emotion_stats)}")
        print(f"Person distribution: {dict(person_stats)}")
        
        return valid_samples, metadata_info
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        traceback.print_exc()
        return [], {}

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
    
    bins = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, float('inf')]
    bin_labels = [
        '0-1s', '1-2s', '2-3s', '3-5s', '5-8s', '8-10s',
        '10-15s', '15-20s', '20-30s', '30-45s', '45-60s',
        '1-1.5min', '1.5-2min', '>2min'
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

def analyze_emotion_patterns(audio_files):
    emotion_durations = defaultdict(list)
    
    for audio_info in audio_files:
        if 'duration' in audio_info and audio_info['duration'] is not None:
            emotion = audio_info.get('emotion_label', 'unknown')
            emotion_durations[emotion].append(audio_info['duration'])
    
    emotion_stats = {}
    for emotion, durations in emotion_durations.items():
        if durations:
            emotion_stats[emotion] = {
                "sample_count": len(durations),
                "mean_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "std_duration": np.std(durations),
                "total_duration": np.sum(durations)
            }
    
    return emotion_stats

def analyze_person_patterns(audio_files):
    person_durations = defaultdict(list)
    
    for audio_info in audio_files:
        if 'duration' in audio_info and audio_info['duration'] is not None:
            person = audio_info.get('person_id', 'unknown')
            person_durations[person].append(audio_info['duration'])
    
    person_stats = {}
    for person, durations in person_durations.items():
        if durations:
            person_stats[person] = {
                "sample_count": len(durations),
                "mean_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "std_duration": np.std(durations),
                "total_duration": np.sum(durations)
            }
    
    return person_stats

def main():
    data_path = "/data/to/your/dataset/path//VESUS"
    json_file_path = os.path.join(data_path, "audio_emotion_dataset.json")
    
    result_dir = './VESUS_Analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== VESUS Dataset Audio Length Analysis ===")
    print(f"Data directory: {data_path}")
    print(f"JSON file: {json_file_path}")
    print(f"Results save path: {result_dir}")
    print()
    
    if not os.path.exists(json_file_path):
        print(f"Error: VESUS dataset JSON file does not exist: {json_file_path}")
        return
    
    if not os.path.exists(data_path):
        print(f"Error: VESUS data directory does not exist: {data_path}")
        return
    
    print("Loading audio information from VESUS dataset...")
    audio_files, metadata_info = load_vesus_for_analysis(json_file_path, data_path)
    
    if not audio_files:
        print("Error: No valid audio files found")
        return
    
    print()
    print("Analyzing audio lengths...")
    
    all_durations = []
    emotion_durations = defaultdict(list)
    person_durations = defaultdict(list)
    failed_files = []
    
    for audio_info in tqdm(audio_files, desc="Analyzing audio lengths", ascii=True):
        audio_full_path = audio_info["full_path"]
        
        duration = analyze_audio_length(audio_full_path)
        
        if duration is not None:
            all_durations.append(duration)
            
            emotion = audio_info.get("emotion_label", "unknown")
            emotion_durations[emotion].append(duration)
            
            person = audio_info.get("person_id", "unknown")
            person_durations[person].append(duration)
            
            audio_info["duration"] = duration
        else:
            failed_files.append({
                "filename": audio_info["filename"],
                "path": audio_info["full_path"],
                "emotion": audio_info.get("emotion_label", "unknown"),
                "person": audio_info.get("person_id", "unknown"),
                "reason": "Audio processing failed"
            })
    
    print()
    
    print("Calculating statistics...")
    
    overall_stats = calculate_statistics(all_durations, "overall")
    
    emotion_stats = {}
    for emotion, durations in emotion_durations.items():
        if durations:
            emotion_stats[emotion] = calculate_statistics(durations, emotion)
    
    person_stats = {}
    for person, durations in person_durations.items():
        if durations:
            person_stats[person] = calculate_statistics(durations, person)
    
    duration_distribution = get_duration_distribution(all_durations)
    
    outliers, outlier_info = detect_outliers(all_durations)
    
    emotion_patterns = analyze_emotion_patterns(audio_files)
    
    person_patterns = analyze_person_patterns(audio_files)
    
    analysis_results = {
        "dataset_info": {
            "data_path": data_path,
            "json_file": json_file_path,
            "metadata_info": metadata_info,
            "total_files_found": len(audio_files),
            "successfully_analyzed": len(all_durations),
            "failed_files": len(failed_files),
            "emotions": list(emotion_durations.keys()),
            "persons": list(person_durations.keys()),
            "emotion_counts": {k: len(v) for k, v in emotion_durations.items()},
            "person_counts": {k: len(v) for k, v in person_durations.items()}
        },
        "overall_statistics": overall_stats,
        "emotion_statistics": emotion_stats,
        "person_statistics": person_stats,
        "emotion_patterns": emotion_patterns,
        "person_patterns": person_patterns,
        "duration_distribution": duration_distribution,
        "outlier_analysis": {
            "outliers": outliers,
            "outlier_info": outlier_info,
            "outlier_count": len(outliers)
        },
        "failed_files": failed_files
    }
    
    output_file = os.path.join(result_dir, 'vesus_audio_length_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("🎭 VESUS Dataset Audio Length Statistics Report")
    print("="*80)
    
    print(f"\n📁 Dataset Information:")
    print(f"   Data directory: {data_path}")
    print(f"   JSON file: {json_file_path}")
    print(f"   Original samples: {metadata_info.get('original_samples', 0)}")
    print(f"   Filtered samples: {metadata_info.get('filtered_samples', 0)}")
    print(f"   Missing files: {metadata_info.get('missing_files', 0)}")
    print(f"   Valid samples: {len(audio_files)}")
    print(f"   Successfully analyzed: {len(all_durations)}")
    print(f"   Analysis failed: {len(failed_files)}")
    print(f"   Number of emotion types: {len(emotion_durations)}")
    print(f"   Number of persons: {len(person_durations)}")
    
    if overall_stats:
        print(f"\n🎭 Overall Audio Length Statistics:")
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
    
    if emotion_stats:
        print(f"\n😊 Audio Length Statistics by Emotion Type:")
        for emotion, stats in emotion_stats.items():
            print(f"\n  🎭 {emotion.upper()}:")
            print(f"     Sample count: {stats['sample_count']}")
            print(f"     Mean duration: {format_duration(stats['mean_duration'])}")
            print(f"     Median duration: {format_duration(stats['median_duration'])}")
            print(f"     Min duration: {format_duration(stats['min_duration'])}")
            print(f"     Max duration: {format_duration(stats['max_duration'])}")
            print(f"     Total duration: {format_duration(stats['total_duration'])}")
    
    if person_stats:
        print(f"\n👥 Audio Length Statistics by Person (Top 10):")
        sorted_persons = sorted(person_stats.items(), 
                               key=lambda x: x[1]['sample_count'], reverse=True)
        
        for i, (person, stats) in enumerate(sorted_persons[:10]):
            print(f"\n  👤 {person}:")
            print(f"     Sample count: {stats['sample_count']}")
            print(f"     Mean duration: {format_duration(stats['mean_duration'])}")
            print(f"     Median duration: {format_duration(stats['median_duration'])}")
            print(f"     Duration range: {format_duration(stats['min_duration'])} - {format_duration(stats['max_duration'])}")
            print(f"     Total duration: {format_duration(stats['total_duration'])}")
    
    if duration_distribution:
        print(f"\n📈 Audio Length Distribution:")
        for range_label, dist_info in duration_distribution.items():
            count = dist_info['count']
            percentage = dist_info['percentage']
            if count > 0:
                print(f"   {range_label}: {count} files ({percentage:.1f}%)")
    
    emotion_counts = analysis_results["dataset_info"]["emotion_counts"]
    person_counts = analysis_results["dataset_info"]["person_counts"]
    
    print(f"\n🏷️  Emotion Distribution:")
    total_emotion_samples = sum(emotion_counts.values())
    for emotion, count in emotion_counts.items():
        percentage = count / total_emotion_samples * 100 if total_emotion_samples > 0 else 0
        print(f"   {emotion}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n👥 Person Distribution (Top 10):")
    sorted_person_counts = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)
    total_person_samples = sum(person_counts.values())
    for person, count in sorted_person_counts[:10]:
        percentage = count / total_person_samples * 100 if total_person_samples > 0 else 0
        print(f"   {person}: {count} samples ({percentage:.1f}%)")
    
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
            print(f"   Outlier samples (First 5):")
            for sample in outlier_samples[:5]:
                duration = sample.get('duration', 0)
                emotion = sample.get('emotion_label', 'unknown')
                person = sample.get('person_id', 'unknown')
                filename = sample.get('filename', 'unknown')
                print(f"     - {filename} ({person}, {emotion}): {format_duration(duration)}")
    
    if failed_files:
        print(f"\n❌ Failed Processing Files:")
        for failed_file in failed_files[:10]:
            print(f"   - {failed_file['filename']} ({failed_file['person']}, {failed_file['emotion']}) - {failed_file['reason']}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more files")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*80)
    
    if all_durations:
        short_audios = [d for d in all_durations if d < 3]
        medium_audios = [d for d in all_durations if 3 <= d <= 10]
        long_audios = [d for d in all_durations if d > 15]
        
        print(f"\n📏 VESUS Emotional Speech Length Analysis:")
        print(f"   Short audio (<3s): {len(short_audios)} files ({len(short_audios)/len(all_durations)*100:.1f}%)")
        if short_audios:
            print(f"     Average length: {format_duration(np.mean(short_audios))}")
        
        print(f"   Medium audio (3-10s): {len(medium_audios)} files ({len(medium_audios)/len(all_durations)*100:.1f}%)")
        if medium_audios:
            print(f"     Average length: {format_duration(np.mean(medium_audios))}")
            
        print(f"   Long audio (>15s): {len(long_audios)} files ({len(long_audios)/len(all_durations)*100:.1f}%)")
        if long_audios:
            print(f"     Average length: {format_duration(np.mean(long_audios))}")
    
    if emotion_patterns:
        print(f"\n🎭 Emotional Speech Characteristics Analysis:")
        emotion_avg_durations = {emotion: patterns['mean_duration'] 
                               for emotion, patterns in emotion_patterns.items()}
        
        shortest_emotion = min(emotion_avg_durations.items(), key=lambda x: x[1])
        longest_emotion = max(emotion_avg_durations.items(), key=lambda x: x[1])
        
        print(f"   Shortest average emotion: {shortest_emotion[0]} ({format_duration(shortest_emotion[1])})")
        print(f"   Longest average emotion: {longest_emotion[0]} ({format_duration(longest_emotion[1])})")
        
        analysis_notes = []
        if 'angry' in emotion_avg_durations and emotion_avg_durations['angry'] > np.mean(list(emotion_avg_durations.values())):
            analysis_notes.append("Angry emotional speech has longer duration")
        if 'sad' in emotion_avg_durations and emotion_avg_durations['sad'] > np.mean(list(emotion_avg_durations.values())):
            analysis_notes.append("Sad emotional speech has longer duration")
        if 'happy' in emotion_avg_durations and emotion_avg_durations['happy'] < np.mean(list(emotion_avg_durations.values())):
            analysis_notes.append("Happy emotional speech has shorter duration")
        
        if analysis_notes:
            print(f"   Emotional characteristic observations:")
            for note in analysis_notes:
                print(f"     - {note}")
    
    if overall_stats:
        quality_issues = []
        
        if overall_stats['min_duration'] < 0.5:
            quality_issues.append("Audio files too short exist (<0.5s)")
        
        if overall_stats['max_duration'] > 120:
            quality_issues.append("Audio files too long exist (>2min)")
        
        if overall_stats['std_duration'] > overall_stats['mean_duration']:
            quality_issues.append("Audio length distribution has high variance")
        
        if metadata_info.get('filtered_samples', 0) > metadata_info.get('valid_samples', 0) * 0.1:
            quality_issues.append("High proportion of filtered samples")
        
        if quality_issues:
            print(f"\n⚠️  Data Quality Reminders:")
            for issue in quality_issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ Data quality is good, suitable for emotion recognition model training and evaluation")

if __name__ == "__main__":
    main()