import os
import json
import glob
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import sys
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

def load_desed_qa_dataset_for_analysis(json_file, audio_base_dir):
    dataset = []
    metadata = {}
    
    if not os.path.exists(json_file):
        print(f"Error: DESED JSON file does not exist: {json_file}")
        return [], {}
    
    print(f"Loading DESED QA format JSON: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read DESED JSON file: {e}")
        return [], {}
    
    if isinstance(data, dict) and "tasks" in data:
        print("Detected new format JSON file")
        metadata = {
            "task_info": data.get("task_info", {}),
            "missing_files_info": data.get("missing_files_info", {}),
            "json_format": "structured"
        }
        tasks = data.get("tasks", [])
    elif isinstance(data, list):
        print("Detected old format JSON file: direct list format")
        tasks = data
        metadata = {"json_format": "list"}
    else:
        print(f"Error: DESED JSON file format is incorrect")
        return [], {}
    
    print(f"Loaded {len(tasks)} tasks from DESED JSON")
    
    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            continue
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: DESED audio file does not exist: {full_audio_path}")
            continue
        
        task_type = task.get("task_type", "unknown")
        
        item = {
            "path": full_audio_path,
            "filename": os.path.basename(full_audio_path),
            "dataset": "DESED",
            "task_type": task_type,
            "uniq_id": task.get("uniq_id", i),
        }
        
        dataset.append(item)
        task_type_stats[task_type] += 1
    
    if missing_files > 3:
        print(f"Warning: DESED has {missing_files} audio files that do not exist")
    
    print(f"DESED loaded {len(dataset)} valid samples")
    return dataset, metadata

def load_gtzan_metadata_for_analysis(metadata_path, data_path_root):
    dataset = []
    metadata_info = {}
    
    if not os.path.exists(metadata_path):
        print(f"Error: GTZAN metadata file does not exist: {metadata_path}")
        return [], {}
    
    print(f"Loading GTZAN metadata file: {metadata_path}")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read GTZAN metadata file: {e}")
        return [], {}
    
    if not isinstance(data, list):
        print(f"Error: GTZAN metadata file format is incorrect, expected list format")
        return [], {}
    
    print(f"Loaded {len(data)} music samples from GTZAN metadata")
    
    genre_stats = defaultdict(int)
    missing_files = 0
    
    for i, sample in enumerate(data):
        required_fields = ["path", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer_gt"]
        if not all(field in sample for field in required_fields):
            continue
        
        audio_rel_path = sample["path"]
        audio_full_path = os.path.join(data_path_root, audio_rel_path)
        
        if not os.path.exists(audio_full_path):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: GTZAN audio file does not exist: {audio_full_path}")
            continue
        
        genre_label = sample.get("genre_label", "unknown")
        
        item = {
            "path": audio_full_path,
            "filename": os.path.basename(audio_full_path),
            "dataset": "GTZAN",
            "genre_label": genre_label,
            "uniq_id": sample.get("uniq_id", i),
        }
        
        dataset.append(item)
        genre_stats[genre_label] += 1
    
    if missing_files > 3:
        print(f"Warning: GTZAN has {missing_files} audio files that do not exist")
    
    metadata_info = {
        "total_samples": len(dataset),
        "genre_distribution": dict(genre_stats),
        "genres": list(genre_stats.keys())
    }
    
    print(f"GTZAN loaded {len(dataset)} valid samples")
    return dataset, metadata_info

def load_had_dataset_for_analysis(root_dir):
    dataset = []
    metadata_info = {}
    
    if not os.path.exists(root_dir):
        print(f"Error: HAD dataset root directory does not exist: {root_dir}")
        return [], {}
    
    print(f"Loading HAD dataset: {root_dir}")
    
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")
    
    label_stats = defaultdict(int)
    missing_dirs = []
    
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for wav_path in real_files:
            item = {
                "path": wav_path,
                "filename": os.path.basename(wav_path),
                "dataset": "HAD",
                "label": "real",
                "uniq_id": f"real_{len(dataset)}"
            }
            dataset.append(item)
            label_stats["real"] += 1
        print(f"Found {len(real_files)} real audio files")
    else:
        print(f"Warning: HAD real audio directory does not exist: {real_dir}")
        missing_dirs.append("real")
    
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for wav_path in fake_files:
            item = {
                "path": wav_path,
                "filename": os.path.basename(wav_path),
                "dataset": "HAD",
                "label": "fake",
                "uniq_id": f"fake_{len(dataset)}"
            }
            dataset.append(item)
            label_stats["fake"] += 1
        print(f"Found {len(fake_files)} fake audio files")
    else:
        print(f"Warning: HAD fake audio directory does not exist: {fake_dir}")
        missing_dirs.append("fake")
    
    metadata_info = {
        "total_samples": len(dataset),
        "label_distribution": dict(label_stats),
        "labels": list(label_stats.keys()),
        "missing_directories": missing_dirs
    }
    
    print(f"HAD loaded {len(dataset)} valid samples")
    return dataset, metadata_info

def load_librispeech_for_analysis(base_dir, split="test-clean"):
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Error: LibriSpeech dataset path does not exist: {split_dir}")
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
                        missing_files += 1
                        if missing_files <= 3:
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
                        missing_files += 1
                        if missing_files <= 3:
                            print(f"Failed to read trans file {trans_file}: {e}")
                        continue
                
                if transcription:
                    try:
                        audio_info = sf.info(flac_file)
                        
                        item = {
                            "path": flac_file,
                            "filename": os.path.basename(flac_file),
                            "dataset": "LibriSpeech",
                            "transcription": transcription,
                            "duration": audio_info.duration,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "language": "en",
                            "id": f"{speaker_id}_{chapter_id}_{base_name}",
                        }
                        
                        dataset.append(item)
                        speaker_stats[speaker_id] += 1
                        chapter_stats[f"{speaker_id}-{chapter_id}"] += 1
                        
                    except Exception as e:
                        missing_files += 1
                        if missing_files <= 3:
                            print(f"Cannot process audio file {flac_file}: {e}")
                        continue
                else:
                    missing_files += 1
                    if missing_files <= 3:
                        print(f"Transcription file not found: {base_name}")
    
    if missing_files > 3:
        print(f"Warning: LibriSpeech has {missing_files} files that cannot be processed")
    
    metadata_info = {
        "split": split,
        "total_samples": len(dataset),
        "speaker_count": len(speaker_stats),
        "chapter_count": len(chapter_stats),
        "speaker_distribution": dict(speaker_stats),
        "missing_files": missing_files
    }
    
    print(f"Loaded {len(dataset)} valid audio samples from LibriSpeech {split}")
    return dataset, metadata_info

def load_race_dataset_for_analysis(data_path_root):
    dataset = []
    metadata_info = {}
    
    bench_path = os.path.join(data_path_root, "race_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"Error: Cannot find RACE benchmark file: {bench_path}")
        return [], {}
    
    print(f"Loading RACE benchmark file: {bench_path}")
    
    try:
        with open(bench_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
    except Exception as e:
        print(f"Failed to read RACE benchmark file: {e}")
        return [], {}
    
    print(f"Loaded {len(benchmark_data)} sample info from RACE benchmark")
    
    missing_files = 0
    audio_info_errors = 0
    difficulty_stats = defaultdict(int)
    
    for i, sample in enumerate(benchmark_data):
        audio_rel = sample.get("audio_path", "")
        if not audio_rel:
            continue
            
        audio_full = os.path.join(data_path_root, audio_rel)
        
        if not os.path.exists(audio_full):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: RACE file does not exist {audio_full}")
            continue
        
        try:
            audio_info = sf.info(audio_full)
            duration = audio_info.duration
        except Exception as e:
            audio_info_errors += 1
            if audio_info_errors <= 3:
                print(f"Warning: Cannot get RACE audio info {audio_full}: {e}")
            continue
        
        difficulty = "unknown"
        if "high" in audio_rel.lower():
            difficulty = "high"
        elif "middle" in audio_rel.lower():
            difficulty = "middle"
        
        item = {
            "path": audio_full,
            "filename": os.path.basename(audio_full),
            "dataset": "RACE",
            "relative_path": audio_rel,
            "difficulty": difficulty,
            "duration": duration,
            "uniq_id": sample.get("id", f"race_sample_{i}"),
            "task": "Reading_Comprehension"
        }
        
        dataset.append(item)
        difficulty_stats[difficulty] += 1
    
    if missing_files > 3:
        print(f"Warning: RACE has {missing_files} audio files that do not exist")
    if audio_info_errors > 3:
        print(f"Warning: RACE has {audio_info_errors} audio files with failed info retrieval")
    
    metadata_info = {
        "total_samples": len(dataset),
        "difficulty_distribution": dict(difficulty_stats),
        "difficulties": list(difficulty_stats.keys()),
        "missing_files": missing_files,
        "audio_info_errors": audio_info_errors
    }
    
    print(f"RACE loaded {len(dataset)} valid samples")
    return dataset, metadata_info

def load_slue_dataset_for_analysis(json_file, audio_base_dir):
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: SLUE JSON file does not exist: {json_file}")
        return [], {}
    
    print(f"Loading SLUE JSON file: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read SLUE JSON file: {e}")
        return [], {}
    
    if not isinstance(data, list):
        print(f"Error: SLUE JSON file format is incorrect, expected list format")
        return [], {}
    
    print(f"Loaded {len(data)} tasks from SLUE JSON")
    
    task_type_stats = defaultdict(int)
    dataset_stats = defaultdict(int)
    missing_files = 0
    audio_info_errors = 0
    
    for i, task in enumerate(data):
        relative_path = task.get("path", "")
        if not relative_path:
            continue

        full_audio_path = os.path.join(audio_base_dir, relative_path)
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: SLUE audio file does not exist: {full_audio_path}")
            continue
        
        try:
            audio_info = sf.info(full_audio_path)
            duration = audio_info.duration
        except Exception as e:
            audio_info_errors += 1
            if audio_info_errors <= 3:
                print(f"Warning: Cannot get SLUE audio info {full_audio_path}: {e}")
            continue

        item = {
            "path": full_audio_path,
            "relative_path": relative_path,
            "filename": os.path.basename(full_audio_path),
            "dataset": "SLUE",
            "task_name": task.get("task_name", "unknown"),
            "dataset_name": task.get("dataset_name", "unknown"),
            "duration": duration,
            "id": f"slue_task_{task.get('uniq_id', i)}"
        }
        
        dataset.append(item)
        task_type_stats[item["task_name"]] += 1
        dataset_stats[item["dataset_name"]] += 1
    
    if missing_files > 3:
        print(f"Warning: SLUE has {missing_files} audio files that do not exist")
    if audio_info_errors > 3:
        print(f"Warning: SLUE has {audio_info_errors} audio files with failed info retrieval")
    
    metadata_info = {
        "total_samples": len(dataset),
        "task_type_distribution": dict(task_type_stats),
        "dataset_distribution": dict(dataset_stats),
        "missing_files": missing_files,
        "audio_info_errors": audio_info_errors
    }
    
    print(f"SLUE loaded {len(dataset)} valid samples")
    return dataset, metadata_info

def load_tau_dataset_for_analysis(root_dir):
    dataset = []
    metadata_info = {}
    
    meta_file = os.path.join(root_dir, "acoustic_scene_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Cannot find TAU metadata file: {meta_file}")
        return [], {}
    
    print(f"Loading TAU metadata file: {meta_file}")
    
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Failed to read TAU metadata file: {e}")
        return [], {}
    
    print(f"Loaded {len(metadata)} sample info from TAU metadata")
    
    scene_stats = defaultdict(int)
    missing_files = 0
    audio_info_errors = 0
    
    for i, item in enumerate(metadata):
        if "path" in item:
            rel_path = item["path"]
            audio_path = os.path.join(root_dir, "concatenated_resampled", rel_path)
        elif "audio_file" in item:
            audio_path = os.path.join(root_dir, "concatenated_resampled", item["audio_file"])
        else:
            continue
        
        if not os.path.exists(audio_path):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: TAU audio file does not exist: {audio_path}")
            continue
        
        scene_label = item.get("scene_label", "unknown")
        
        try:
            duration = librosa.get_duration(path=audio_path)
        except Exception:
            try:
                info = sf.info(audio_path)
                duration = info.duration
            except Exception as e:
                audio_info_errors += 1
                if audio_info_errors <= 3:
                    print(f"Warning: Cannot get TAU audio info {audio_path}: {e}")
                continue
        
        audio_info_item = {
            "path": audio_path,
            "filename": os.path.basename(audio_path),
            "dataset": "TAU",
            "scene_label": scene_label,
            "duration": duration,
            "sample_id": item.get("id", f"tau_sample_{len(dataset)}")
        }
        
        dataset.append(audio_info_item)
        scene_stats[scene_label] += 1
    
    if missing_files > 3:
        print(f"Warning: TAU has {missing_files} audio files that do not exist")
    if audio_info_errors > 3:
        print(f"Warning: TAU has {audio_info_errors} audio files with failed info retrieval")
    
    metadata_info = {
        "total_samples": len(dataset),
        "scene_distribution": dict(scene_stats),
        "scenes": list(scene_stats.keys()),
        "missing_files": missing_files,
        "audio_info_errors": audio_info_errors
    }
    
    print(f"TAU loaded {len(dataset)} valid samples")
    return dataset, metadata_info

def load_vesus_dataset_for_analysis(json_file_path, data_path):
    dataset = []
    metadata_info = {}
    
    if not os.path.exists(json_file_path):
        print(f"Error: VESUS dataset file does not exist: {json_file_path}")
        return [], {}
    
    print(f"Loading VESUS emotion dataset: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load VESUS dataset: {e}")
        return [], {}
    
    emotion_stats = defaultdict(int)
    person_stats = defaultdict(int)
    
    valid_samples = []
    filtered_count = 0
    missing_files = 0
    
    for item in data:
        if isinstance(item, dict) and all(key in item for key in ['path', 'question', 'answer_gt']):
            person_id = item.get('person_id', '')
            emotion_label = item.get('emotion_label', '').lower()
            
            if (person_id in ['person2', 'person10'] and emotion_label == 'happy'):
                filtered_count += 1
                continue
            
            audio_path = item.get("path", "")
            full_audio_path = os.path.join(data_path, audio_path)
            
            if not os.path.exists(full_audio_path):
                missing_files += 1
                if missing_files <= 3:
                    print(f"Warning: VESUS audio file does not exist: {full_audio_path}")
                continue
            
            enhanced_item = {
                "path": full_audio_path,
                "filename": os.path.basename(audio_path),
                "dataset": "VESUS",
                "emotion_label": emotion_label,
                "person_id": person_id,
                "question": item.get("question", ""),
                "answer_gt": item.get("answer_gt", ""),
                "uniq_id": item.get("uniq_id", len(valid_samples))
            }
            
            valid_samples.append(enhanced_item)
            
            emotion_stats[emotion_label] += 1
            person_stats[person_id] += 1
    
    if missing_files > 3:
        print(f"Warning: VESUS has {missing_files} audio files that do not exist")
    
    metadata_info = {
        "original_samples": len(data),
        "filtered_samples": filtered_count,
        "missing_files": missing_files,
        "valid_samples": len(valid_samples),
        "emotion_distribution": dict(emotion_stats),
        "person_distribution": dict(person_stats),
        "emotions": list(emotion_stats.keys()),
        "persons": list(person_stats.keys())
    }
    
    print(f"VESUS filtered {filtered_count} samples")
    print(f"VESUS loaded {len(valid_samples)} valid samples")
    
    return valid_samples, metadata_info

def load_vox_age_dataset_for_analysis(root_dir):
    dataset = []
    metadata_info = {}
    
    meta_file = os.path.join(root_dir, "age_classification_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Cannot find Vox_age metadata file: {meta_file}")
        return [], {}
    
    print(f"Loading VoxCeleb age classification metadata file: {meta_file}")
    
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Failed to read Vox_age metadata file: {e}")
        return [], {}
    
    print(f"Loaded {len(metadata)} sample info from Vox_age metadata")
    
    missing_files = 0
    audio_info_errors = 0
    age_group_stats = defaultdict(int)
    
    for i, item in enumerate(metadata):
        rel_path = item["path"]
        wav_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(wav_path):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: Vox_age file does not exist {wav_path}")
            continue
        
        try:
            audio_info = sf.info(wav_path)
            duration = audio_info.duration
        except Exception as e:
            audio_info_errors += 1
            if audio_info_errors <= 3:
                print(f"Warning: Cannot get Vox_age audio info {wav_path}: {e}")
            continue
        
        speaker_id = item["speaker_id_original"]
        age_group = item["answer_gt"].strip()
        speaker_age = item.get("speaker_age", 0)
        
        sample_data = {
            "path": wav_path,
            "filename": os.path.basename(wav_path),
            "dataset": "Vox_age",
            "speaker_id": speaker_id,
            "age_group": age_group,
            "speaker_age": speaker_age,
            "duration": duration,
            "task": "Speaker_Age_Classification",
            "id": f"vox_age_{item.get('uniq_id', i)}"
        }
        
        dataset.append(sample_data)
        age_group_stats[age_group] += 1
    
    if missing_files > 3:
        print(f"Warning: Vox_age has {missing_files} audio files that do not exist")
    if audio_info_errors > 3:
        print(f"Warning: Vox_age has {audio_info_errors} audio files with failed info retrieval")
    
    metadata_info = {
        "total_samples": len(dataset),
        "age_group_distribution": dict(age_group_stats),
        "age_groups": list(age_group_stats.keys()),
        "missing_files": missing_files,
        "audio_info_errors": audio_info_errors
    }
    
    print(f"Vox_age loaded {len(dataset)} valid samples")
    return dataset, metadata_info

def load_vox_gender_dataset_for_analysis(root_dir):
    dataset = []
    metadata_info = {}
    
    meta_file = os.path.join(root_dir, "gender_id_task_meta.json")
    if not os.path.exists(meta_file):
        print(f"Error: Cannot find Vox_gender metadata file: {meta_file}")
        return [], {}
    
    print(f"Loading VoxCeleb gender classification metadata file: {meta_file}")
    
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Failed to read Vox_gender metadata file: {e}")
        return [], {}
    
    print(f"Loaded {len(metadata)} sample info from Vox_gender metadata")
    
    missing_files = 0
    audio_info_errors = 0
    gender_stats = defaultdict(int)
    speaker_stats = defaultdict(int)
    
    for i, item in enumerate(metadata):
        rel_path = item["path"]
        audio_path = os.path.join(root_dir, "wav", rel_path)
        
        if not os.path.exists(audio_path):
            missing_files += 1
            if missing_files <= 3:
                print(f"Warning: Vox_gender file does not exist {audio_path}")
            continue
        
        speaker_id = item["speaker_id_original"]
        gender = item["answer_gt"].lower().strip()
        
        try:
            duration = librosa.get_duration(path=audio_path)
        except Exception:
            try:
                info = sf.info(audio_path)
                duration = info.duration
            except Exception as e:
                audio_info_errors += 1
                if audio_info_errors <= 3:
                    print(f"Warning: Cannot get Vox_gender audio info {audio_path}: {e}")
                continue
        
        audio_info = {
            "path": audio_path,
            "filename": os.path.basename(audio_path),
            "dataset": "Vox_gender",
            "relative_path": rel_path,
            "speaker_id": speaker_id,
            "gender": gender,
            "duration": duration,
            "sample_id": item.get("id", f"vox_gender_{len(dataset)}")
        }
        
        dataset.append(audio_info)
        gender_stats[gender] += 1
        speaker_stats[speaker_id] += 1
    
    if missing_files > 3:
        print(f"Warning: Vox_gender has {missing_files} audio files that do not exist")
    if audio_info_errors > 3:
        print(f"Warning: Vox_gender has {audio_info_errors} audio files with failed info retrieval")
    
    metadata_info = {
        "total_samples": len(dataset),
        "gender_distribution": dict(gender_stats),
        "speaker_count": len(speaker_stats),
        "genders": list(gender_stats.keys()),
        "missing_files": missing_files,
        "audio_info_errors": audio_info_errors
    }
    
    print(f"Vox_gender loaded {len(dataset)} valid samples")
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
                return None

def get_audio_path(audio_info):
    dataset_name = audio_info["dataset"]
    
    return audio_info.get("path")

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
    
    bins = [0, 2, 5, 10, 15, 20, 25, 30, 35, 45, 60, 90, 120, 180, 300, 600, float('inf')]
    bin_labels = [
        '0-2s', '2-5s', '5-10s', '10-15s', '15-20s', '20-25s', '25-30s',
        '30-35s', '35-45s', '45-60s', '1-1.5min', '1.5-2min', 
        '2-3min', '3-5min', '5-10min', '10min+'
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
    print("=== Combined Audio Length Analysis for Ten Datasets ===")
    print("DESED + GTZAN + HAD + LibriSpeech + RACE + SLUE + TAU + VESUS + Vox_age + Vox_gender")
    print()
    
    desed_qa_json_file = "/data/to/your/dataset/path//DESED/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json"
    desed_audio_base_dir = "/data/to/your/dataset/path//DESED/DESED_dataset/concatenated_audio"
    
    gtzan_data_path_root = "/data/to/your/dataset/path//GTZAN/concatenated_audio/wav"
    gtzan_metadata_file = "/data/to/your/dataset/path//GTZAN/concatenated_audio/music_genre_classification_meta.json"
    
    had_data_path_root = "/data/to/your/dataset/path//HAD/concatenated_audio"
    
    librispeech_path = "/data/to/your/dataset/path//librispeech-long"
    librispeech_split = "test-clean"
    
    race_data_path_root = "/data/to/your/dataset/path//race_audio"
    
    slue_json_file = "/data/to/your/dataset/path//SLUE/merged_audio_data.json"
    slue_audio_base_dir = "/data/to/your/dataset/path//SLUE"
    
    tau_data_root = "/data/to/your/dataset/path//TAU"
    
    vesus_data_path = "/data/to/your/dataset/path//VESUS"
    vesus_json_file_path = os.path.join(vesus_data_path, "audio_emotion_dataset.json")
    
    vox_age_data_path_root = "/data/to/your/dataset/path//VoxCeleb/concatenated_audio"
    
    vox_gender_data_path_root = "/data/to/your/dataset/path//VoxCeleb/concatenated_audio"
    
    result_dir = './Combined_Ten_Datasets_Analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"📁 Dataset paths:")
    print(f"   DESED: {desed_qa_json_file}")
    print(f"   GTZAN: {gtzan_metadata_file}")
    print(f"   HAD: {had_data_path_root}")
    print(f"   LibriSpeech: {librispeech_path}")
    print(f"   RACE: {race_data_path_root}")
    print(f"   SLUE: {slue_json_file}")
    print(f"   TAU: {tau_data_root}")
    print(f"   VESUS: {vesus_json_file_path}")
    print(f"   Vox_age: {vox_age_data_path_root}")
    print(f"   Vox_gender: {vox_gender_data_path_root}")
    print(f"📁 Result save path: {result_dir}")
    print()
    
    print("🎵 Loading DESED dataset...")
    desed_files, desed_metadata = load_desed_qa_dataset_for_analysis(desed_qa_json_file, desed_audio_base_dir)
    print()
    
    print("🎼 Loading GTZAN dataset...")
    gtzan_files, gtzan_metadata = load_gtzan_metadata_for_analysis(gtzan_metadata_file, gtzan_data_path_root)
    print()
    
    print("🎙️  Loading HAD dataset...")
    had_files, had_metadata = load_had_dataset_for_analysis(had_data_path_root)
    print()
    
    print("🗣️  Loading LibriSpeech dataset...")
    librispeech_files, librispeech_metadata = load_librispeech_for_analysis(librispeech_path, librispeech_split)
    print()
    
    print("📚 Loading RACE dataset...")
    race_files, race_metadata = load_race_dataset_for_analysis(race_data_path_root)
    print()
    
    print("🎶 Loading SLUE dataset...")
    slue_files, slue_metadata = load_slue_dataset_for_analysis(slue_json_file, slue_audio_base_dir)
    print()
    
    print("🎤 Loading TAU dataset...")
    tau_files, tau_metadata = load_tau_dataset_for_analysis(tau_data_root)
    print()
    
    print("😊 Loading VESUS dataset...")
    vesus_files, vesus_metadata = load_vesus_dataset_for_analysis(vesus_json_file_path, vesus_data_path)
    print()
    
    print("👥 Loading Vox_age dataset...")
    vox_age_files, vox_age_metadata = load_vox_age_dataset_for_analysis(vox_age_data_path_root)
    print()
    
    print("🚻 Loading Vox_gender dataset...")
    vox_gender_files, vox_gender_metadata = load_vox_gender_dataset_for_analysis(vox_gender_data_path_root)
    print()
    
    all_audio_files = (desed_files + gtzan_files + had_files + librispeech_files + race_files + 
                       slue_files + tau_files + vesus_files + vox_age_files + vox_gender_files)
    
    print(f"📊 Total files after merging: {len(all_audio_files)}")
    print(f"   DESED: {len(desed_files)}, GTZAN: {len(gtzan_files)}, HAD: {len(had_files)}")
    print(f"   LibriSpeech: {len(librispeech_files)}, RACE: {len(race_files)}, SLUE: {len(slue_files)}")
    print(f"   TAU: {len(tau_files)}, VESUS: {len(vesus_files)}")
    print(f"   Vox_age: {len(vox_age_files)}, Vox_gender: {len(vox_gender_files)}")
    
    if not all_audio_files:
        print("Error: No valid audio files found")
        return
    
    print()
    
    print("🔍 Analyzing audio lengths...")
    
    dataset_durations = {
        "DESED": [],
        "GTZAN": [],
        "HAD": [],
        "LibriSpeech": [],
        "RACE": [],
        "SLUE": [],
        "TAU": [],
        "VESUS": [],
        "Vox_age": [],
        "Vox_gender": []
    }
    
    all_durations = []
    failed_files = []
    
    for audio_info in tqdm(all_audio_files, desc="Analyzing audio lengths", ascii=True):
        dataset_name = audio_info["dataset"]
        
        audio_path = get_audio_path(audio_info)
        if audio_path is None:
            failed_files.append({
                "filename": audio_info.get("filename", "unknown"),
                "dataset": dataset_name,
                "reason": "Missing path field"
            })
            continue
        
        if dataset_name in ["LibriSpeech", "RACE", "SLUE", "TAU", "Vox_age", "Vox_gender"] and 'duration' in audio_info and audio_info['duration'] is not None:
            duration = audio_info['duration']
        else:
            duration = analyze_audio_length(audio_path)
        
        if duration is not None:
            all_durations.append(duration)
            dataset_durations[dataset_name].append(duration)
        else:
            failed_files.append({
                "filename": audio_info.get("filename", "unknown"),
                "path": audio_path,
                "dataset": dataset_name,
                "reason": "Audio length analysis failed"
            })
    
    print()
    
    print("📈 Calculating statistics...")
    
    overall_stats = calculate_statistics(all_durations, "combined_ten_datasets")
    
    dataset_stats = {}
    for dataset_name, durations in dataset_durations.items():
        if durations:
            dataset_stats[dataset_name] = calculate_statistics(durations, dataset_name)
    
    overall_distribution = get_duration_distribution(all_durations)
    
    analysis_results = {
        "combined_info": {
            "total_files": len(all_audio_files),
            "successfully_analyzed": len(all_durations),
            "failed_files": len(failed_files),
            "dataset_file_counts": {k: len([f for f in all_audio_files if f["dataset"] == k]) for k in dataset_durations.keys()},
            "dataset_analyzed_counts": {k: len(v) for k, v in dataset_durations.items()}
        },
        "metadata_info": {
            "desed": desed_metadata,
            "gtzan": gtzan_metadata,
            "had": had_metadata,
            "librispeech": librispeech_metadata,
            "race": race_metadata,
            "slue": slue_metadata,
            "tau": tau_metadata,
            "vesus": vesus_metadata,
            "vox_age": vox_age_metadata,
            "vox_gender": vox_gender_metadata
        },
        "statistics": {
            "combined_overall": overall_stats,
            "dataset_statistics": dataset_stats
        },
        "distributions": {
            "combined": overall_distribution
        },
        "failed_files": failed_files
    }
    
    output_file = os.path.join(result_dir, 'combined_ten_datasets_audio_length_analysis_fixed.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*100)
    print("🎵🎼🎙️🗣️📚🎶🎤😊👥🚻 Combined Ten Datasets Audio Length Statistics Report [Fixed Version]")
    print("="*100)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total files: {len(all_audio_files)}")
    print(f"   Successfully analyzed: {len(all_durations)}")
    print(f"   Analysis failed: {len(failed_files)}")
    
    for dataset_name in dataset_durations.keys():
        file_count = len([f for f in all_audio_files if f["dataset"] == dataset_name])
        analyzed_count = len(dataset_durations[dataset_name])
        success_rate = (analyzed_count / file_count * 100) if file_count > 0 else 0
        print(f"   {dataset_name} files: {file_count} (Successfully analyzed: {analyzed_count}, Success rate: {success_rate:.1f}%)")
    
    if overall_stats:
        print(f"\n🔥 [Combined Ten Datasets] Overall Audio Length Statistics:")
        print(f"   📈 Sample count: {overall_stats['sample_count']}")
        print(f"   📈 Average audio length: {format_duration(overall_stats['mean_duration'])}")
        print(f"   📈 Median audio length: {format_duration(overall_stats['median_duration'])}")
        print(f"   📈 Shortest audio length: {format_duration(overall_stats['min_duration'])}")
        print(f"   📈 Longest audio length: {format_duration(overall_stats['max_duration'])}")
        print(f"   📈 Standard deviation: {overall_stats['std_duration']:.2f} seconds")
        print(f"   📈 Total duration: {format_duration(overall_stats['total_duration'])}")
    
    dataset_icons = {
        "DESED": "🎵", "GTZAN": "🎼", "HAD": "🎙️", "LibriSpeech": "🗣️", "RACE": "📚",
        "SLUE": "🎶", "TAU": "🎤", "VESUS": "😊", "Vox_age": "👥", "Vox_gender": "🚻"
    }
    
    for dataset_name, stats in dataset_stats.items():
        if stats:
            icon = dataset_icons.get(dataset_name, "📊")
            print(f"\n{icon} [{dataset_name} Dataset] Audio Length Statistics:")
            print(f"   Sample count: {stats['sample_count']}")
            print(f"   Average length: {format_duration(stats['mean_duration'])}")
            print(f"   Median length: {format_duration(stats['median_duration'])}")
            print(f"   Shortest length: {format_duration(stats['min_duration'])}")
            print(f"   Longest length: {format_duration(stats['max_duration'])}")
            print(f"   Total duration: {format_duration(stats['total_duration'])}")
    
    if overall_distribution:
        print(f"\n📈 [Combined Ten Datasets] Audio Length Distribution:")
        for range_label, dist_info in overall_distribution.items():
            count = dist_info['count']
            percentage = dist_info['percentage']
            if count > 0:
                print(f"   {range_label}: {count} files ({percentage:.1f}%)")
    
    if len(dataset_stats) >= 2:
        print(f"\n🔍 Ten Datasets Comparative Analysis:")
        datasets = [(name, stats) for name, stats in dataset_stats.items()]
        
        print(f"   Average length comparison:")
        for name, stats in datasets:
            print(f"     {name}: {format_duration(stats['mean_duration'])}")
        
        print(f"   Median comparison:")
        for name, stats in datasets:
            print(f"     {name}: {format_duration(stats['median_duration'])}")
        
        print(f"   Sample count comparison:")
        for name, stats in datasets:
            print(f"     {name}: {stats['sample_count']} samples")
    
    if failed_files:
        print(f"\n❌ Files that failed processing ({len(failed_files)}):")
        failed_by_dataset = defaultdict(int)
        for f in failed_files:
            failed_by_dataset[f['dataset']] += 1
        
        for dataset_name, count in failed_by_dataset.items():
            if count > 0:
                print(f"   {dataset_name} failed files: {count}")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*100)
    
    print(f"\n🎯 [Final Summary] - Combined Ten Datasets Statistics:")
    print(f"   🔸 Total sample count: {overall_stats.get('sample_count', 0)}")
    print(f"   🔸 Average audio length: {format_duration(overall_stats.get('mean_duration', 0))}")
    print(f"   🔸 Shortest audio length: {format_duration(overall_stats.get('min_duration', 0))}")
    print(f"   🔸 Longest audio length: {format_duration(overall_stats.get('max_duration', 0))}")
    print(f"   🔸 Median audio length: {format_duration(overall_stats.get('median_duration', 0))}")
    
    print(f"\n📊 Dataset Diversity Analysis:")
    if dataset_stats:
        mean_durations = [stats['mean_duration'] for stats in dataset_stats.values()]
        median_durations = [stats['median_duration'] for stats in dataset_stats.values()]
        
        print(f"   Dataset average length range: {format_duration(min(mean_durations))} - {format_duration(max(mean_durations))}")
        print(f"   Dataset median range: {format_duration(min(median_durations))} - {format_duration(max(median_durations))}")
        print(f"   Cross-dataset standard deviation: {np.std(mean_durations):.2f} seconds")
    
    print("="*100)

if __name__ == "__main__":
    main()