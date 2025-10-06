import os
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import random

def read_label_file(label_path):
    audio_info = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                print(f"Warning: Invalid format: {line.strip()}")
                continue
                
            audio_name = parts[0]
            segments_info = parts[1].split('/')
            has_fake = parts[2] == '0' if len(parts) > 2 else False
            
            segments = []
            for segment in segments_info:
                try:
                    start, end, authenticity = segment.split('-')
                    segments.append({
                        'start': float(start),
                        'end': float(end),
                        'is_real': authenticity == 'T'
                    })
                except ValueError:
                    print(f"Warning: Invalid segment format: {segment}")
                    continue
            
            segments.sort(key=lambda x: x['start'])
            audio_info[audio_name] = {
                'segments': segments,
                'has_fake': has_fake
            }
    
    return audio_info

def load_segment(segments_dir, audio_name, start, end, sample_rate):
    segment_path = os.path.join(segments_dir, f"{audio_name}.wav")
    
    if not os.path.exists(segment_path):
        print(f"Warning: Audio file not found {segment_path}")
        return None, 0
    
    try:
        audio_full, sr = sf.read(segment_path)
        
        if sr != sample_rate:
            print(f"Warning: Sample rate mismatch {segment_path}. Expected {sample_rate}, actual {sr}")
        
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        if end_sample > len(audio_full):
            print(f"Warning: Segment end point exceeds audio length, adjusting {audio_name} {end_sample} > {len(audio_full)}")
            end_sample = len(audio_full)
        
        audio_segment = audio_full[start_sample:end_sample]
        segment_duration = len(audio_segment) / sr
        
        return audio_segment, segment_duration
    except Exception as e:
        print(f"Error: Error processing file {segment_path}: {str(e)}")
        return None, 0

def concatenate_audio_segments(segments_dir, label_path, output_dir, sample_rate=16000, 
                              min_duration=180, max_duration=300, start_index=131):
    real_output_dir = os.path.join(output_dir, "real")
    fake_output_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)
    
    audio_info = read_label_file(label_path)
    print(f"Loaded {len(audio_info)} audio entries from label file")
    
    fake_count = sum(1 for info in audio_info.values() if info['has_fake'])
    real_count = len(audio_info) - fake_count
    print(f"Completely real audio: {real_count}, audio containing fake parts: {fake_count}")
    
    real_audio_data = []
    fake_audio_data = []
    
    for audio_name, info in tqdm(audio_info.items(), desc="Collecting audio segments"):
        segments = info['segments']
        has_fake = info['has_fake']
        
        for segment in segments:
            start = segment['start']
            end = segment['end']
            
            audio_data, duration = load_segment(segments_dir, audio_name, start, end, sample_rate)
            
            if audio_data is not None and len(audio_data) > 0:
                segment_info = {
                    'audio_data': audio_data,
                    'duration': duration,
                    'audio_name': audio_name,
                    'is_real': segment['is_real']
                }
                
                if has_fake:
                    fake_audio_data.append(segment_info)
                else:
                    real_audio_data.append(segment_info)
    
    print(f"Collected {len(real_audio_data)} real audio segments and {len(fake_audio_data)} audio segments containing fake parts")
    
    random.shuffle(real_audio_data)
    random.shuffle(fake_audio_data)
    
    fake_index = start_index
    real_index = start_index
    
    fake_index = generate_merged_files(fake_audio_data, fake_output_dir, "fake", 
                                    fake_index, sample_rate, min_duration, max_duration)
    
    real_index = generate_merged_files(real_audio_data, real_output_dir, "real", 
                                     real_index, sample_rate, min_duration, max_duration)
    
    print(f"Processing complete. Generated {fake_index - start_index} audio files containing fake parts and {real_index - start_index} real audio files")
    
    return fake_index, real_index

def generate_merged_files(audio_data, output_dir, type_prefix, start_index, sample_rate, 
                         min_duration, max_duration):
    if not audio_data:
        print(f"Warning: No {type_prefix} audio data available")
        return start_index
    
    min_samples = int(min_duration * sample_rate)
    max_samples = int(max_duration * sample_rate)
    
    current_index = start_index
    
    while len(audio_data) > 0:
        batch_audio = []
        batch_sources = []
        total_samples = 0
        used_indices = []
        
        for i, segment in enumerate(audio_data):
            if i in used_indices:
                continue
                
            audio_segment = segment['audio_data']
            segment_samples = len(audio_segment)
            
            if total_samples + segment_samples > max_samples and total_samples >= min_samples:
                break
            
            batch_audio.append(audio_segment)
            batch_sources.append(segment['audio_name'])
            total_samples += segment_samples
            used_indices.append(i)
            
            if total_samples >= max_samples:
                break
        
        if total_samples < min_samples and len(used_indices) == len(audio_data):
            audio_copy = np.concatenate(batch_audio)
            while len(audio_copy) < min_samples:
                if len(audio_copy) == 0:
                    break
                needed = min_samples - len(audio_copy)
                to_add = min(needed, len(audio_copy))
                audio_copy = np.concatenate([audio_copy, audio_copy[:to_add]])
            
            if type_prefix == "fake":
                output_filename = f"HAD_train_fake_{current_index}.wav"
            else:
                output_filename = f"HAD_train_real_{current_index}.wav"
                
            output_path = os.path.join(output_dir, output_filename)
            sf.write(output_path, audio_copy, sample_rate)
            duration = len(audio_copy) / sample_rate
            print(f"Created {output_path}, duration: {duration:.2f}s, merged {len(batch_sources)} segments, padded with repetition")
            
            audio_data = [seg for i, seg in enumerate(audio_data) if i not in used_indices]
            current_index += 1
        
        elif total_samples >= min_samples:
            final_audio = np.concatenate(batch_audio)
            
            if len(final_audio) > max_samples:
                final_audio = final_audio[:max_samples]
            
            if type_prefix == "fake":
                output_filename = f"HAD_train_fake_{current_index}.wav"
            else:
                output_filename = f"HAD_train_real_{current_index}.wav"
                
            output_path = os.path.join(output_dir, output_filename)
            sf.write(output_path, final_audio, sample_rate)
            duration = len(final_audio) / sample_rate
            print(f"Created {output_path}, duration: {duration:.2f}s, merged {len(batch_sources)} segments")
            
            audio_data = [seg for i, seg in enumerate(audio_data) if i not in used_indices]
            current_index += 1
        
        else:
            if batch_audio:
                final_audio = np.concatenate(batch_audio)
                while len(final_audio) < min_samples:
                    needed = min_samples - len(final_audio)
                    to_add = min(needed, len(final_audio))
                    final_audio = np.concatenate([final_audio, final_audio[:to_add]])
                
                if type_prefix == "fake":
                    output_filename = f"HAD_train_fake_{current_index}.wav"
                else:
                    output_filename = f"HAD_train_real_{current_index}.wav"
                    
                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, final_audio, sample_rate)
                duration = len(final_audio) / sample_rate
                print(f"Created {output_path}, duration: {duration:.2f}s, merged all remaining segments and padded with repetition")
                
                current_index += 1
            
            break
    
    return current_index

if __name__ == "__main__":
    segments_dir = "/data/to/your/dataset/path/HAD/HAD_train/train"
    label_path = "/data/to/your/dataset/path/HAD/HAD_train/HAD_train_label.txt"
    output_dir = "/data/to/your/dataset/path/HAD/concatenated_audio"
    
    fake_end, real_end = concatenate_audio_segments(
        segments_dir, label_path, output_dir,
        min_duration=180,
        max_duration=300,
        start_index=131
    )