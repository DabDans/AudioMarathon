import os
import csv
from pydub import AudioSegment
from collections import defaultdict
import random



CSV_FILE_PATH = "/path/to/your/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv"


AUDIO_ROOT_PATH = "/path/to/your/TAU-urban-acoustic-scenes-2020-mobile-development"


OUTPUT_DIR_BASE = os.path.join("/path/to/your/TAU-urban-acoustic-scenes-2020-mobile-development", "concatenated_audio")


MIN_DURATION_MINUTES = 1.5
MAX_DURATION_MINUTES = 3.5
MIN_DURATION_MS = MIN_DURATION_MINUTES * 60 * 1000
MAX_DURATION_MS = MAX_DURATION_MINUTES * 60 * 1000


SHUFFLE_FILES = False


def get_audio_duration_ms(audio_segment):
    """Get duration of AudioSegment in milliseconds"""
    return len(audio_segment)

def ensure_dir(directory_path):
    """Ensure directory exists, create if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Output directory confirmed/created: {directory_path}")


def process_audio_from_csv():
    """
    Read audio file information from CSV file and concatenate audio by scene_label and identifier.
    """
    if not os.path.isfile(CSV_FILE_PATH):
        print(f"Error: CSV file not found: {CSV_FILE_PATH}")
        return


    audio_groups = defaultdict(list)
    
    print(f"Loading audio data from CSV file: {CSV_FILE_PATH}")
    

    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)
        
        for row in reader:
            if len(row) >= 4:
                filename = row[0]
                scene_label = row[1]
                identifier = row[2]
                source_label = row[3]
                

                audio_path = os.path.join(AUDIO_ROOT_PATH, filename)
                

                group_key = f"{scene_label}_{identifier}"
                audio_groups[group_key].append(audio_path)

    total_files = sum(len(files) for files in audio_groups.values())
    print(f"\n共收集到 {total_files} 个音频文件，分为 {len(audio_groups)} 个分组。")
    if not audio_groups:
        print("没有找到任何可供处理的音频文件。")
        return

    ensure_dir(OUTPUT_DIR_BASE)

    print("\n开始拼接音频...")
    for group_key, audio_files in audio_groups.items():
        print(f"\n处理分组: {group_key} (共 {len(audio_files)} 个文件)")
        

        scene_label = group_key.split('_')[0]
        

        scene_output_dir = os.path.join(OUTPUT_DIR_BASE, scene_label)
        ensure_dir(scene_output_dir)
        
        if SHUFFLE_FILES:
            random.shuffle(audio_files)
        else:
            audio_files.sort()

        current_segment_audio = AudioSegment.empty()
        segment_count = 1
        
        for audio_path in audio_files:
            try:
                if not os.path.isfile(audio_path):
                    print(f"  警告: 音频文件不存在: {audio_path}。跳过此文件。")
                    continue
                audio = AudioSegment.from_file(audio_path)
            except Exception as e:
                print(f"  警告: 无法加载音频文件 {audio_path}。错误: {e}。跳过此文件。")
                continue

            if get_audio_duration_ms(current_segment_audio) == 0 or \
               (get_audio_duration_ms(current_segment_audio) + get_audio_duration_ms(audio) <= MAX_DURATION_MS):
                current_segment_audio += audio
            else:
                if get_audio_duration_ms(current_segment_audio) >= MIN_DURATION_MS:
                    output_filename = f"{group_key}_segment_{segment_count}.wav"
                    output_path = os.path.join(scene_output_dir, output_filename)
                    try:
                        print(f"  保存片段: {output_path} (时长: {get_audio_duration_ms(current_segment_audio)/1000.0:.2f}s)")
                        current_segment_audio.export(output_path, format="wav")
                        segment_count += 1
                    except Exception as e:
                        print(f"  错误: 保存文件 {output_path} 失败: {e}")
                current_segment_audio = audio 


        if get_audio_duration_ms(current_segment_audio) > 0:
            if get_audio_duration_ms(current_segment_audio) >= MIN_DURATION_MS:
                output_filename = f"{group_key}_segment_{segment_count}.wav"
                output_path = os.path.join(scene_output_dir, output_filename)
                try:
                    print(f"  保存最后片段: {output_path} (时长: {get_audio_duration_ms(current_segment_audio)/1000.0:.2f}s)")
                    current_segment_audio.export(output_path, format="wav")
                except Exception as e:
                    print(f"  错误: 保存文件 {output_path} 失败: {e}")
            else:
                 print(f"  注意: 分组 {group_key} 的最后一个片段时长 "
                      f"{get_audio_duration_ms(current_segment_audio)/1000.0:.2f}s, "
                      f"少于指定的最小时长 {MIN_DURATION_MINUTES} 分钟。此片段未保存。")

    print("\n音频拼接处理完成。")


if __name__ == "__main__":
    if CSV_FILE_PATH == "d:/我的文档/meta.csv" and not os.path.exists(CSV_FILE_PATH):
        print("错误：请在脚本中正确配置 'CSV_FILE_PATH' 变量！")
    elif not os.path.exists(CSV_FILE_PATH):
        print(f"错误: CSV_FILE_PATH ('{CSV_FILE_PATH}') 不存在，请检查路径！")
    elif AUDIO_ROOT_PATH == "/root/autodl-tmp/project" and not os.path.exists(AUDIO_ROOT_PATH):
        print("错误：请在脚本中正确配置 'AUDIO_ROOT_PATH' 变量！")
    elif not os.path.exists(AUDIO_ROOT_PATH):
        print(f"错误: AUDIO_ROOT_PATH ('{AUDIO_ROOT_PATH}') 不存在，请检查路径！")
    else:


        process_audio_from_csv()