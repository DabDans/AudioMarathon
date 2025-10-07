import os
import json
import glob
import random
from datetime import datetime




HAD_ROOT_PATH = "/path/to/your/HAD"


CONCATENATED_AUDIO_BASE_DIR = os.path.join(HAD_ROOT_PATH, "concatenated_audio")


OUTPUT_JSON_PATH = os.path.join(HAD_ROOT_PATH, "had_audio_classification_task.json")


TASK_NAME = "Audio_Authenticity_Classification"
DATASET_NAME = "HAD_Half_Truth_Audio_Detection"
QUESTION_TEXT = "Is this audio authentic or does it contain artificially synthesized (fake) segments?"
CHOICE_A = "Real"
CHOICE_B = "Fake"


RANDOM_SEED = 42

def scan_audio_files(base_dir):
    """
    Scan audio directory and collect information for all audio files
    
    Returns:
        list: List of dictionaries containing audio file information
    """
    audio_files = []
    

    real_dir = os.path.join(base_dir, "real")
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        for audio_file in real_files:
            relative_path = os.path.relpath(audio_file, base_dir).replace("\\", "/")
            audio_files.append({
                "path": relative_path,
                "category": "real",
                "full_path": audio_file
            })
        print(f"找到 {len(real_files)} 个真实音频文件")
    else:
        print(f"警告: 真实音频目录不存在: {real_dir}")
    

    fake_dir = os.path.join(base_dir, "fake")
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        for audio_file in fake_files:
            relative_path = os.path.relpath(audio_file, base_dir).replace("\\", "/")
            audio_files.append({
                "path": relative_path,
                "category": "fake",
                "full_path": audio_file
            })
        print(f"找到 {len(fake_files)} 个包含伪造部分的音频文件")
    else:
        print(f"警告: 伪造音频目录不存在: {fake_dir}")
    
    return audio_files

def get_audio_duration(audio_path):
    """
    Get audio file duration (optional feature, requires librosa or soundfile)
    """
    try:
        import soundfile as sf
        with sf.SoundFile(audio_path) as f:
            duration = len(f) / f.samplerate
            return round(duration, 2)
    except ImportError:

        return None
    except Exception as e:
        print(f"警告: 无法获取音频时长 {audio_path}: {e}")
        return None

def create_had_classification_json():
    """
    Create JSON metadata file for HAD audio classification task
    """

    if not os.path.exists(CONCATENATED_AUDIO_BASE_DIR):
        print(f"错误: 拼接音频目录不存在: {CONCATENATED_AUDIO_BASE_DIR}")
        print("请确保您已运行音频拼接脚本，且路径正确。")
        return
    

    print(f"扫描音频目录: {CONCATENATED_AUDIO_BASE_DIR}")
    audio_files = scan_audio_files(CONCATENATED_AUDIO_BASE_DIR)
    
    if not audio_files:
        print("未找到任何音频文件！")
        return
    

    audio_files.sort(key=lambda x: x["path"])
    print(f"总计找到 {len(audio_files)} 个音频文件")
    

    random.seed(RANDOM_SEED)
    random.shuffle(audio_files)
    print(f"使用随机种子 {RANDOM_SEED} 打乱音频文件顺序")
    

    output_json_data = []
    real_count = 0
    fake_count = 0
    
    for idx, audio_info in enumerate(audio_files):

        if audio_info["category"] == "real":
            answer = CHOICE_A  # "Real"
            real_count += 1
        else:
            answer = CHOICE_B  # "Fake"
            fake_count += 1
        

        duration = get_audio_duration(audio_info["full_path"])
        

        entry = {
            "path": audio_info["path"],
            "question": QUESTION_TEXT,
            "choice_a": CHOICE_A,
            "choice_b": CHOICE_B,
            "answer_gt": answer,
            "task_name": TASK_NAME,
            "dataset_name": DATASET_NAME,
            "category": audio_info["category"],
            "uniq_id": idx
        }
        

        if duration is not None:
            entry["duration_seconds"] = duration
        
        output_json_data.append(entry)
    

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    

    metadata = {
        "task_description": "Audio authenticity classification task for detecting half-truth audio with artificially synthesized segments",
        "creation_date": datetime.now().isoformat(),
        "total_samples": len(output_json_data),
        "real_samples": real_count,
        "fake_samples": fake_count,
        "real_ratio": round(real_count / len(output_json_data) * 100, 2),
        "fake_ratio": round(fake_count / len(output_json_data) * 100, 2),
        "random_seed": RANDOM_SEED,
        "base_directory": CONCATENATED_AUDIO_BASE_DIR
    }
    

    final_output = {
        "metadata": metadata,
        "samples": output_json_data
    }
    

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n成功生成HAD音频分类任务JSON文件: {OUTPUT_JSON_PATH}")
        print(f"总计样本数: {len(output_json_data)}")
        print(f"真实音频: {real_count} 个 ({real_count/len(output_json_data)*100:.1f}%)")
        print(f"伪造音频: {fake_count} 个 ({fake_count/len(output_json_data)*100:.1f}%)")
        print(f"随机种子: {RANDOM_SEED}")
        

        print(f"\n前3samples示例:")
        for i in range(min(3, len(output_json_data))):
            sample = output_json_data[i]
            print(f"  {i+1}. {sample['path']} -> {sample['answer_gt']} ({sample['category']})")
            
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")

def validate_json_file():
    """
    Validate the generated JSON file
    """
    if not os.path.exists(OUTPUT_JSON_PATH):
        print(f"JSON文件不存在: {OUTPUT_JSON_PATH}")
        return
    
    try:
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get("metadata", {})
        samples = data.get("samples", [])
        
        print(f"\n=== JSON文件验证 ===")
        print(f"文件路径: {OUTPUT_JSON_PATH}")
        print(f"元数据: {len(metadata)} 个字段")
        print(f"样本数量: {len(samples)}")
        
        if samples:

            required_fields = ["path", "question", "choice_a", "choice_b", "answer_gt", "uniq_id"]
            sample = samples[0]
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                print(f"警告: 样本缺少必要字段: {missing_fields}")
            else:
                print("✓ 所有必要字段都存在")
            

            real_answers = sum(1 for s in samples if s.get("answer_gt") == CHOICE_A)
            fake_answers = sum(1 for s in samples if s.get("answer_gt") == CHOICE_B)
            
            print(f"答案分布: Real={real_answers}, Fake={fake_answers}")
            
    except Exception as e:
        print(f"验证JSON文件时出错: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("HAD (Half-Truth Audio Detection) 分类任务元数据生成器")
    print("=" * 80)
    

    if not os.path.exists(HAD_ROOT_PATH):
        print(f"错误: HAD根目录不存在: {HAD_ROOT_PATH}")
        exit(1)
    
    if not os.path.exists(CONCATENATED_AUDIO_BASE_DIR):
        print(f"错误: 拼接音频目录不存在: {CONCATENATED_AUDIO_BASE_DIR}")
        print("请确保您已运行音频拼接脚本。")
        exit(1)
    

    create_had_classification_json()
    

    validate_json_file()
    
    print("\n" + "=" * 80)
    print("HAD音频分类任务JSON生成完成！")
    print("=" * 80)