import os
import json
import glob
import pandas as pd

VOXCELEB1_ROOT_PATH = "/data/to/your/dataset/path/VoxCeleb"

CONCATENATED_AUDIO_SUBDIR = "wav"
CONCATENATED_AUDIO_BASE_DIR = os.path.join(VOXCELEB1_ROOT_PATH, "concatenated_audio", CONCATENATED_AUDIO_SUBDIR)

VOXCELEB_METADATA_CSV_PATH = "/data/to/your/dataset/path/VoxCeleb/vox2_meta.csv"

OUTPUT_JSON_PATH = os.path.join(VOXCELEB1_ROOT_PATH, "concatenated_audio", "gender_id_task_meta.json")

TASK_NAME = "Speaker_Gender_Identification_From_Concatenated_Audio"
DATASET_NAME = "VoxCeleb2_Concatenated"
QUESTION_TEXT = "What is the gender of the speaker in this audio segment?"
CHOICE_A = "Male"
CHOICE_B = "Female"

def load_speaker_gender_map(csv_path):
    gender_map = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"CSV first line: '{first_line}'")
        
        try:
            df = pd.read_csv(csv_path, engine='python', sep=None)
            print(f"Auto-detected column names: {df.columns.tolist()}")
        except:
            df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
            print(f"Column names read with tab separator: {df.columns.tolist()}")
        
        df.columns = [col.strip() for col in df.columns]
        print(f"Cleaned column names: {df.columns.tolist()}")
        
        vox_id_col = None
        gender_col = None
        
        for col in df.columns:
            if 'voxceleb' in col.lower() and 'id' in col.lower():
                vox_id_col = col
                print(f"Found ID column: '{vox_id_col}'")
            if 'gender' in col.lower():
                gender_col = col
                print(f"Found gender column: '{gender_col}'")
        
        if not vox_id_col or not gender_col:
            raise ValueError(f"Required columns not found. Need VoxCeleb ID column and Gender column.")
            
        for index, row in df.iterrows():
            if index < 5:
                print(f"Processing row {index}: ID={row[vox_id_col]}, Gender={row[gender_col]}")
                
            speaker_id = str(row[vox_id_col]).strip()
            gender_abbr = str(row[gender_col]).strip().lower()
            
            if gender_abbr == 'm':
                gender_map[speaker_id] = CHOICE_A
            elif gender_abbr == 'f':
                gender_map[speaker_id] = CHOICE_B
                
        print(f"Loaded gender information for {len(gender_map)} speakers from CSV")
        
    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        print(f"Please check CSV file format and ensure column names are correct.")
        return None
        
    return gender_map

VOXCELEB_ROOT_PATH = VOXCELEB1_ROOT_PATH

def create_gender_id_json():
    speaker_gender_map = load_speaker_gender_map(VOXCELEB_METADATA_CSV_PATH)
    if speaker_gender_map is None:
        print("Could not load speaker gender map. Aborting JSON generation.")
        return

    if not os.path.isdir(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: Concatenated audio base directory not found: {CONCATENATED_AUDIO_BASE_DIR}")
        print("Please ensure you have run the concatenation script and the path is correct.")
        return

    output_json_data = []
    unique_id_counter = 0

    print(f"Scanning for concatenated audio files in: {CONCATENATED_AUDIO_BASE_DIR}...")

    speaker_id_dirs = glob.glob(os.path.join(CONCATENATED_AUDIO_BASE_DIR, "id*"))

    for speaker_dir_path in speaker_id_dirs:
        if not os.path.isdir(speaker_dir_path):
            continue
        
        original_speaker_id = os.path.basename(speaker_dir_path)

        if original_speaker_id not in speaker_gender_map:
            continue
        
        speaker_gender = speaker_gender_map[original_speaker_id]

        concatenated_files = glob.glob(os.path.join(speaker_dir_path, "*.wav"))

        for audio_file_full_path in concatenated_files:
            relative_audio_path = os.path.relpath(audio_file_full_path, CONCATENATED_AUDIO_BASE_DIR)
            relative_audio_path = relative_audio_path.replace("\\", "/")

            entry = {
                "path": relative_audio_path,
                "question": QUESTION_TEXT,
                "choice_a": CHOICE_A,
                "choice_b": CHOICE_B,
                "answer_gt": speaker_gender,
                "task_name": TASK_NAME,
                "dataset_name": DATASET_NAME,
                "speaker_id_original": original_speaker_id,
                "uniq_id": unique_id_counter
            }
            output_json_data.append(entry)
            unique_id_counter += 1
            
    if not output_json_data:
        print("No concatenated audio files found or no gender information for corresponding speakers.")
        print("Please check:")
        print(f"  1. Concatenated audio directory: {CONCATENATED_AUDIO_BASE_DIR}")
        print(f"  2. Metadata CSV path: {VOXCELEB_METADATA_CSV_PATH}")
        print(f"  3. Speaker IDs match between audio folders and CSV.")
        return

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=4)
        print(f"\nSuccessfully generated JSON metadata with {len(output_json_data)} entries at: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    if VOXCELEB1_ROOT_PATH == "/path/to/your/VoxCeleb" or \
       VOXCELEB_METADATA_CSV_PATH == "/path/to/your/vox1_meta.csv":
        print("ERROR: Please update 'VOXCELEB1_ROOT_PATH' and 'VOXCELEB_METADATA_CSV_PATH' in the script!")
    elif not os.path.exists(VOXCELEB1_ROOT_PATH):
        print(f"ERROR: VOXCELEB1_ROOT_PATH '{VOXCELEB1_ROOT_PATH}' does not exist.")
    elif not os.path.exists(VOXCELEB_METADATA_CSV_PATH):
        print(f"ERROR: VOXCELEB_METADATA_CSV_PATH '{VOXCELEB_METADATA_CSV_PATH}' does not exist.")
    elif not os.path.exists(CONCATENATED_AUDIO_BASE_DIR):
        print(f"ERROR: CONCATENATED_AUDIO_BASE_DIR '{CONCATENATED_AUDIO_BASE_DIR}' does not exist. "
              "Ensure you have run the audio concatenation script first and the path is correct.")
    else:
        create_gender_id_json()