import os
import json
import glob
import pandas as pd

VOXCELEB_ROOT_PATH = "/data/to/your/dataset/path/VoxCeleb"

CONCATENATED_AUDIO_SUBDIR = "wav"
CONCATENATED_AUDIO_BASE_DIR = os.path.join(VOXCELEB_ROOT_PATH, "concatenated_audio", CONCATENATED_AUDIO_SUBDIR)

VOXCELEB1_METADATA_CSV_PATH = "/data/to/your/dataset/path/VoxCeleb/vox1_meta.csv"
VOXCELEB2_METADATA_CSV_PATH = "/data/to/your/dataset/path/VoxCeleb/vox2_meta.csv"

OUTPUT_JSON_PATH = os.path.join(VOXCELEB_ROOT_PATH, "concatenated_audio", "gender_id_task_meta.json")

TASK_NAME = "Speaker_Gender_Identification_From_Concatenated_Audio"
DATASET_NAME = "VoxCeleb_Combined"
QUESTION_TEXT = "What is the gender of the speaker in this audio segment?"
CHOICE_A = "Male"
CHOICE_B = "Female"

def load_speaker_gender_map_from_csv(csv_path, version=2):
    """
    Load speaker gender information from CSV file, compatible with VoxCeleb1 and VoxCeleb2 formats.
    
    Args:
        csv_path: CSV file path
        version: 1 for VoxCeleb1 format, 2 for VoxCeleb2 format
    
    Returns:
        Dictionary mapping speaker ID to gender
    """
    gender_map = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"CSV file ({csv_path}) first line: '{first_line}'")
        
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
            if version == 1 and ('voxceleb1' in col.lower() or 'vox1' in col.lower()) and 'id' in col.lower():
                vox_id_col = col
                print(f"Found VoxCeleb1 ID column: '{vox_id_col}'")
            elif version == 2 and ('voxceleb2' in col.lower() or 'vox2' in col.lower()) and 'id' in col.lower():
                vox_id_col = col
                print(f"Found VoxCeleb2 ID column: '{vox_id_col}'")
            
            if 'gender' in col.lower():
                gender_col = col
                print(f"Found gender column: '{gender_col}'")
        
        if not vox_id_col:
            for col in df.columns:
                if ('id' in col.lower() and 'vggface' not in col.lower() and 
                    'vgg' not in col.lower() and 'uniq' not in col.lower()):
                    vox_id_col = col
                    print(f"Found generic ID column: '{vox_id_col}'")
                    break
        
        if not vox_id_col or not gender_col:
            print(f"Warning: Required columns not found in {csv_path}. Need VoxCeleb ID column and Gender column.")
            print(f"Available columns: {df.columns.tolist()}")
            return {}
        
        processed_count = 0
        for index, row in df.iterrows():
            if index < 5:
                print(f"Processing row {index}: ID={row[vox_id_col]}, Gender={row[gender_col]}")
                
            speaker_id = str(row[vox_id_col]).strip()
            gender_abbr = str(row[gender_col]).strip().lower()
            
            if gender_abbr == 'm':
                gender_map[speaker_id] = CHOICE_A
                processed_count += 1
            elif gender_abbr == 'f':
                gender_map[speaker_id] = CHOICE_B
                processed_count += 1
                
        print(f"Loaded gender information for {len(gender_map)} speakers from {csv_path} (processed {processed_count} rows)")
        
    except FileNotFoundError:
        print(f"Warning: Metadata CSV file not found {csv_path}")
    except Exception as e:
        print(f"Error reading or parsing CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        print(f"Please check CSV file format and ensure column names are correct.")
    
    return gender_map

def load_combined_speaker_gender_map():
    """
    Load and combine speaker gender information from VoxCeleb1 and VoxCeleb2.
    If there are duplicate IDs, use VoxCeleb2 information as priority.
    """
    combined_gender_map = {}
    
    print("\n=== Loading VoxCeleb1 metadata ===")
    vox1_map = load_speaker_gender_map_from_csv(VOXCELEB1_METADATA_CSV_PATH, version=1)
    if vox1_map:
        male_count = sum(1 for gender in vox1_map.values() if gender == CHOICE_A)
        female_count = sum(1 for gender in vox1_map.values() if gender == CHOICE_B)
        print(f"VoxCeleb1 gender statistics: Male={male_count}, Female={female_count}, Total={len(vox1_map)}")
        combined_gender_map.update(vox1_map)
    
    print("\n=== Loading VoxCeleb2 metadata ===")
    vox2_map = load_speaker_gender_map_from_csv(VOXCELEB2_METADATA_CSV_PATH, version=2)
    if vox2_map:
        male_count = sum(1 for gender in vox2_map.values() if gender == CHOICE_A)
        female_count = sum(1 for gender in vox2_map.values() if gender == CHOICE_B)
        print(f"VoxCeleb2 gender statistics: Male={male_count}, Female={female_count}, Total={len(vox2_map)}")
        
        overlap_ids = set(vox1_map.keys()) & set(vox2_map.keys())
        if overlap_ids:
            print(f"Found {len(overlap_ids)} overlapping IDs between VoxCeleb1 and VoxCeleb2. Will use VoxCeleb2 gender information.")
            
            conflicts = sum(1 for speaker_id in overlap_ids if vox1_map[speaker_id] != vox2_map[speaker_id])
            if conflicts:
                print(f"Warning: Found {conflicts} overlapping IDs with inconsistent gender labels. Will use VoxCeleb2 labels.")
        
        combined_gender_map.update(vox2_map)
    
    if combined_gender_map:
        male_count = sum(1 for gender in combined_gender_map.values() if gender == CHOICE_A)
        female_count = sum(1 for gender in combined_gender_map.values() if gender == CHOICE_B)
        print(f"\nCombined gender statistics: Male={male_count}, Female={female_count}, Total={len(combined_gender_map)}")
    else:
        print("\nWarning: Failed to load gender information from any metadata file.")
    
    return combined_gender_map

def create_gender_id_json():
    """
    Scan concatenated audio files and generate JSON metadata.
    """
    speaker_gender_map = load_combined_speaker_gender_map()
    if not speaker_gender_map:
        print("Unable to load speaker gender mapping. Terminating JSON generation.")
        return

    if not os.path.isdir(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: Concatenated audio base directory not found: {CONCATENATED_AUDIO_BASE_DIR}")
        print("Please ensure you have run the concatenation script and the path is correct.")
        return

    output_json_data = []
    unique_id_counter = 0
    missing_gender_count = 0

    print(f"\nScanning for concatenated audio files in directory: {CONCATENATED_AUDIO_BASE_DIR}...")
    speaker_id_dirs = glob.glob(os.path.join(CONCATENATED_AUDIO_BASE_DIR, "id*"))
    print(f"Found {len(speaker_id_dirs)} potential speaker directories")

    for speaker_dir_path in speaker_id_dirs:
        if not os.path.isdir(speaker_dir_path):
            continue
        
        original_speaker_id = os.path.basename(speaker_dir_path)

        if original_speaker_id not in speaker_gender_map:
            missing_gender_count += 1
            if missing_gender_count <= 5:
                print(f"Warning: Gender information not found for speaker {original_speaker_id}. Skipping their audio files.")
            elif missing_gender_count == 6:
                print("More speakers with missing gender information exist (not showing all warnings).")
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
    
    if missing_gender_count > 0:
        print(f"\nWarning: {missing_gender_count} speakers have no gender information in metadata and were skipped.")
            
    if not output_json_data:
        print("\nNo concatenated audio files found or corresponding speakers have no gender information.")
        print("Please check:")
        print(f"  1. Concatenated audio directory: {CONCATENATED_AUDIO_BASE_DIR}")
        print(f"  2. VoxCeleb1 metadata CSV path: {VOXCELEB1_METADATA_CSV_PATH}")
        print(f"  3. VoxCeleb2 metadata CSV path: {VOXCELEB2_METADATA_CSV_PATH}")
        print(f"  4. Audio folder matching with speaker IDs in CSV.")
        return

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=4)
        
        male_count = sum(1 for entry in output_json_data if entry["answer_gt"] == CHOICE_A)
        female_count = sum(1 for entry in output_json_data if entry["answer_gt"] == CHOICE_B)
        
        print(f"\nSuccessfully generated JSON metadata with {len(output_json_data)} entries: {OUTPUT_JSON_PATH}")
        print(f"Gender distribution: {male_count} male audio, {female_count} female audio")
        print(f"Male-female ratio: {male_count/(male_count+female_count)*100:.1f}% male, {female_count/(male_count+female_count)*100:.1f}% female")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("VoxCeleb Combined Dataset Gender Identification Task Metadata Generator")
    print("=" * 80)
    
    path_errors = False
    
    if not os.path.exists(VOXCELEB_ROOT_PATH):
        print(f"Error: VOXCELEB_ROOT_PATH '{VOXCELEB_ROOT_PATH}' does not exist.")
        path_errors = True
    
    if not os.path.exists(VOXCELEB1_METADATA_CSV_PATH):
        print(f"Warning: VOXCELEB1_METADATA_CSV_PATH '{VOXCELEB1_METADATA_CSV_PATH}' does not exist.")
    
    if not os.path.exists(VOXCELEB2_METADATA_CSV_PATH):
        print(f"Warning: VOXCELEB2_METADATA_CSV_PATH '{VOXCELEB2_METADATA_CSV_PATH}' does not exist.")
    
    if not os.path.exists(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: CONCATENATED_AUDIO_BASE_DIR '{CONCATENATED_AUDIO_BASE_DIR}' does not exist. "
              "Please ensure you have run the audio concatenation script and the path is correct.")
        path_errors = True
    
    if not os.path.exists(VOXCELEB1_METADATA_CSV_PATH) and not os.path.exists(VOXCELEB2_METADATA_CSV_PATH):
        print("Error: Both metadata files do not exist. At least one metadata file is required.")
        path_errors = True
    
    if not path_errors:
        create_gender_id_json()
    else:
        print("\nScript terminated due to path errors. Please check the above errors and correct the path configuration.")