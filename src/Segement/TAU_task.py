import os
import json
import glob
import random

TAU_ROOT_PATH = "/data/to/your/dataset/path/TAU"

CONCATENATED_AUDIO_BASE_DIR = os.path.join(TAU_ROOT_PATH, "concatenated_audio")

OUTPUT_JSON_PATH = os.path.join(TAU_ROOT_PATH, "concatenated_audio", "acoustic_scene_task_meta.json")

TASK_NAME = "Acoustic_Scene_Classification"
DATASET_NAME = "TAU_Urban_Acoustic_Scenes"
QUESTION_TEXT = "What type of acoustic scene is represented in this audio segment?"

SCENE_OPTIONS = [
    "Airport - airport",
    "Indoor shopping mall - shopping_mall", 
    "Metro station - metro_station",
    "Pedestrian street - street_pedestrian",
    "Public square - public_square",
    "Street with medium level of traffic - street_traffic",
    "Traveling by a tram - tram",
    "Traveling by a bus - bus",
    "Traveling by an underground metro - metro",
    "Urban park - park"
]

SCENE_MAPPING = {
    "airport": "Airport - airport",
    "shopping_mall": "Indoor shopping mall - shopping_mall",
    "metro_station": "Metro station - metro_station",
    "street_pedestrian": "Pedestrian street - street_pedestrian",
    "public_square": "Public square - public_square",
    "street_traffic": "Street with medium level of traffic - street_traffic",
    "tram": "Traveling by a tram - tram",
    "bus": "Traveling by a bus - bus",
    "metro": "Traveling by an underground metro - metro",
    "park": "Urban park - park"
}

def extract_scene_from_path(audio_path):
    try:
        parts = audio_path.split(os.sep)
        for part in parts:
            if part.lower() in SCENE_MAPPING:
                return part.lower()
            
        filename = os.path.basename(audio_path)
        for scene in SCENE_MAPPING.keys():
            if scene in filename.lower():
                return scene
                
        parent_dir = os.path.basename(os.path.dirname(audio_path))
        if parent_dir.lower() in SCENE_MAPPING:
            return parent_dir.lower()
            
        print(f"Warning: Unable to extract scene label from path: {audio_path}")
        return None
    except Exception as e:
        print(f"Error extracting scene label: {e}")
        return None

def create_acoustic_scene_json():
    if not os.path.isdir(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: Processed audio directory not found: {CONCATENATED_AUDIO_BASE_DIR}")
        return

    output_json_data = []
    unique_id_counter = 0
    scene_counts = {}

    print(f"\nScanning audio files in directory: {CONCATENATED_AUDIO_BASE_DIR}...")
    
    audio_files = glob.glob(os.path.join(CONCATENATED_AUDIO_BASE_DIR, "**", "*.wav"), recursive=True)
    print(f"Found {len(audio_files)} audio files")

    random.seed(42)

    for audio_file_full_path in audio_files:
        scene_label = extract_scene_from_path(audio_file_full_path)
        
        if not scene_label or scene_label not in SCENE_MAPPING:
            print(f"Skipping file with unrecognized scene: {audio_file_full_path}")
            continue
            
        scene_counts[scene_label] = scene_counts.get(scene_label, 0) + 1
        
        correct_option = SCENE_MAPPING[scene_label]
        
        other_options = [opt for opt in SCENE_OPTIONS if opt != correct_option]
        random_options = random.sample(other_options, 3)
        
        all_options = [correct_option] + random_options
        random.shuffle(all_options)
        
        correct_index = all_options.index(correct_option)
        correct_letter = chr(65 + correct_index)
        
        relative_audio_path = os.path.relpath(audio_file_full_path, CONCATENATED_AUDIO_BASE_DIR)
        relative_audio_path = relative_audio_path.replace("\\", "/")

        entry = {
            "path": relative_audio_path,
            "question": QUESTION_TEXT,
            "choice_a": all_options[0],
            "choice_b": all_options[1],
            "choice_c": all_options[2],
            "choice_d": all_options[3],
            "answer_gt": correct_letter,
            "task_name": TASK_NAME,
            "dataset_name": DATASET_NAME,
            "scene_label": scene_label,
            "uniq_id": unique_id_counter
        }
        
        output_json_data.append(entry)
        unique_id_counter += 1
    
    if not output_json_data:
        print("\nNo valid audio files found or unable to extract scene labels.")
        return

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=4)
        
        print(f"\nSuccessfully generated JSON metadata with {len(output_json_data)} entries: {OUTPUT_JSON_PATH}")
        
        print("\nScene distribution:")
        for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
            friendly_name = SCENE_MAPPING[scene]
            percentage = count/len(output_json_data)*100
            print(f"  {friendly_name}: {count} files ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("TAU Urban Acoustic Scene Task Metadata Generator")
    print("=" * 80)
    
    if not os.path.exists(TAU_ROOT_PATH):
        print(f"Error: TAU_ROOT_PATH '{TAU_ROOT_PATH}' does not exist.")
    elif not os.path.exists(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: CONCATENATED_AUDIO_BASE_DIR '{CONCATENATED_AUDIO_BASE_DIR}' does not exist.")
    else:
        create_acoustic_scene_json()