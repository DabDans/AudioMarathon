import os
import json
import glob
import random




GTZAN_ROOT_PATH = "/path/to/your/GTZAN"


CONCATENATED_AUDIO_BASE_DIR = os.path.join(GTZAN_ROOT_PATH, "concatenated_audio")


OUTPUT_JSON_PATH = os.path.join(GTZAN_ROOT_PATH, "concatenated_audio", "music_genre_classification_meta.json")


TASK_NAME = "Music_Genre_Classification"
DATASET_NAME = "GTZAN_Music_Genre"
QUESTION_TEXT = "What music genre is represented in this audio segment?"


GENRE_OPTIONS = [
    "Blues - blues",
    "Classical - classical",
    "Country - country",
    "Disco - disco",
    "Hip-hop - hiphop",
    "Jazz - jazz",
    "Metal - metal",
    "Pop - pop",
    "Reggae - reggae",
    "Rock - rock"
]


GENRE_MAPPING = {
    "blues": "Blues - blues",
    "classical": "Classical - classical",
    "country": "Country - country",
    "disco": "Disco - disco",
    "hiphop": "Hip-hop - hiphop",
    "jazz": "Jazz - jazz",
    "metal": "Metal - metal",
    "pop": "Pop - pop",
    "reggae": "Reggae - reggae",
    "rock": "Rock - rock"
}


def extract_genre_from_path(audio_path):
    """
    Extract music genre label from audio file path.
    Assumes format like .../concatenated_audio/genre_label/audio.wav
    or genre label contained in filename
    
    Args:
        audio_path: Complete path to the audio file
    Returns:
        Extracted genre label, or None if extraction fails
    """
    try:

        parts = audio_path.split(os.sep)
        for part in parts:
            if part.lower() in GENRE_MAPPING:
                return part.lower()
            

        filename = os.path.basename(audio_path)
        for genre in GENRE_MAPPING.keys():
            if genre in filename.lower():
                return genre
                

        parent_dir = os.path.basename(os.path.dirname(audio_path))
        if parent_dir.lower() in GENRE_MAPPING:
            return parent_dir.lower()
            
        print(f"Warning: Unable to extract genre label from path: {audio_path}")
        return None
    except Exception as e:
        print(f"Error extracting genre label: {e}")
        return None

def create_music_genre_json():
    """
    Scan processed audio files and generate JSON metadata.
    """
    if not os.path.isdir(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: Processed audio directory not found: {CONCATENATED_AUDIO_BASE_DIR}")
        return

    output_json_data = []
    unique_id_counter = 0
    genre_counts = {}

    print(f"\nScanning audio files in directory: {CONCATENATED_AUDIO_BASE_DIR}...")
    

    audio_files = glob.glob(os.path.join(CONCATENATED_AUDIO_BASE_DIR, "**", "*.wav"), recursive=True)
    print(f"Found {len(audio_files)} audio files")


    random.seed(42)

    for audio_file_full_path in audio_files:

        genre_label = extract_genre_from_path(audio_file_full_path)
        
        if not genre_label or genre_label not in GENRE_MAPPING:
            print(f"Skipping file with unrecognized genre: {audio_file_full_path}")
            continue
            

        genre_counts[genre_label] = genre_counts.get(genre_label, 0) + 1
        

        correct_option = GENRE_MAPPING[genre_label]
        

        other_options = [opt for opt in GENRE_OPTIONS if opt != correct_option]
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
            "genre_label": genre_label,
            "uniq_id": unique_id_counter
        }
        
        output_json_data.append(entry)
        unique_id_counter += 1
    
    if not output_json_data:
        print("\nNo valid audio files found or unable to extract genre labels.")
        return


    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=4)
        
        print(f"\nSuccessfully generated JSON metadata with {len(output_json_data)} entries: {OUTPUT_JSON_PATH}")
        

        print("\nGenre distribution:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            friendly_name = GENRE_MAPPING[genre]
            percentage = count/len(output_json_data)*100
            print(f"  {friendly_name}: {count} files ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error writing JSON file: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("GTZAN Music Genre Classification Task Metadata Generator")
    print("=" * 80)
    

    if not os.path.exists(GTZAN_ROOT_PATH):
        print(f"Error: GTZAN_ROOT_PATH '{GTZAN_ROOT_PATH}' does not exist.")
    elif not os.path.exists(CONCATENATED_AUDIO_BASE_DIR):
        print(f"Error: CONCATENATED_AUDIO_BASE_DIR '{CONCATENATED_AUDIO_BASE_DIR}' does not exist.")
    else:
        create_music_genre_json()