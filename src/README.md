# src Directory Description

This directory contains scripts for multi-dataset multimodal (primarily audio) evaluation and method comparison, based on different inference/sparse strategies of the **Qwen2.5-Omni** model, along with several auxiliary analysis tools and result files.

## Directory Overview

```
README.md                  This documentation file
sanitize_comments.py       Executed batch comment/path sanitization script (retained for reuse)
LibriSpeech_analysis.py    (if exists) Summary or analysis script for LibriSpeech results

analyse_audio_duration/    Scripts for analyzing total audio duration or related statistics of each dataset
LibriSpeech_Results/       Evaluation output JSON for LibriSpeech (including sparse/different pruning ratios)
Qwen_2.5/                  Core code for Qwen2.5-Omni model adaptation, processing, and inference
Qwen_Methods/              Scripts for evaluating various datasets using "original/basic" methods (with optional pruning)
Qwen_Dart/                 Evaluation scripts using DART sparse attention mechanism
Segement/                  Processing scripts for specific tasks (such as segmentation/subtask splitting/Vox gender classification, etc.)
```

> Note: `.bak` files with the same name are backups created during the first modification by the sanitization script, which can be deleted or kept for comparison.

## Functions of Each Subdirectory and File

### 1. `Qwen_2.5/`
Contains:
- `modeling_qwen2_5_omni*.py` Multiple modes (original/low VRAM/DART/fast variant) of model wrappers or customizations.
- `processing_qwen2_5_omni.py` / `audio_process.py` / `vision_process.py` Multimodal input preprocessing.
- `configuration_qwen2_5_omni.py` Model configuration class.

These files provide a unified interface for upper-level evaluation scripts (loading, processing audio tokens/visual tokens/text tokens, etc.).

### 2. `Qwen_Methods/`
Evaluation scripts written for several datasets (such as GTZAN, HAD, LibriSpeech, RACE, SLUE, TAU, VESUS, Vox, etc.), with file naming pattern: `<DATASET>_qwen2.5.py`.

Common features:
- Read dataset audio/text descriptions.
- Construct multi-turn or single-turn multimodal prompts.
- Call Qwen2.5-Omni for generation or classification, parse output and calculate accuracy, F1, and other metrics.
- Support optional pruning parameters (via environment variables) to control audio token/layer pruning strategies.

### 3. `Qwen_Dart/`
Similar to the above, but integrates **DART sparse attention mechanism**. Scripts expose a set of command-line parameters (such as `--sparse`, `--pruned_layer`, `--reduction_ratio`, `--pivot_audio_token`, etc.) to control sparse configuration and retention ratio. Used to explore reducing computation/memory consumption while maintaining performance.

### 4. `Segement/`
Segmentation and subtask scripts for specific tasks, for example:
- `HAD_segment.py` may segment long audio from a dataset into blocks before feeding into the model.
- `Vox_task.py`, `Vox_total_gender.py`, `Vox2_task.py` Preprocessing or decomposition logic for VoxCeleb or age/gender classification tasks.
- `TAU_task.py` Processing for TAU soundscape/event datasets.

### 5. `analyse_audio_duration/`
Quick statistics on total audio duration, sample count, mean, variance, etc., of multiple datasets (specific to each file implementation), helping with:
- Estimating inference costs
- Designing batch/pruning strategies
- Comparing distribution characteristics of different datasets

### 6. `LibriSpeech_Results/`
Stores `*_results_*.json` and `*_timing_stats_*.json`:
- `results` files: Per-sample or aggregated prediction outputs, model response parsing, and accuracy/F1, etc.
- `timing_stats` files: Inference time, prefill time, token count statistics (usually averaged over the first N samples).

These file names often include `sparse_ratio_x.xxx` to distinguish sparse/pruning settings.

To run again if needed:
```
python sanitize_comments.py
```

## Dependencies (Recommended)

pip install -r requirements.txt

## Usage Examples

### Basic Method Scripts (Example: GTZAN)
```
python Qwen_Methods/GTZAN_qwen2.5.py \
	--model-path <MODEL_DIR>
```
Optional pruning control via environment variables:
```
set PRUNE_LAYER_IDX=2
set PRUNE_RATIO=0.5
set PRUNE_METHOD=frame   (options: base / frame / random)
set SAMPLE_LIMIT=100      (test only first 100 samples)
```
> Windows PowerShell can use `$env:PRUNE_RATIO="0.5"` format; Linux/macOS use `export`.

### DART Sparse Version (Example: GTZAN)
```
python Qwen_Dart/GTZAN_qwen_dart.py \
	--model-path <MODEL_DIR> \
	--sparse true \
	--pruned_layer 2 \
	--reduction_ratio 0.7 \
	--pivot_audio_token 4
```
Common parameters:
- `--sparse`: Whether to enable DART
- `--reduction_ratio`: Retention ratio (lower values are more sparse)
- `--audio_token_start_index / --audio_token_length`: Audio token start and length (depends on data preprocessing)
- `--sample_limit`: Limit sample count to speed up debugging

### Dataset Root Path Setting
If placeholders or relative paths appear in scripts, please change variables (e.g., `data_path_root`) to your local dataset directory:
```
data_path_root = "<DATASET_ROOT>/GTZAN/concatenated_audio"
```
Ensure directory structure is consistent with script reading methods (print/debug as necessary).

### Results and Metrics
Scripts typically output:
- One sample/aggregated JSON per line (`*.jsonl` or `*_results_*.json`)
- Timing statistics (`*_timing_stats_*.json`)
External analysis scripts can be written for cross-comparison.

## Frequently Asked Questions (FAQ)
1. Model not found: Confirm `--model-path` points to the downloaded Qwen2.5-Omni model directory.
2. CUDA out of memory:
	 - Use low VRAM version scripts (related paths to `modeling_qwen2_5_omni_low_VRAM_mode.py`)
	 - Reduce batch size / use pruning or DART to reduce tokens
3. Accuracy degradation: Gradually increase `reduction_ratio` or disable pruning to trace back to baseline.
4. Empty results: Check if dataset path is correct and audio files are readable (try testing with `librosa.load` separately).