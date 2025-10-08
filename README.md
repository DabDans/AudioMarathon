# AudioMarathon: Long-Form Audio Understanding Benchmark



## 📖 Overview

**AudioMarathon** is a comprehensive benchmark designed to evaluate Audio Large Language Models (Audio-LLMs) on long-form audio understanding tasks. This repository contains the evaluation code and tools for testing various state-of-the-art audio-language models across multiple challenging tasks.

### Key Features

- 🎯 **Multi-Task Evaluation**: Supports 10+ diverse audio understanding tasks
- 🔊 **Long-Form Audio**: Handles extended audio sequences up to several minutes
- 🧠 **Multiple Models**: Evaluation scripts for Phi-4-MM, Qwen2.5-Omni, and Aero-1.
- ⚡ **Audio Token Pruning**: Built-in support for various KV-cache compression methods
- 📊 **Comprehensive Metrics**: Detailed performance analysis with timing statistics

## 🎪 Supported Tasks

AudioMarathon evaluates models across the following task categories with 6,563 samples:

### Speech Content Extraction (1,514 samples - 23.07%)

| Task | Dataset | Samples | Description |
|------|---------|---------|-------------|
| **Automatic Speech Recognition (ASR)** | LibriSpeech | 204 (3.10%) | Transcribe and understand spoken content |
| **Speech Content Reasoning (SCR)** | RACE | 820 (12.49%) | Answer questions based on read-aloud passages |
| **Speech Entity Recognition (SER)** | SLUE | 490 (7.46%) | Recognize and extract entities from spoken language |

### Audio Classification (1,519 samples - 23.14%)

| Task | Dataset | Samples | Description |
|------|---------|---------|-------------|
| **Audio Scene Classifier (ASC)** | TAU | 1,145 (17.44%) | Classify acoustic scenes (indoor/outdoor environments) |
| **Music Classifier (MC)** | GTZAN | 120 (1.83%) | Classify music genres from audio clips |
| **Sound Event Detection (SED)** | DESED | 254 (3.87%) | Detect and classify sound events in domestic environments |

### Speaker Recognition (3,530 samples - 53.79%)

| Task | Dataset | Samples | Description |
|------|---------|---------|-------------|
| **Emotion Recognition (ER)** | VESUS | 185 (2.82%) | Recognize emotions from speech |
| **Speech Detection (SD)** | HAD | 776 (11.82%) | Distinguish between real and AI-generated speech |
| **Speaker Age Recognition (SAR)** | VoxCeleb | 959 (14.60%) | Classify speaker age groups from voice |
| **Speaker Gender Recognition (SGR)** | VoxCeleb | 1,614 (24.58%) | Classify speaker gender from voice |

## 🏗️ Repository Structure

```
AudioMarathon/
├── Phi4MM/
│   ├── DART/                    # DART (Dynamic Audio Reduction Technique) implementations
│   ├── Others/                  # Standard evaluation scripts for Phi-4-MM
│   │   ├── DESED_test.py       # Sound event detection evaluation
│   │   ├── gtzan_test.py       # Music genre classification
│   │   ├── HAD_test.py         # Audio deepfake detection
│   │   ├── race_test.py        # Reading comprehension
│   │   ├── SLUE_test.py        # Spoken language understanding
│   │   ├── TAU_test.py         # Acoustic scene classification
│   │   ├── VESUS_test.py       # Emotion recognition
│   │   ├── Vox_age_test.py     # Age classification
│   │   └── Vox_test.py         # Gender classification
│   └── phi4_kvpress/            # KV-cache compression methods
│
├── Qwen_2.5_Omni/
│   ├── Dart/                    # DART implementations for Qwen
│   ├── Others/                  # Standard evaluation scripts for Qwen2.5-Omni
│   └── qwen_kvpress/            # KV-cache compression methods
│
├── Voxtral/
│   ├── eval_DESED.py           # Voxtral evaluation scripts
│   ├── eval_GTZAN.py
│   ├── eval_HAD.py
│   ├── eval_LibriSpeech.py
│   ├── eval_RACE.py
│   ├── eval_SLUE.py
│   ├── eval_TAU.py
│   ├── eval_VESUS.py
│   ├── eval_Vox_Age.py
│   └── eval_Vox.py
│
├── Aero-1/                      # Aero-1 model evaluation scripts
│   ├── DART/
│   └── Others/
│
├── kvpress/                     # KV-cache compression implementations
│   ├── presses/                 # Various compression strategies
│   ├── attention_patch.py
│   ├── audio_features.py
│   └── pipeline.py
│
├── Segment/                     # Audio segmentation tools
│   ├── GTZAN_task.py
│   ├── HAD_segment.py
│   ├── TAU_task.py
│   └── Vox2_task.py
│
└── analyse_audio_duration/      # Audio duration analysis utilities
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/AudioMarathon.git
cd AudioMarathon
```

2. **Install dependencies**

Choose the appropriate requirements file based on the model you want to evaluate:

```bash
# For Phi-4-MM
pip install -r Phi4_requirements.txt

# For Qwen2.5-Omni
pip install -r Qwen_requirements.txt

# For Aero-1
pip install -r Aero1_requirements.txt
```

**Note**: Each model has its own environment requirements. We recommend using separate virtual environments for different models to avoid dependency conflicts.

3. **Download models**
```bash
# For Phi-4-MM
huggingface-cli download microsoft/Phi-4-multimodal-instruct

# For Qwen2.5-Omni
huggingface-cli download Qwen/Qwen2.5-Omni-7B

# For Aero-1
huggingface-cli download Aero-1/Aero-1-7B
```


### Basic Usage

#### Evaluate Phi-4-MM on GTZAN (Music Genre Classification)

```bash
cd Phi4MM/Others

# Basic evaluation
export CUDA_VISIBLE_DEVICES=0
export PRUNE_RATIO=0
export PRUNE_METHOD=base
export SAMPLE_LIMIT=100
export RESULTS_DIR=./GTZAN_Results

python gtzan_test.py
```

#### Evaluate with Audio Token Pruning

```bash
# Using FastV pruning with 50% compression
export CUDA_VISIBLE_DEVICES=0
export PRUNE_LAYER_IDX=2
export PRUNE_RATIO=0.5
export PRUNE_METHOD=fast_v
export RESULTS_DIR=./GTZAN_Results_FastV50

python gtzan_test.py
```

#### Batch Evaluation

For evaluating multiple sparsity ratios in a single run, use the batch testing scripts:

```bash
cd Qwen_2.5_Omni/Dart

# Basic usage - test HAD task with default ratios (0.1-0.9)
bash batch_test.sh HAD

# Test with custom GPU
bash batch_test.sh --gpu-id 1 TAU

# Test with sample limit (useful for quick testing)
bash batch_test.sh --sample-limit 100 SLUE

# Test with specific sparsity ratios
bash batch_test.sh --ratios 0.0,0.3,0.5,0.7 VESUS

# Test with custom pruned layers and output directory
bash batch_test.sh --pruned-layer 3 --output-dir ./my_results GTZAN

# Comprehensive test with all options
bash batch_test.sh \
  --gpu-id 0 \
  --sample-limit 200 \
  --pruned-layer 2 \
  --ratios 0.0,0.2,0.4,0.6,0.8 \
  --output-dir ./Qwen_DART_Results \
  race
```

**Available Options:**
- `-g, --gpu-id <id>`: Specify GPU device (default: 0)
- `-s, --sample-limit <num>`: Limit number of samples (default: 0 for no limit)
- `-l, --pruned-layer <num>`: Number of layers to prune (default: 2)
- `-r, --ratios <ratios>`: Comma-separated sparsity ratios (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
- `-o, --output-dir <dir>`: Results output directory (default: ./Qwen_DART_Results)
- `-h, --help`: Show help message

**Supported Tasks:**
HAD, race, SLUE, TAU, VESUS, Vox, Vox_age, LibriSpeech, DESED, GTZAN

The batch script will:
1. Automatically run tests for all specified sparsity ratios
2. Generate logs for each test in `<output-dir>/logs/`
3. Create a summary report with all results
4. Display timing statistics and accuracy metrics

## ⚙️ Configuration

### Environment Variables

All evaluation scripts support the following environment variables:

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` | Any valid GPU ID |
| `PRUNE_LAYER_IDX` | Layer index for audio pruning | `2` | Integer >= 0 |
| `PRUNE_RATIO` | Ratio of audio tokens to prune | `0` | 0.0 - 1.0 |
| `PRUNE_METHOD` | Pruning method to use | `base` | `base`, `fast_v`, `random`, `frame` |
| `SAMPLE_LIMIT` | Limit number of samples | `0` (no limit) | Integer >= 0 |
| `RESULTS_DIR` | Output directory for results | Task-specific | Any valid path |

### Audio Token Pruning Methods

AudioMarathon supports multiple KV-cache compression strategies:

1. **base**: No pruning (baseline)
2. **Fast_v**: FastV attention-based pruning
3. **Random**: Random token pruning
4. **Frame**: Frame-based structured pruning
5. **DART**: Pruning tokens based on its duplication with other tokens


## 📈 Performance Analysis

### Timing Analysis

Each script tracks detailed timing metrics:
- **Prefill Time**: Time for initial audio encoding
- **Decode Time**: Time for generating responses
- **Tokens per Second**: Generation throughput
- **Audio Duration**: Input audio length


## 🛠️ Utility Tools

### Audio Duration Analysis

Analyze audio lengths in your datasets:

```bash
cd analyse_audio_duration
python GTZAN.py
python HAD.py
python TAU.py
```

### Audio Segmentation

Segment long audio files for processing:

```bash
cd Segment
python GTZAN_task.py
python HAD_segment.py
python TAU_task.py
```

## 📝 Data Preparation

### Expected Data Format

Each task expects data in a specific format:

#### GTZAN (JSON metadata)
```json
[
  {
    "path": "audio/blues_001.wav",
    "question": "What is the genre of this music?",
    "choice_a": "Blues",
    "choice_b": "Classical",
    "choice_c": "Rock",
    "choice_d": "Jazz",
    "answer_gt": "A"
  }
]
```

#### HAD (Directory structure)
```
HAD/
├── real/
│   ├── audio_001.wav
│   └── audio_002.wav
└── fake/
    ├── audio_001.wav
    └── audio_002.wav
```

#### DESED (JSON with tasks)
```json
{
  "tasks": [
    {
      "path": "audio_001.wav",
      "task_type": "detection",
      "question": "What sound events are present?",
      "choices": {
        "A": "Dog barking",
        "B": "Car horn",
        "C": "Phone ringing",
        "D": "Door slamming"
      },
      "answer_gt": "C"
    }
  ]
}
```



## 📄 Citation

If you use AudioMarathon in your research, please cite:


## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: 1006803978@qq.com

## 🙏 Acknowledgments

We thank the following for their contributions:
- Microsoft for Phi-4-Multimodal
- Qwen team for Qwen2.5-Omni
- Fixie AI for Ultravox/Voxtral
- All dataset providers (DESED, GTZAN, HAD, LibriSpeech, RACE, SLUE, TAU, VESUS, VoxCeleb)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This benchmark is designed for research purposes. Please ensure you have the proper licenses and permissions for all datasets before use.
