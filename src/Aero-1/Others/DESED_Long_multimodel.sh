export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default configuration
GPU_ID=0
PRUNE_LAYER_IDX=2
DATA_PATH="/data/to/your/desed/dataset"
SAMPLE_LIMIT=0
BATCH_MODE=false
MODEL_TYPE="qwen2.5"  # Default model is Qwen2.5-Omni, optional: "phi4", "aero1", "kimi", "qwen2.5"
CHUNK_SIZE=10         # Audio chunk size (seconds)

# Qwen2.5-Omni multi-GPU config
QWEN_MAIN_GPU=0       # Qwen2.5-Omni main model GPU (Default GPU 0)
QWEN_COMPONENT_GPU=1  # Qwen2.5-Omni component GPU (if needed)
QWEN_AUTO_GPU=true    # Automatically select the GPU with the largest memory as main GPU

# Kimi-Audio multi-GPU config
KIMI_MAIN_GPU=0       # Kimi-Audio main model GPU
KIMI_COMPONENT_GPU=1  # Kimi-Audio component GPU
KIMI_AUTO_GPU=true    # Automatically select the GPU with the largest memory

# Batch experiment config - modify as needed
METHODS=("random" "frame" "fast_v")
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Show help
show_help() {
    echo "Audio Event Detection Batch Evaluation Script - Supports Multiple Model Architectures"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu GPU_ID             Set GPU ID (default: 0)"
    echo "  --qwen-main-gpu GPU_ID       Set Qwen2.5-Omni main model GPU (default: 0)"
    echo "  --qwen-component-gpu GPU_ID  Set Qwen2.5-Omni component GPU (default: 1)"
    echo "  --kimi-main-gpu GPU_ID       Set Kimi-Audio main model GPU (default: 0)"
    echo "  --kimi-component-gpu GPU_ID  Set Kimi-Audio component GPU (default: 1)"
    echo "  --auto-gpu                   Automatically select the GPU with the largest memory as main GPU"
    echo "  -l, --layer LAYER_IDX        Set prune layer index (default: 2)"
    echo "  -d, --data PATH              Set data path"
    echo "  -s, --samples LIMIT          Set sample limit (default: 0, unlimited)"
    echo "  -m, --model MODEL_TYPE       Set model type (phi4|aero1|kimi|qwen2.5, default: qwen2.5)"
    echo "  -c, --chunk CHUNK_SIZE       Set audio chunk size (seconds) (default: 10)"
    echo "  -b, --batch                  Enable batch mode (run all method and ratio combinations)"
    echo "  --methods METHOD1,METHOD2    Specify method list (comma separated)"
    echo "  --ratios RATIO1,RATIO2       Specify ratio list (comma separated)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Batch mode examples:"
    echo "  $0 --batch                    # Run all default method and ratio combinations"
    echo "  $0 --batch --methods random,frame --ratios 0.2,0.5,0.8"
    echo "  $0 --batch --gpu 1 --samples 100"
    echo "  $0 --batch --model qwen2.5 --chunk 15   # Use Qwen2.5-Omni model, 15s audio chunk"
    echo "  $0 --batch --model qwen2.5 --qwen-main-gpu 1 --qwen-component-gpu 0"
    echo "  $0 --batch --model kimi --auto-gpu      # Use Kimi-Audio model, auto GPU assignment"
    echo ""
    echo "Single run examples:"
    echo "  $0 --gpu 0 --methods random --ratios 0.2"
    echo "  $0 --model qwen2.5 --methods attention --ratios 0.3"
    echo "  $0 --model kimi --kimi-main-gpu 2 --kimi-component-gpu 3"
    echo ""
    echo "Supported models:"
    echo "  phi4    - Phi-4-multimodal-instruct"
    echo "  aero1   - Aero-1-Audio-1.5B"
    echo "  kimi    - Kimi-Audio-7B-Instruct"
    echo "  qwen2.5 - Qwen2.5-Omni (default, multi-GPU supported)"
    echo ""
    echo "Multi-GPU config notes:"
    echo "  Main GPU: Loads main language model (high memory usage)"
    echo "  Component GPU: Loads audio processing component (relatively lower memory usage)"
    echo "  It is recommended to set the main GPU to the one with the largest memory"
    echo ""
    echo "Default methods: ${METHODS[*]}"
    echo "Default ratios: ${RATIOS[*]}"
}

# Parse command-line args
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --qwen-main-gpu)
            QWEN_MAIN_GPU="$2"
            shift 2
            ;;
        --qwen-component-gpu)
            QWEN_COMPONENT_GPU="$2"
            shift 2
            ;;
        --kimi-main-gpu)
            KIMI_MAIN_GPU="$2"
            shift 2
            ;;
        --kimi-component-gpu)
            KIMI_COMPONENT_GPU="$2"
            shift 2
            ;;
        --auto-gpu)
            QWEN_AUTO_GPU=true
            KIMI_AUTO_GPU=true
            shift
            ;;
        -l|--layer)
            PRUNE_LAYER_IDX="$2"
            shift 2
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -s|--samples)
            SAMPLE_LIMIT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -c|--chunk)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
            shift
            ;;
        --methods)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2
            ;;
        --ratios)
            IFS=',' read -ra RATIOS <<< "$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use $0 --help to see help information"
            exit 1
            ;;
    esac
done

# Validate model type
validate_model_type() {
    local valid_models=("phi4" "aero1" "kimi" "qwen2.5")
    if [[ ! " ${valid_models[*]} " =~ " ${MODEL_TYPE} " ]]; then
        echo "Error: Invalid model type '$MODEL_TYPE'"
        echo "Supported models: ${valid_models[*]}"
        exit 1
    fi
}

# For non-aero1 models, adjust chunk_size
if [[ ! " aero1 " =~ " ${MODEL_TYPE} " ]]; then
    CHUNK_SIZE=1000
fi

# Validate prune methods
validate_methods() {
    local valid_methods=("base" "random" "frame" "fast_v" "attention")
    for method in "${METHODS[@]}"; do
        if [[ ! " ${valid_methods[*]} " =~ " ${method} " ]]; then
            echo "Error: Invalid prune method '$method'"
            echo "Supported methods: ${valid_methods[*]}"
            exit 1
        fi
    done
}

# Validate prune ratios
validate_ratios() {
    for ratio in "${RATIOS[@]}"; do
        if ! [[ "$ratio" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$ratio < 0" | bc -l) )) || (( $(echo "$ratio > 1" | bc -l) )); then
            echo "Error: Invalid prune ratio '$ratio' (should be between 0-1)"
            exit 1
        fi
    done
}

# Validate audio chunk size
validate_chunk_size() {
    if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || (( CHUNK_SIZE < 1 )) || (( CHUNK_SIZE > 1200 )); then
        echo "Error: Invalid audio chunk size '$CHUNK_SIZE' (should be between 1-1200 seconds)"
        exit 1
    fi
}

# Check GPU info and auto-select largest memory GPU
check_and_setup_gpu() {
    echo "Checking GPU environment..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found, cannot check GPU info"
        return
    fi
    
    # Get GPU count
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $gpu_count GPUs:"
    
    # Show all GPU info
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while IFS=',' read -r index name total used free; do
        echo "  GPU $index: $name, Total Memory: ${total}MB, Used: ${used}MB, Free: ${free}MB"
    done
    
    # Auto GPU selection logic
    if [ "$QWEN_AUTO_GPU" = true ] && [ "$MODEL_TYPE" = "qwen2.5" ]; then
        echo ""
        echo "Auto-selecting the GPU with the largest memory as Qwen2.5-Omni main GPU..."
        
        local max_memory_gpu=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | cut -d',' -f1)
        QWEN_MAIN_GPU=$max_memory_gpu
        
        if [ "$gpu_count" -gt 1 ]; then
            local second_gpu=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | sort -t',' -k2 -nr | sed -n '2p' | cut -d',' -f1)
            if [ -n "$second_gpu" ] && [ "$second_gpu" != "$QWEN_MAIN_GPU" ]; then
                QWEN_COMPONENT_GPU=$second_gpu
            else
                for i in $(seq 0 $((gpu_count - 1))); do
                    if [ "$i" -ne "$QWEN_MAIN_GPU" ]; then
                        QWEN_COMPONENT_GPU=$i
                        break
                    fi
                done
            fi
        else
            QWEN_COMPONENT_GPU=$QWEN_MAIN_GPU
        fi
        
        echo "Qwen2.5-Omni auto selection result:"
        echo "  Main GPU: $QWEN_MAIN_GPU (largest memory)"
        echo "  Component GPU: $QWEN_COMPONENT_GPU"
    fi
    
    # Kimi-Audio auto GPU selection
    if [ "$KIMI_AUTO_GPU" = true ] && [ "$MODEL_TYPE" = "kimi" ]; then
        echo ""
        echo "Auto-selecting the GPU with the largest memory as Kimi-Audio main GPU..."
        
        local max_memory_gpu=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | cut -d',' -f1)
        KIMI_MAIN_GPU=$max_memory_gpu
        
        if [ "$gpu_count" -gt 1 ]; then
            local second_gpu=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | sort -t',' -k2 -nr | sed -n '2p' | cut -d',' -f1)
            if [ -n "$second_gpu" ] && [ "$second_gpu" != "$KIMI_MAIN_GPU" ]; then
                KIMI_COMPONENT_GPU=$second_gpu
            else
                for i in $(seq 0 $((gpu_count - 1))); do
                    if [ "$i" -ne "$KIMI_MAIN_GPU" ]; then
                        KIMI_COMPONENT_GPU=$i
                        break
                    fi
                done
            fi
        else
            KIMI_COMPONENT_GPU=$KIMI_MAIN_GPU
        fi
        
        echo "Kimi-Audio auto selection result:"
        echo "  Main GPU: $KIMI_MAIN_GPU (largest memory)"
        echo "  Component GPU: $KIMI_COMPONENT_GPU"
    fi
    
    # Validate GPU index
    if [ "$MODEL_TYPE" = "qwen2.5" ]; then
        if [ "$QWEN_MAIN_GPU" -ge "$gpu_count" ]; then
            echo "Warning: Qwen main GPU index $QWEN_MAIN_GPU out of range, using GPU 0"
            QWEN_MAIN_GPU=0
        fi
        if [ "$QWEN_COMPONENT_GPU" -ge "$gpu_count" ]; then
            echo "Warning: Qwen component GPU index $QWEN_COMPONENT_GPU out of range, using GPU 0"
            QWEN_COMPONENT_GPU=0
        fi
    elif [ "$MODEL_TYPE" = "kimi" ]; then
        if [ "$KIMI_MAIN_GPU" -ge "$gpu_count" ]; then
            echo "Warning: Kimi main GPU index $KIMI_MAIN_GPU out of range, using GPU 0"
            KIMI_MAIN_GPU=0
        fi
        if [ "$KIMI_COMPONENT_GPU" -ge "$gpu_count" ]; then
            echo "Warning: Kimi component GPU index $KIMI_COMPONENT_GPU out of range, using GPU 0"
            KIMI_COMPONENT_GPU=0
        fi
    fi
}

# Check data path
check_data_path() {
    if [ ! -d "$DATA_PATH" ]; then
        echo "Error: Data path does not exist: $DATA_PATH"
        exit 1
    fi
}

# Check Python environment
check_python_env() {
    echo "Checking Python environment and dependencies..."
    
    if ! command -v python &> /dev/null; then
        echo "Error: Python not found"
        exit 1
    fi

    python -c "
import sys
missing_packages = []

# Basic dependencies
basic_packages = ['torch', 'transformers', 'soundfile', 'numpy', 'tqdm', 'scipy', 'librosa']

# Check model type and required packages
model_type = '$MODEL_TYPE'
if model_type == 'qwen2.5':
    try:
        import qwen_omni_utils
        print('Qwen2.5-Omni inference package is installed')
    except ImportError:
        print('Warning: Qwen2.5-Omni inference package not installed, please refer to the official documentation')
        print('GitHub: https://github.com/QwenLM/Qwen2.5-Omni')
elif model_type == 'kimi':
    try:
        import kimia_infer
        print('Kimi-Audio inference package is installed')
    except ImportError:
        print('Warning: Kimi-Audio inference package not installed, please refer to the official documentation')
        print('GitHub: https://github.com/MoonshotAI/Kimi-Audio')

for package in basic_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Error: Missing required Python packages: {missing_packages}')
    print(f'Please install with: pip install {\" \".join(missing_packages)}')
    if model_type == 'qwen2.5':
        print('For Qwen2.5-Omni, also install: pip install -r requirements.txt')
        print('Reference: https://github.com/QwenLM/Qwen2.5-Omni')
    elif model_type == 'kimi':
        print('For Kimi-Audio, also install: pip install -r requirements.txt')
        print('Reference: https://github.com/MoonshotAI/Kimi-Audio')
    sys.exit(1)
else:
    print('Basic dependencies are installed')
"

    if [ $? -ne 0 ]; then
        exit 1
    fi
}

# Run a single experiment
run_single_experiment() {
    local method="$1"
    local ratio="$2"
    
    echo ""
    echo "=========================================="
    echo "Running experiment: method=$method, ratio=$ratio"
    echo "=========================================="
    
    export PRUNE_LAYER_IDX=$PRUNE_LAYER_IDX
    export PRUNE_RATIO=$ratio
    export PRUNE_METHOD=$method
    export SAMPLE_LIMIT=$SAMPLE_LIMIT
    export CHUNK_SIZE=$CHUNK_SIZE
    
    # Set special env for model type
    if [ "$MODEL_TYPE" = "qwen2.5" ]; then
        export QWEN_MAIN_GPU=$QWEN_MAIN_GPU
        export QWEN_COMPONENT_GPU=$QWEN_COMPONENT_GPU
        export CUDA_VISIBLE_DEVICES="${QWEN_MAIN_GPU},${QWEN_COMPONENT_GPU}"
        export QWEN_AUDIO_SAMPLE_RATE=16000
        export QWEN_AUDIO_CHUNK_SIZE=$CHUNK_SIZE
        
        echo "Qwen2.5-Omni config:"
        echo "  Main model GPU: $QWEN_MAIN_GPU"
        echo "  Component GPU: $QWEN_COMPONENT_GPU"
        echo "  CUDA visible devices: $CUDA_VISIBLE_DEVICES"
        echo "  Audio config: sample rate=${QWEN_AUDIO_SAMPLE_RATE}Hz, chunk size=${CHUNK_SIZE}s"
    elif [ "$MODEL_TYPE" = "kimi" ]; then
        export KIMI_MAIN_GPU=$KIMI_MAIN_GPU
        export KIMI_COMPONENT_GPU=$KIMI_COMPONENT_GPU
        export CUDA_VISIBLE_DEVICES="${KIMI_MAIN_GPU},${KIMI_COMPONENT_GPU}"
        export KIMI_AUDIO_SAMPLE_RATE=16000
        export KIMI_AUDIO_CHUNK_SIZE=$CHUNK_SIZE
        
        echo "Kimi-Audio config:"
        echo "  Main model GPU: $KIMI_MAIN_GPU"
        echo "  Component GPU: $KIMI_COMPONENT_GPU"
        echo "  CUDA visible devices: $CUDA_VISIBLE_DEVICES"
        echo "  Audio config: sample rate=${KIMI_AUDIO_SAMPLE_RATE}Hz, chunk size=${CHUNK_SIZE}s"
    else
        export CUDA_VISIBLE_DEVICES=$GPU_ID
    fi
    
    # Create result filename
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    if [ "$MODEL_TYPE" = "qwen2.5" ]; then
        local result_file="DESED_Results/desed_${MODEL_TYPE}_${method}_ratio${ratio}_chunk${CHUNK_SIZE}_mgpu${QWEN_MAIN_GPU}${QWEN_COMPONENT_GPU}_${timestamp}.log"
    elif [ "$MODEL_TYPE" = "kimi" ]; then
        local result_file="DESED_Results/desed_${MODEL_TYPE}_${method}_ratio${ratio}_chunk${CHUNK_SIZE}_mgpu${KIMI_MAIN_GPU}${KIMI_COMPONENT_GPU}_${timestamp}.log"
    else
        local result_file="DESED_Results/desed_${MODEL_TYPE}_${method}_ratio${ratio}_chunk${CHUNK_SIZE}_gpu${GPU_ID}_${timestamp}.log"
    fi
    
    echo "Result will be saved to: $result_file"
    echo "Start time: $(date)"
    
    # Select test script by model type
    local test_script=""
    case $MODEL_TYPE in
        "phi4")
            test_script="DESED_test_phi4.py"
            ;;
        "aero1")
            test_script="desed_aero1_test_sing_event.py"
            ;;
        "kimi")
            test_script="DESED_test_kimi.py"
            ;;
        "qwen2.5")
            test_script="DESED_test_qwen2.5.py"
            ;;
        *)
            echo "Error: Unknown model type '$MODEL_TYPE'"
            return 1
            ;;
    esac
    
    echo "Using test script: $test_script"
    echo "Model type: $MODEL_TYPE"
    echo "Audio chunk size: ${CHUNK_SIZE}s"
    
    # Check if test script exists
    if [ ! -f "$test_script" ]; then
        echo "Error: Test script does not exist: $test_script"
        echo "Make sure the following scripts exist:"
        echo "  - DESED_test_phi4.py"
        echo "  - DESED_test_aero.py"
        echo "  - DESED_test_kimi.py"
        echo "  - DESED_test_qwen2.5.py"
        return 1
    fi
    
    # Run experiment and save log
    python "$test_script" 2>&1 | tee "$result_file"
    local exit_code=${PIPESTATUS[0]}
    
    echo "End time: $(date)"
    
    if [ $exit_code -eq 0 ]; then
        echo "Experiment completed successfully!"
        echo "Log file: $result_file"
        
        # Try to show evaluation summary
        if [ -f "DESED_Results/latest_results.json" ]; then
            echo "Latest evaluation results:"
            cat "DESED_Results/latest_results.json" | head -20
        fi
    else
        echo "Experiment failed (exit code: $exit_code)"
        echo "Error log: $result_file"
        return $exit_code
    fi
    
    echo "=========================================="
    return $exit_code
}

# Batch experiment runner
run_batch_experiments() {
    local total_experiments=$((${#METHODS[@]} * ${#RATIOS[@]}))
    local current_exp=0
    local failed_experiments=()
    local successful_experiments=0
    
    echo ""
    echo "Starting batch experiments (multi-model support)"
    echo "Total experiments: $total_experiments"
    echo "Methods: ${METHODS[*]}"
    echo "Ratios: ${RATIOS[*]}"
    echo "Audio chunk size: ${CHUNK_SIZE}s"
    echo ""
    
    local start_time=$(date +%s)
    
    for method in "${METHODS[@]}"; do
        for ratio in "${RATIOS[@]}"; do
            ((current_exp++))
            echo "Progress: [$current_exp/$total_experiments]"
            
            if run_single_experiment "$method" "$ratio"; then
                ((successful_experiments++))
            else
                failed_experiments+=("${method}_${ratio}")
            fi
            
            # Pause between experiments to avoid resource conflicts
            if [ $current_exp -lt $total_experiments ]; then
                echo "Waiting 10 seconds before the next experiment..."
                sleep 10
            fi
        done
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo ""
    echo "Batch experiments completed!"
    echo "==============================="
    echo "Total experiments: $total_experiments"
    echo "Successful experiments: $successful_experiments"
    echo "Failed experiments: $((total_experiments - successful_experiments))"
    echo "Total duration: ${hours}h ${minutes}m ${seconds}s"
    echo "Model type: $MODEL_TYPE"
    echo "Audio chunk size: ${CHUNK_SIZE}s"
    
    if [ ${#failed_experiments[@]} -gt 0 ]; then
        echo ""
        echo "Failed experiments:"
        for failed in "${failed_experiments[@]}"; do
            echo "  - $failed"
        done
    fi
    
    echo ""
    echo "All result files are saved in: DESED_Results/"
    ls -la DESED_Results/ | tail -10
    
    # Generate experiment summary report
    echo ""
    echo "Generating experiment summary report..."
    local summary_file="DESED_Results/experiment_summary_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "=== Audio Event Detection Batch Experiment Summary ==="
        echo "Experiment time: $(date)"
        echo "Model type: $MODEL_TYPE"
        echo "Total experiments: $total_experiments"
        echo "Successful experiments: $successful_experiments"
        echo "Failed experiments: $((total_experiments - successful_experiments))"
        echo "Total duration: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        echo "Experiment config:"
        echo "  Methods: ${METHODS[*]}"
        echo "  Ratios: ${RATIOS[*]}"
        echo "  Prune layer: $PRUNE_LAYER_IDX"
        echo "  Audio chunk size: ${CHUNK_SIZE}s"
        echo "  Sample limit: $SAMPLE_LIMIT"
        echo ""
        if [ ${#failed_experiments[@]} -gt 0 ]; then
            echo "Failed experiments:"
            for failed in "${failed_experiments[@]}"; do
                echo "  - $failed"
            done
        fi
    } > "$summary_file"
    
    echo "Experiment summary saved to: $summary_file"
}

# Main execution flow
main() {
    # Validate inputs
    validate_model_type
    validate_methods
    validate_ratios
    validate_chunk_size
    check_data_path
    
    # Check and set GPU config
    check_and_setup_gpu
    
    check_python_env
    
    # Create result directory
    mkdir -p "DESED_Results"
    
    # Show config
    echo "============================"
    echo "Audio Event Detection Batch Evaluation Config (Multi-Model Enhanced Edition)"
    echo "============================"
    echo "Model type: $MODEL_TYPE"
    if [ "$MODEL_TYPE" = "qwen2.5" ]; then
        echo "Qwen main GPU: $QWEN_MAIN_GPU"
        echo "Qwen component GPU: $QWEN_COMPONENT_GPU"
        echo "Auto GPU selection: $QWEN_AUTO_GPU"
    elif [ "$MODEL_TYPE" = "kimi" ]; then
        echo "Kimi main GPU: $KIMI_MAIN_GPU"
        echo "Kimi component GPU: $KIMI_COMPONENT_GPU"
        echo "Auto GPU selection: $KIMI_AUTO_GPU"
    else
        echo "GPU ID: $GPU_ID"
    fi
    echo "Prune layer index: $PRUNE_LAYER_IDX"
    echo "Data path: $DATA_PATH"
    echo "Sample limit: $SAMPLE_LIMIT"
    echo "Audio chunk size: ${CHUNK_SIZE}s"
    echo "Batch mode: $BATCH_MODE"
    echo "Method list: ${METHODS[*]}"
    echo "Ratio list: ${RATIOS[*]}"
    echo "============================"
    
    # Show current GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "Current GPU usage config:"
        if [ "$MODEL_TYPE" = "qwen2.5" ]; then
            echo "  Main GPU ($QWEN_MAIN_GPU):"
            nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits --id=$QWEN_MAIN_GPU | while IFS=',' read -r name used total; do
                echo "    $name: ${used}MB / ${total}MB"
            done
            if [ "$QWEN_COMPONENT_GPU" -ne "$QWEN_MAIN_GPU" ]; then
                echo "  Component GPU ($QWEN_COMPONENT_GPU):"
                nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits --id=$QWEN_COMPONENT_GPU | while IFS=',' read -r name used total; do
                    echo "    $name: ${used}MB / ${total}MB"
                done
            fi
        elif [ "$MODEL_TYPE" = "kimi" ]; then
            echo "  Main GPU ($KIMI_MAIN_GPU):"
            nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits --id=$KIMI_MAIN_GPU | while IFS=',' read -r name used total; do
                echo "    $name: ${used}MB / ${total}MB"
            done
            if [ "$KIMI_COMPONENT_GPU" -ne "$KIMI_MAIN_GPU" ]; then
                echo "  Component GPU ($KIMI_COMPONENT_GPU):"
                nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits --id=$KIMI_COMPONENT_GPU | while IFS=',' read -r name used total; do
                    echo "    $name: ${used}MB / ${total}MB"
                done
            fi
        else
            echo "  GPU $GPU_ID:"
            nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits --id=$GPU_ID | while IFS=',' read -r name used total; do
                echo "    $name: ${used}MB / ${total}MB"
            done
        fi
        echo ""
    fi
    
    if [ "$BATCH_MODE" = true ]; then
        run_batch_experiments
    else
        # Single run mode
        if [ ${#METHODS[@]} -gt 0 ] && [ ${#RATIOS[@]} -gt 0 ]; then
            run_single_experiment "${METHODS[0]}" "${RATIOS[0]}"
        else
            echo "Error: No valid method and ratio specified"
            exit 1
        fi
    fi
}

# Signal handler - allow graceful exit
trap 'echo ""; echo "Received interrupt signal, exiting evaluation..."; exit 130' INT TERM

# Execute main function
main "$@"