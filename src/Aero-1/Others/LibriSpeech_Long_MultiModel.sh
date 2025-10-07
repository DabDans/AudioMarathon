export CUDA_LAUNCH_BLOCKING=1

# Default parameters
GPU_ID=0
PRUNE_LAYER_IDX=2
SAMPLE_LIMIT=0
BATCH_MODE=false
MODEL_TYPE="qwen2.5"  # Default is Qwen2.5-Omni model, options: "phi4", "aero1", "kimi", "qwen2.5"
CHUNK_SIZE=30
LIBRISPEECH_PATH="/data/to/your/librispeech-long"
# Qwen2.5-Omni multi-GPU configuration
QWEN_MAIN_GPU=0       # Qwen2.5-Omni main model GPU (default GPU 0)
QWEN_COMPONENT_GPU=1  # Qwen2.5-Omni component GPU (if needed)
QWEN_AUTO_GPU=true    # Automatically select GPU with maximum memory as main GPU

# Batch experiment configuration - can be modified as needed
METHODS=("random" "frame" "fast_v")
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Show usage help
show_help() {
    echo "LibriSpeech ASR Batch Evaluation Script - Supports Multi-Model Evaluation"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu GPU_ID             Set GPU ID (default: 0)"
    echo "  --qwen-main-gpu GPU_ID       Set Qwen2.5-Omni main model GPU (default: 0)"
    echo "  --qwen-component-gpu GPU_ID  Set Qwen2.5-Omni component GPU (default: 1)"
    echo "  --auto-gpu                   Automatically select max memory GPU as Qwen main GPU"
    echo "  -l, --layer LAYER_IDX        Set pruning layer index (default: 2)"
    echo "  -d, --data PATH              Set data path"
    echo "  -s, --samples LIMIT          Set sample limit (default: 0, unlimited)"
    echo "  -m, --model MODEL_TYPE       Set model type (phi4|aero1|kimi|qwen2.5, default: qwen2.5)"
    echo "  -c, --chunk CHUNK_SIZE       Set audio chunk size (seconds) (default: 30)"
    echo "  -b, --batch                  Enable batch mode (run all method and ratio combinations)"
    echo "  --methods METHOD1,METHOD2    Specify method list (comma separated)"
    echo "  --ratios RATIO1,RATIO2       Specify ratio list (comma separated)"
    echo "  -h, --help                   Show this help information"
    echo ""
    echo "Batch mode example:"
    echo "  $0 --batch --methods random,frame --ratios 0.2,0.5,0.8"
    echo "  $0 --batch --model qwen2.5 --chunk 45   # Use Qwen2.5-Omni model, 45s audio chunk"
    echo "  $0 --batch --model qwen2.5 --auto-gpu   # Automatically select max memory GPU"
    echo ""
    echo "Single run example:"
    echo "  $0 --gpu 0 --methods random --ratios 0.2"
    echo "  $0 --model qwen2.5 --methods attention --ratios 0.3  # Use Qwen2.5-Omni model"
    echo "  $0 --model qwen2.5 --qwen-main-gpu 2 --qwen-component-gpu 3  # Multi-GPU configuration"
    echo ""
}

# Parse command line arguments
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
        --auto-gpu)
            QWEN_AUTO_GPU=true
            shift
            ;;
        -l|--layer)
            PRUNE_LAYER_IDX="$2"
            shift 2
            ;;
        -d|--data)
            LIBRISPEECH_PATH="$2"
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

# Check data path
check_data_path() {
    if [[ ! -d "$LIBRISPEECH_PATH" ]]; then
        echo "Error: .sh file check data path does not exist: $LIBRISPEECH_PATH"
        exit 1
    fi
}

# Run a single experiment
run_single_experiment() {
    local method="$1"
    local ratio="$2"
    
    echo ""
    echo "=========================================="
    echo "Running experiment: Method=$method, Ratio=$ratio"
    echo "=========================================="
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export PRUNE_LAYER_IDX=$PRUNE_LAYER_IDX
    export PRUNE_RATIO=$ratio
    export PRUNE_METHOD=$method
    export SAMPLE_LIMIT=$SAMPLE_LIMIT
    export CHUNK_SIZE=$CHUNK_SIZE
    export LIBRISPEECH_PATH="$LIBRISPEECH_PATH"
    
    # Select test script based on model type
    local test_script=""
    case $MODEL_TYPE in
        "phi4")
            test_script="librispeech_test_phi4.py"
            ;;
        "aero1")
            test_script="librispeech_test_aero.py"
            ;;
        "kimi")
            test_script="librispeech_test_kimi.py"
            ;;
        "qwen2.5")
            test_script="librispeech_test_qwen2.5.py"
            ;;
        *)
            echo "Error: Unknown model type '$MODEL_TYPE'"
            return 1
            ;;
    esac
    
    echo "Test script: $test_script"
    python "$test_script"
}

# Main process
main() {
    validate_model_type
    check_data_path
    
    if [[ "$BATCH_MODE" == true ]]; then
        for method in "${METHODS[@]}"; do
            for ratio in "${RATIOS[@]}"; do
                run_single_experiment "$method" "$ratio"
            done
        done
    else
        run_single_experiment "${METHODS[0]}" "${RATIOS[0]}"
    fi
}

main "$@"