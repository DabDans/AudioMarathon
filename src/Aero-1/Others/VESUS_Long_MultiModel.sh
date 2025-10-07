export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default parameters
GPU_ID=0
PRUNE_LAYER_IDX=2
SAMPLE_LIMIT=0
BATCH_MODE=false
MODEL_TYPE="qwen2.5"  # Default is Qwen2.5-Omni model, options: "phi4", "aero1", "kimi", "qwen2.5"
METHODS=("base" "random" "fast_v" "frame")
RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5)
DATA_PATH="/data/to/your/vesus/data"
RESULTS_BASE_DIR="/data/to/your/results/dir"

# Show help information
show_help() {
    echo "VESUS emotion recognition batch evaluation script - supports multi-model evaluation"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu GPU_ID             Set GPU ID (default: 0)"
    echo "  -l, --layer LAYER_IDX        Set pruning layer index (default: 2)"
    echo "  -s, --samples LIMIT          Set sample limit (default: 0, unlimited)"
    echo "  -m, --model MODEL_TYPE       Set model type (phi4|aero1|kimi|qwen2.5, default: qwen2.5)"
    echo "  -b, --batch                  Enable batch mode (run all method and ratio combinations)"
    echo "  --methods METHOD1,METHOD2    Specify method list (comma separated)"
    echo "  --ratios RATIO1,RATIO2       Specify ratio list (comma separated)"
    echo "  --data-path PATH             Specify VESUS dataset path"
    echo "  --results-dir DIR            Specify result save directory (default: /data/to/your/results/dir)"
    echo "  -h, --help                   Show this help information"
    echo ""
    echo "Batch mode example:"
    echo "  $0 --batch --methods random,frame --ratios 0.2,0.5,0.8"
    echo "  $0 --batch --model kimi --samples 100"
    echo ""
    echo "Single run example:"
    echo "  $0 --gpu 0 --methods random --ratios 0.2"
    echo "  $0 --model qwen2.5 --methods fast_v --ratios 0.3  # Use Qwen2.5-Omni model"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -l|--layer)
            PRUNE_LAYER_IDX="$2"
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
        --methods)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2
            ;;
        --ratios)
            IFS=',' read -ra RATIOS <<< "$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use $0 --help to view help information"
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
    if [[ ! -d "$DATA_PATH" ]]; then
        echo "Error: Data path does not exist: $DATA_PATH"
        exit 1
    fi

    local json_file="$DATA_PATH/audio_emotion_dataset.json"
    if [[ ! -f "$json_file" ]]; then
        echo "Error: Emotion dataset file does not exist: $json_file"
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
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export PRUNE_LAYER_IDX=$PRUNE_LAYER_IDX
    export PRUNE_RATIO=$ratio
    export PRUNE_METHOD=$method
    export SAMPLE_LIMIT=$SAMPLE_LIMIT
    export VESUS_DATA_PATH=$DATA_PATH
    
    # Create result file name
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local result_file="$RESULTS_BASE_DIR/vesus_${method}_ratio${ratio}_gpu${GPU_ID}_${timestamp}.log"
    
    echo "Result will be saved to: $result_file"
    echo "Start time: $(date)"
    
    # Select test script according to model type
    local test_script=""
    case $MODEL_TYPE in
        "phi4")
            test_script="VESUS_test_phi4.py"
            ;;
        "aero1")
            test_script="VESUS_test_aero.py"
            ;;
        "kimi")
            test_script="VESUS_test_kimi.py"
            ;;
        "qwen2.5")
            test_script="VESUS_test_qwen2.5.py"
            ;;
        *)
            echo "Error: Unknown model type '$MODEL_TYPE'"
            return 1
            ;;
    esac
    
    # Run experiment and save log
    python "$test_script" 2>&1 | tee "$result_file"
    local exit_code=${PIPESTATUS[0]}
    
    echo "End time: $(date)"
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Experiment completed successfully!"
        echo "Log file: $result_file"
    else
        echo "❌ Experiment failed (exit code: $exit_code)"
        echo "Error log: $result_file"
        return $exit_code
    fi
    
    echo "=========================================="
    return $exit_code
}

# Batch run experiments
run_batch_experiments() {
    local total_experiments=$((${#METHODS[@]} * ${#RATIOS[@]}))
    local current_exp=0
    
    echo ""
    echo "🚀 Starting batch experiments"
    echo "Total experiments: $total_experiments"
    echo "Methods: ${METHODS[*]}"
    echo "Ratios: ${RATIOS[*]}"
    echo ""
    
    for method in "${METHODS[@]}"; do
        for ratio in "${RATIOS[@]}"; do
            ((current_exp++))
            echo "Progress: [$current_exp/$total_experiments]"
            run_single_experiment "$method" "$ratio"
        done
    done
    
    echo ""
    echo "🎉 Batch experiments completed!"
    echo "All result files saved in: $RESULTS_BASE_DIR/"
    ls -la "$RESULTS_BASE_DIR/" | tail -10
}

# Main execution flow
main() {
    validate_model_type
    check_data_path
    
    mkdir -p "$RESULTS_BASE_DIR"
    
    if [[ "$BATCH_MODE" == true ]]; then
        run_batch_experiments
    else
        run_single_experiment "${METHODS[0]}" "${RATIOS[0]}"
    fi
}

main "$@"