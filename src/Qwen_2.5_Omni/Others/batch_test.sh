#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="$PYTHONPATH:/data/to/your/Qwen_2.5_Code/path/"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

DEFAULT_GPU_ID=0
DEFAULT_SAMPLE_LIMIT=0
DEFAULT_RESULTS_DIR="./Qwen25_Token_Prune_Results"

SUPPORTED_TASKS=("HAD" "race" "SLUE" "TAU" "VESUS" "Vox" "Vox_age" "LibriSpeech" "DESED" "GTZAN")

DEFAULT_PRUNE_METHODS=("random,frame")

DEFAULT_PRUNE_RATIOS=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

DEFAULT_PRUNE_LAYER_IDX=2

show_usage() {
    echo "Qwen2.5 Model Batch Testing Script (Token Prune Method)"
    echo ""
    echo "Usage: $0 [options] <task_name>"
    echo ""
    echo "Supported tasks:"
    for task in "${SUPPORTED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "Options:"
    echo "  -g, --gpu-id <id>           GPU ID (default: $DEFAULT_GPU_ID)"
    echo "  -s, --sample-limit <num>    Sample count limit (default: $DEFAULT_SAMPLE_LIMIT, 0 means no limit)"
    echo "  -m, --methods <methods>     Prune method list, comma separated (default: ${DEFAULT_PRUNE_METHODS[*]})"
    echo "  -r, --ratios <ratios>       Prune ratio list, comma separated (default: ${DEFAULT_PRUNE_RATIOS[*]})"
    echo "  -l, --layer <idx>           Prune layer index (default: $DEFAULT_PRUNE_LAYER_IDX)"
    echo "  -c, --single-method <method> Single prune method test"
    echo "  -p, --single-ratio <ratio>  Single prune ratio test"
    echo "  -o, --output-dir <dir>      Results output directory (default: $DEFAULT_RESULTS_DIR)"
    echo "  --batch                     Enable batch mode, run all method and ratio combinations"
    echo "  --single METHOD RATIO       Run single experiment (method+ratio)"
    echo "  -h, --help                  Show this help information"
    echo ""
    echo "Supported Token pruning methods:"
    echo "  - base       : No pruning (baseline)"
    echo "  - fast_v     : Fast vision token pruning"
    echo "  - random     : Random token pruning"
    echo "  - frame      : Frame-based pruning"
    echo ""
    echo "Examples:"
    echo "  $0 --batch HAD                                    # Batch test HAD task, all method and ratio combinations"
    echo "  $0 -g 1 -s 100 --batch race                      # Batch test race task on GPU1, limit 100 samples"
    echo "  $0 -m base,fast_v -r 0,0.3,0.5 --batch TAU       # Test TAU task, specify certain methods and ratios"
    echo "  $0 --single fast_v 0.5 SLUE                      # Test SLUE task, single method and ratio"
    echo "  $0 -o ./my_results --batch VESUS                 # Test VESUS task, custom output directory"
    echo "  $0 -l 3 --batch Vox                              # Test Vox task, use layer 3 for pruning"
    echo "  $0 -c fast_v -p 0.3 Vox_age                      # Test Vox_age task, single configuration"
}

parse_args() {
    GPU_ID=$DEFAULT_GPU_ID
    SAMPLE_LIMIT=$DEFAULT_SAMPLE_LIMIT
    RESULTS_DIR=$DEFAULT_RESULTS_DIR
    PRUNE_LAYER_IDX=$DEFAULT_PRUNE_LAYER_IDX
    TASK=""
    SINGLE_METHOD=""
    SINGLE_RATIO=""
    BATCH_MODE=false
    SINGLE_MODE=false
    
    IFS=','
    METHODS_STRING="${DEFAULT_PRUNE_METHODS[*]}"
    RATIOS_STRING="${DEFAULT_PRUNE_RATIOS[*]}"
    IFS=' '
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--gpu-id)
                GPU_ID="$2"
                shift 2
                ;;
            -s|--sample-limit)
                SAMPLE_LIMIT="$2"
                shift 2
                ;;
            -m|--methods)
                METHODS_STRING="$2"
                shift 2
                ;;
            -r|--ratios)
                RATIOS_STRING="$2"
                shift 2
                ;;
            -l|--layer)
                PRUNE_LAYER_IDX="$2"
                shift 2
                ;;
            -c|--single-method)
                SINGLE_METHOD="$2"
                shift 2
                ;;
            -p|--single-ratio)
                SINGLE_RATIO="$2"
                shift 2
                ;;
            -o|--output-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --batch)
                BATCH_MODE=true
                shift
                ;;
            --single)
                SINGLE_MODE=true
                SINGLE_METHOD="$2"
                SINGLE_RATIO="$3"
                shift 3
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$TASK" ]]; then
                    TASK="$1"
                else
                    print_error "Only one task can be specified"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "$TASK" ]]; then
        print_error "A task must be specified"
        show_usage
        exit 1
    fi
    
    if [[ ! " ${SUPPORTED_TASKS[@]} " =~ " ${TASK} " ]]; then
        print_error "Unsupported task: $TASK"
        print_info "Supported tasks: ${SUPPORTED_TASKS[*]}"
        exit 1
    fi
    
    if [[ "$BATCH_MODE" == false && "$SINGLE_MODE" == false ]]; then
        print_error "Must specify --batch or --single mode"
        show_usage
        exit 1
    fi
    
    if [[ "$SINGLE_MODE" == true ]]; then
        if [[ -z "$SINGLE_METHOD" || -z "$SINGLE_RATIO" ]]; then
            print_error "--single mode requires specifying method and ratio"
            show_usage
            exit 1
        fi
    fi
    
    if [[ -n "$SINGLE_METHOD" && "$BATCH_MODE" == true ]]; then
        METHODS_STRING="$SINGLE_METHOD"
    fi
    
    if [[ -n "$SINGLE_RATIO" && "$BATCH_MODE" == true ]]; then
        RATIOS_STRING="$SINGLE_RATIO"
    fi
    
    IFS=',' read -ra PRUNE_METHODS <<< "$METHODS_STRING"
    IFS=',' read -ra PRUNE_RATIOS <<< "$RATIOS_STRING"
    IFS=' '
    
    valid_methods=("base" "fast_v" "random" "frame")
    for method in "${PRUNE_METHODS[@]}"; do
        method=$(echo "$method" | xargs)
        if [[ ! " ${valid_methods[@]} " =~ " ${method} " ]]; then
            print_error "Invalid pruning method: $method"
            print_info "Supported methods: ${valid_methods[*]}"
            exit 1
        fi
    done
    
    for ratio in "${PRUNE_RATIOS[@]}"; do
        ratio=$(echo "$ratio" | xargs)
        if ! [[ "$ratio" =~ ^0(\.[0-9]+)?$|^1(\.0)?$ ]]; then
            print_error "Invalid pruning ratio: $ratio (should be between 0.0-1.0)"
            exit 1
        fi
    done
}

check_script_exists() {
    local task=$1
    local script_name=""
    
    case $task in
        "HAD")
            script_name="HAD_qwen2.5.py"
            ;;
        "race")
            script_name="race_qwen2.5.py"
            ;;
        "SLUE")
            script_name="SLUE_qwen2.5.py"
            ;;
        "TAU")
            script_name="TAU_qwen2.5.py"
            ;;
        "VESUS")
            script_name="VESUS_qwen2.5.py"
            ;;
        "Vox")
            script_name="Vox_qwen2.5.py"
            ;;
        "Vox_age")
            script_name="Vox_age_qwen2.5.py"
            ;;
        "LibriSpeech")
            script_name="LibriSpeech_qwen2.5.py"
            ;;
        "DESED")
            script_name="DESED_qwen2.5.py"
            ;;
        "GTZAN")
            script_name="GTZAN_qwen2.5.py"
            ;;
    esac
    
    if [[ -f "./$script_name" ]]; then
        echo "./$script_name"
        return 0
    fi
    
    if [[ -f "./Qwen/$script_name" ]]; then
        echo "./Qwen/$script_name"
        return 0
    fi
    
    if [[ -f "../Qwen/$script_name" ]]; then
        echo "../Qwen/$script_name"
        return 0
    fi
    
    if [[ -f "./$task/$script_name" ]]; then
        echo "./$task/$script_name"
        return 0
    fi
    
    if [[ -f "../$task/$script_name" ]]; then
        echo "../$task/$script_name"
        return 0
    fi
    
    case $task in
        "Vox"|"Vox_age")
            local paths=(
                "./Vox/$script_name"
                "../Vox/$script_name"
            )
            ;;
        "VESUS")
            local paths=(
                "./VESUS/$script_name"
                "../VESUS/$script_name"
            )
            ;;
        "HAD")
            local paths=(
                "./HAD/$script_name"
                "../HAD/$script_name"
            )
            ;;
        *)
            local paths=()
            ;;
    esac
    
    for path in "${paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    
    return 1
}

run_single_experiment() {
    local method=$1
    local ratio=$2
    local script_path=$3
    local output_dir=$4
    local test_name="${method}_ratio_${ratio}"
    
    print_info "Starting experiment: $TASK, pruning method: $method, pruning ratio: $ratio"
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export PRUNE_METHOD=$method
    export PRUNE_RATIO=$ratio
    export PRUNE_LAYER_IDX=$PRUNE_LAYER_IDX
    
    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi
    
    local cmd="python $script_path"
    
    local log_file="$output_dir/logs/${TASK}_log_${test_name}.txt"
    mkdir -p "$(dirname "$log_file")"
    
    print_info "Executing command: $cmd"
    print_info "Environment variables: PRUNE_METHOD=$method, PRUNE_RATIO=$ratio, PRUNE_LAYER_IDX=$PRUNE_LAYER_IDX"
    print_info "Log file: $log_file"
    
    local start_time=$(date +%s)
    
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Experiment completed: $test_name (duration: ${duration}s)"
        
        extract_accuracy_from_log "$log_file" "$test_name"
        
        move_result_files "$output_dir" "$method" "$ratio"
        
    else
        print_error "Experiment failed: $test_name"
        print_error "For detailed error information see: $log_file"
        return 1
    fi
}

move_result_files() {
    local output_dir=$1
    local method=$2
    local ratio=$3
    
    local gpu_id_pattern="gpu${GPU_ID}"
    
    local moved_files=0
    
    for pattern in "*${TASK}*${method}*${ratio}*" "*${TASK}*qwen*${method}*${ratio}*" "*timing*${method}*${ratio}*" "*results*${method}*${ratio}*"; do
        for file in $pattern; do
            if [[ -f "$file" && "$file" != *.log ]]; then
                mv "$file" "$output_dir/" 2>/dev/null && ((moved_files++))
                print_info "Moved result file: $file -> $output_dir/"
            fi
        done
    done
    
    for dir_pattern in "*Results*" "*_Results" "${TASK}_Results" "./${TASK}_Results"; do
        if [[ -d "$dir_pattern" ]]; then
            find "$dir_pattern" -name "*${method}*${ratio}*" -type f | while read -r file; do
                if [[ -f "$file" ]]; then
                    mv "$file" "$output_dir/" 2>/dev/null && ((moved_files++))
                    print_info "Moved result file: $file -> $output_dir/"
                fi
            done
            
            if [[ -z "$(ls -A "$dir_pattern" 2>/dev/null)" ]]; then
                rm -rf "$dir_pattern" 2>/dev/null
                print_info "Cleaned empty directory: $dir_pattern"
            fi
        fi
    done
    
    for ext in "json" "jsonl" "txt" "csv"; do
        for file in *.${ext}; do
            if [[ -f "$file" && "$file" != "batch_test.sh" && "$file" != *.log ]]; then
                if [[ "$file" == *"${TASK}"* && ("$file" == *"${method}"* || "$file" == *"${ratio}"*) ]]; then
                    mv "$file" "$output_dir/" 2>/dev/null && ((moved_files++))
                    print_info "Moved result file: $file -> $output_dir/"
                fi
            fi
        done
    done
    
    if [[ $moved_files -eq 0 ]]; then
        print_warning "No movable result files found, please check if script correctly generated output files"
        print_info "Current directory file list:"
        ls -la *.json *.txt 2>/dev/null | head -10 || print_info "  No related files"
    else
        print_info "Successfully moved $moved_files result files"
    fi
}

extract_accuracy_from_log() {
    local log_file=$1
    local test_name=$2
    
    local accuracy=""
    
    accuracy=$(grep -o "Overall accuracy: [0-9.]*[%]*" "$log_file" | tail -1)
    if [[ -z "$accuracy" ]]; then
        accuracy=$(grep -o "Total accuracy: [0-9.]*[%]*" "$log_file" | tail -1)
    fi
    if [[ -z "$accuracy" ]]; then
        accuracy=$(grep -o "Accuracy: [0-9.]*" "$log_file" | tail -1)
    fi
    if [[ -z "$accuracy" ]]; then
        accuracy=$(grep -o "F1 Score: [0-9.]*" "$log_file" | tail -1)
        if [[ -n "$accuracy" ]]; then
            accuracy="F1 $accuracy"
        fi
    fi
    if [[ -z "$accuracy" ]]; then
        accuracy=$(grep -o "Average throughput: [0-9.]* tokens/s" "$log_file" | tail -1)
        if [[ -n "$accuracy" ]]; then
            accuracy="Throughput $accuracy"
        fi
    fi
    
    if [[ -n "$accuracy" ]]; then
        print_info "  └─ $test_name: $accuracy"
    fi
}

generate_summary_report() {
    local summary_file="$RESULTS_DIR/logs/${TASK}_qwen25_batch_summary.txt"
    
    print_info "Generating summary report: $summary_file"
    
    {
        echo "Qwen2.5 Model Batch Testing Summary Report (Token Prune Method)"
        echo "==========================================="
        echo "Task: $TASK"
        echo "GPU ID: $GPU_ID"
        echo "Sample limit: $SAMPLE_LIMIT"
        echo "Prune layer index: $PRUNE_LAYER_IDX"
        echo "Test time: $(date)"
        echo ""
        echo "Test configuration:"
        echo "--------"
        echo "Pruning methods: ${PRUNE_METHODS[*]}"
        echo "Pruning ratios: ${PRUNE_RATIOS[*]}"
        echo ""
        echo "Test results:"
        echo "--------"
        
        for method in "${PRUNE_METHODS[@]}"; do
            echo "Pruning method: $method"
            for ratio in "${PRUNE_RATIOS[@]}"; do
                local test_name="${method}_ratio_${ratio}"
                local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"
                
                if [[ -f "$log_file" ]]; then
                    echo "  Ratio $ratio: Test completed"
                    local accuracy=$(grep -o "Overall accuracy: [0-9.]*[%]*\|Total accuracy: [0-9.]*[%]*\|Accuracy: [0-9.]*\|F1 Score: [0-9.]*" "$log_file" | tail -1)
                    if [[ -n "$accuracy" ]]; then
                        echo "    Result: $accuracy"
                    fi
                else
                    echo "  Ratio $ratio: Test failed or incomplete"
                fi
            done
            echo ""
        done
        
        echo ""
        echo "File locations:"
        echo "--------"
        echo "Result files:"
        find "$RESULTS_DIR" -name "*.json" | grep -v logs | sort
        echo ""
        echo "Log files:"
        find "$RESULTS_DIR/logs" -name "*.txt" | sort
        echo ""
        echo "All related files:"
        find "$RESULTS_DIR" -type f | head -30 | sort
        
    } > "$summary_file"
    
    print_success "Summary report generated: $summary_file"
}

main() {
    print_info "Qwen2.5 Model Batch Testing Script started (Token Prune Method)"
    
    parse_args "$@"
    
    print_info "Test configuration:"
    print_info "  Task: $TASK"
    print_info "  GPU ID: $GPU_ID"
    print_info "  Sample limit: $SAMPLE_LIMIT"
    print_info "  Prune layer index: $PRUNE_LAYER_IDX"
    if [[ "$BATCH_MODE" == true ]]; then
        print_info "  Pruning methods: ${PRUNE_METHODS[*]}"
        print_info "  Pruning ratios: ${PRUNE_RATIOS[*]}"
    else
        print_info "  Pruning method: $SINGLE_METHOD"
        print_info "  Pruning ratio: $SINGLE_RATIO"
    fi
    print_info "  Output directory: $RESULTS_DIR"
    
    print_info "Searching for script file..."
    script_path=$(check_script_exists "$TASK")
    if [[ $? -ne 0 ]]; then
        print_error "Cannot find Qwen2.5 script file for task $TASK"
        print_error "Please ensure script file exists in current directory or Qwen subdirectory"
        print_error "Expected filename: ${TASK}_qwen2.5.py"
        exit 1
    fi
    print_success "Found script file: $script_path"
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found, please ensure Python is installed and in PATH"
        exit 1
    fi
    
    RESULTS_DIR="${RESULTS_DIR}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR/logs"
    
    print_info "Results directory: $RESULTS_DIR"
    
    local total_start_time=$(date +%s)
    
    local total_tests=0
    local failed_tests=0
    
    if [[ "$BATCH_MODE" == true ]]; then
        total_tests=$((${#PRUNE_METHODS[@]} * ${#PRUNE_RATIOS[@]}))
        print_info "Starting batch testing, total $total_tests combinations"
        
        local test_count=0
        for method in "${PRUNE_METHODS[@]}"; do
            method=$(echo "$method" | xargs)
            for ratio in "${PRUNE_RATIOS[@]}"; do
                ratio=$(echo "$ratio" | xargs)
                ((test_count++))
                print_info "Progress: $test_count/$total_tests"
                
                if ! run_single_experiment "$method" "$ratio" "$script_path" "$RESULTS_DIR"; then
                    ((failed_tests++))
                fi
                
                if [[ $test_count -lt $total_tests ]]; then
                    print_info "Waiting 5 seconds before next test..."
                    sleep 5
                fi
            done
        done
    else
        total_tests=1
        print_info "Starting single experiment"
        
        if ! run_single_experiment "$SINGLE_METHOD" "$SINGLE_RATIO" "$script_path" "$RESULTS_DIR"; then
            ((failed_tests++))
        fi
    fi
    
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    generate_summary_report
    
    print_info "Qwen2.5 Model Batch Testing completed!"
    print_info "Total tests: $total_tests"
    print_info "Total duration: ${total_duration}s"
    if [[ $failed_tests -eq 0 ]]; then
        print_success "All tests completed successfully"
    else
        print_warning "$failed_tests tests failed"
    fi
    print_info "Logs saved in: $RESULTS_DIR/logs"
    print_info "Result files saved in: $RESULTS_DIR"
    
    print_info ""
    print_info "Tip: You can use analysis scripts to view detailed results:"
    print_info "  View time statistics: python time_analysis.py --dir $RESULTS_DIR"
    print_info "  View summary report: cat $RESULTS_DIR/logs/${TASK}_qwen25_batch_summary.txt"
}

if ! command -v bc &> /dev/null; then
    print_warning "bc command not found, will use simplified floating point processing"
fi

main "$@"