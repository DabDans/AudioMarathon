#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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
DEFAULT_PRUNED_LAYER=2
DEFAULT_RESULTS_DIR="./Qwen_DART_Results"

SUPPORTED_TASKS=("HAD" "race" "SLUE" "TAU" "VESUS" "Vox" "Vox_age" "LibriSpeech" "DESED" "GTZAN")

DEFAULT_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

show_usage() {
    echo "Qwen2.5-Omni DART Batch Testing Script"
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
    echo "  -s, --sample-limit <num>    Sample limit (default: $DEFAULT_SAMPLE_LIMIT, 0 means no limit)"
    echo "  -l, --pruned-layer <num>    Number of pruned layers (default: $DEFAULT_PRUNED_LAYER)"
    echo "  -r, --ratios <ratios>       Sparsity ratio list, comma-separated (default: ${DEFAULT_RATIOS[*]})"
    echo "  -o, --output-dir <dir>      Results output directory (default: $DEFAULT_RESULTS_DIR)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 HAD                                    # Test HAD task with default ratios"
    echo "  $0 -g 1 -s 100 race                      # Test race task on GPU1 with 100 samples limit"
    echo "  $0 -r 0.0,0.5,0.8 TAU                    # Test TAU task with specific ratios"
    echo "  $0 -l 3 -o ./my_results SLUE             # Test SLUE task with 3 pruned layers, custom output dir"
    echo "  $0 LibriSpeech                           # Test LibriSpeech speech recognition task"
    echo "  $0 DESED                                  # Test DESED sound event detection task"
    echo "  $0 Vox_age                                # Test VoxCeleb age recognition task"
    echo "  $0 GTZAN                                  # Test GTZAN music genre classification task"
}

parse_args() {
    GPU_ID=$DEFAULT_GPU_ID
    SAMPLE_LIMIT=$DEFAULT_SAMPLE_LIMIT
    PRUNED_LAYER=$DEFAULT_PRUNED_LAYER
    RESULTS_DIR=$DEFAULT_RESULTS_DIR
    TASK=""
    
    IFS=','
    RATIOS_STRING="${DEFAULT_RATIOS[*]}"
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
            -l|--pruned-layer)
                PRUNED_LAYER="$2"
                shift 2
                ;;
            -r|--ratios)
                RATIOS_STRING="$2"
                shift 2
                ;;
            -o|--output-dir)
                RESULTS_DIR="$2"
                shift 2
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
    
    IFS=',' read -ra RATIOS <<< "$RATIOS_STRING"
    IFS=' '
}

check_script_exists() {
    local task=$1
    local script_name=""
    
    case $task in
        "HAD")
            script_name="HAD_qwen_dart.py"
            ;;
        "race")
            script_name="race_qwen_dart.py"
            ;;
        "SLUE")
            script_name="SLUE_qwen_dart.py"
            ;;
        "TAU")
            script_name="TAU_qwen_dart.py"
            ;;
        "VESUS")
            script_name="VESUS_qwen_dart.py"
            ;;
        "Vox")
            script_name="Vox_qwen_dart.py"
            ;;
        "Vox_age")
            script_name="Vox_age_qwen_dart.py"
            ;;
        "LibriSpeech")
            script_name="LibriSpeech_qwen_dart.py"
            ;;
        "DESED")
            script_name="DESED_qwen_dart.py"
            ;;
        "GTZAN")
            script_name="GTZAN_qwen_dart.py"
            ;;
    esac
    
    if [[ -f "./$script_name" ]]; then
        echo "./$script_name"
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
        "LibriSpeech")
            if [[ -f "./LibriSpeech-Long/$script_name" ]]; then
                echo "./LibriSpeech-Long/$script_name"
                return 0
            fi
            if [[ -f "../LibriSpeech-Long/$script_name" ]]; then
                echo "../LibriSpeech-Long/$script_name"
                return 0
            fi
            ;;
        "DESED")
            if [[ -f "./DESED_test/$script_name" ]]; then
                echo "./DESED_test/$script_name"
                return 0
            fi
            if [[ -f "../DESED_test/$script_name" ]]; then
                echo "../DESED_test/$script_name"
                return 0
            fi
            ;;
        "GTZAN")
            if [[ -f "./GTZAN/$script_name" ]]; then
                echo "./GTZAN/$script_name"
                return 0
            fi
            if [[ -f "../GTZAN/$script_name" ]]; then
                echo "../GTZAN/$script_name"
                return 0
            fi
            ;;
    esac
    
    return 1
}

run_single_test() {
    local ratio=$1
    local script_path=$2
    local test_name=""
    
    if (( $(echo "$ratio == 0.0" | bc -l) )); then
        test_name="base"
        sparse_flag="False"
    else
        test_name="sparse_${ratio}"
        sparse_flag="True"
    fi
    
    print_info "Starting test: $TASK, ratio: $ratio ($test_name)"
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi
    
    local cmd="python $script_path"
    cmd="$cmd --sparse $sparse_flag"
    cmd="$cmd --pruned_layer $PRUNED_LAYER"
    if [[ "$sparse_flag" == "True" ]]; then
        cmd="$cmd --reduction_ratio $ratio"
    fi
    
    local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"
    mkdir -p "$(dirname "$log_file")"
    
    print_info "Executing command: $cmd"
    print_info "Log file: $log_file"
    
    local start_time=$(date +%s)
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Test completed: $test_name (duration: ${duration}s)"
        
        extract_accuracy_from_log "$log_file" "$test_name"
    else
        print_error "Test failed: $test_name"
        print_error "Detailed error information in: $log_file"
        return 1
    fi
}

extract_accuracy_from_log() {
    local log_file=$1
    local test_name=$2
    
    local accuracy=""
    
    accuracy=$(grep -o "Overall accuracy: [0-9.]*%" "$log_file" | tail -1)
    if [[ -z "$accuracy" ]]; then
        accuracy=$(grep -o "Total accuracy: [0-9.]*" "$log_file" | tail -1)
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
    
    if [[ -n "$accuracy" ]]; then
        print_info "  └─ $test_name: $accuracy"
    fi
}

generate_summary_report() {
    local summary_file="$RESULTS_DIR/logs/${TASK}_batch_test_summary.txt"
    
    print_info "Generating summary report: $summary_file"
    
    {
        echo "Qwen2.5-Omni DART Batch Test Summary Report"
        echo "==========================================="
        echo "Task: $TASK"
        echo "GPU ID: $GPU_ID"
        echo "Sample limit: $SAMPLE_LIMIT"
        echo "Pruned layers: $PRUNED_LAYER"
        echo "Test time: $(date)"
        echo ""
        echo "Test results:"
        echo "------------"
        
        for ratio in "${RATIOS[@]}"; do
            if (( $(echo "$ratio == 0.0" | bc -l) )); then
                test_name="base"
            else
                test_name="sparse_${ratio}"
            fi
            
            local result_file=""
            local possible_paths=(
                "$RESULTS_DIR/${TASK}_Results/${TASK}_results_qwen_dart_${test_name/_*}.json"
                "$RESULTS_DIR/${TASK}_Results/${TASK}_results_gpu*_${test_name/_*}_*.json"
                "$RESULTS_DIR/*_Results/${TASK}*results*.json"
            )
            
            for path_pattern in "${possible_paths[@]}"; do
                local found_file=$(find "$RESULTS_DIR" -name "$(basename "$path_pattern")" -type f 2>/dev/null | head -1)
                if [[ -n "$found_file" ]]; then
                    result_file="$found_file"
                    break
                fi
            done
            
            if [[ -f "$result_file" ]]; then
                echo "Ratio $ratio ($test_name): Result file exists"
                
                if command -v python3 &> /dev/null; then
                    local accuracy=$(python3 -c "
import json
try:
    with open('$result_file', 'r', encoding='utf-8') as f:
        data = json.load(f)
    summary = data.get('summary', {})
    if 'overall_accuracy' in summary:
        print(f'Accuracy: {summary[\"overall_accuracy\"]:.4f}')
    elif 'accuracy' in summary:
        print(f'Accuracy: {summary[\"accuracy\"]:.4f}')
    elif 'f1_macro' in summary.get('sklearn_metrics', {}):
        print(f'F1 (macro avg): {summary[\"sklearn_metrics\"][\"f1_macro\"]:.4f}')
    elif 'f1_score' in summary.get('metrics', {}):
        print(f'F1: {summary[\"metrics\"][\"f1_score\"]:.4f}')
except Exception as e:
    print('Cannot parse: ' + str(e))
" 2>/dev/null)
                    if [[ -n "$accuracy" ]]; then
                        echo "  └─ $accuracy"
                    fi
                fi
            else
                echo "Ratio $ratio ($test_name): Result file missing"
            fi
        done
        
        echo ""
        echo "File locations:"
        echo "--------------"
        find "$RESULTS_DIR" -name "*.json" -o -name "*.txt" | sort
        
    } > "$summary_file"
    
    print_success "Summary report generated: $summary_file"
}

main() {
    print_info "Qwen2.5-Omni DART Batch Testing Script started"
    
    parse_args "$@"
    
    print_info "Test configuration:"
    print_info "  Task: $TASK"
    print_info "  GPU ID: $GPU_ID"
    print_info "  Sample limit: $SAMPLE_LIMIT"
    print_info "  Pruned layers: $PRUNED_LAYER"
    print_info "  Sparsity ratios: ${RATIOS[*]}"
    print_info "  Output directory: $RESULTS_DIR"
    
    print_info "Looking for script file..."
    script_path=$(check_script_exists "$TASK")
    if [[ $? -ne 0 ]]; then
        print_error "Cannot find script file for task $TASK"
        print_error "Please ensure the script file exists in the current directory or task subdirectory"
        exit 1
    fi
    print_success "Found script file: $script_path"
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found, please ensure Python is installed and in PATH"
        exit 1
    fi
    
    mkdir -p "$RESULTS_DIR/logs"
    
    local total_start_time=$(date +%s)
    
    print_info "Starting batch test with ${#RATIOS[@]} ratios"
    local failed_tests=0
    
    for i in "${!RATIOS[@]}"; do
        local ratio=${RATIOS[$i]}
        print_info "Progress: $((i+1))/${#RATIOS[@]}"
        
        if ! run_single_test "$ratio" "$script_path"; then
            ((failed_tests++))
        fi
        
        if [[ $i -lt $((${#RATIOS[@]}-1)) ]]; then
            print_info "Waiting 5 seconds before next test..."
            sleep 5
        fi
    done
    
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    generate_summary_report
    
    print_info "Batch test completed!"
    print_info "Total duration: ${total_duration}s"
    if [[ $failed_tests -eq 0 ]]; then
        print_success "All tests completed successfully"
    else
        print_warning "$failed_tests tests failed"
    fi
    print_info "Logs saved in: $RESULTS_DIR/logs"
    print_info "Result files saved in: $RESULTS_DIR (following default directory structure of each test script)"
}

if ! command -v bc &> /dev/null; then
    print_warning "bc command not found, using simplified float processing"
    bc() {
        if [[ "$1" =~ ^.*==.*0\.0.*$ ]]; then
            local num=$(echo "$1" | sed 's/.*== *\([0-9.]*\).*/\1/')
            if [[ "$num" == "0.0" || "$num" == "0" ]]; then
                echo "1"
            else
                echo "0"
            fi
        else
            echo "0"
        fi
    }
fi

main "$@"