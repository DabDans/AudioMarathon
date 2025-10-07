#!/bin/bash

# DART Batch Testing Script
# Used to test DART performance under different sparsity ratios
# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Set color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
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

# Default configuration
DEFAULT_GPU_ID=0
DEFAULT_SAMPLE_LIMIT=0  # 0 means no limit
DEFAULT_PRUNED_LAYER=2
DEFAULT_RESULTS_DIR="/data/to/your/results/dir"

# Supported task list
SUPPORTED_TASKS=("HAD" "race" "SLUE" "TAU" "VESUS" "Vox" "Vox_age" "LibriSpeech" "DESED" "GTZAN")

# Default sparsity ratios
DEFAULT_RATIOS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Usage information
show_usage() {
    echo "DART batch testing script using aero1 model"
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
    echo "  -l, --pruned-layer <num>    Pruned layer count (default: $DEFAULT_PRUNED_LAYER)"
    echo "  -r, --ratios <ratios>       Sparsity ratios, comma separated (default: ${DEFAULT_RATIOS[*]})"
    echo "  -o, --output-dir <dir>      Output directory (default: $DEFAULT_RESULTS_DIR)"
    echo "  -h, --help                  Show this help information"
    echo ""
    echo "Examples:"
    echo "  $0 HAD                                    # Test HAD task, default ratios"
    echo "  $0 -g 1 -s 100 race                      # Test race task on GPU1, limit 100 samples"
    echo "  $0 -r 0.0,0.5,0.8 TAU                    # Test TAU with specific ratios"
    echo "  $0 -l 3 -o /data/to/your/custom_results SLUE # Test SLUE with pruned layer 3, custom output dir"
    echo "  $0 LibriSpeech                           # Test LibriSpeech ASR task"
    echo "  $0 DESED                                 # Test DESED sound event detection"
    echo "  $0 Vox_age                               # Test VoxCeleb age recognition"
    echo "  $0 GTZAN                                 # Test GTZAN music genre classification"
}

# Parse command line arguments
parse_args() {
    GPU_ID=$DEFAULT_GPU_ID
    SAMPLE_LIMIT=$DEFAULT_SAMPLE_LIMIT
    PRUNED_LAYER=$DEFAULT_PRUNED_LAYER
    RESULTS_DIR=$DEFAULT_RESULTS_DIR
    TASK=""
    
    # Convert default ratios array to string
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
    
    # Check if task is specified
    if [[ -z "$TASK" ]]; then
        print_error "A task must be specified"
        show_usage
        exit 1
    fi
    
    # Check if task is supported
    if [[ ! " ${SUPPORTED_TASKS[@]} " =~ " ${TASK} " ]]; then
        print_error "Unsupported task: $TASK"
        print_info "Supported tasks: ${SUPPORTED_TASKS[*]}"
        exit 1
    fi
    
    # Parse ratio string to array
    IFS=',' read -ra RATIOS <<< "$RATIOS_STRING"
    IFS=' '
}

# Check if script file exists
check_script_exists() {
    local task=$1
    local script_name=""
    
    case $task in
        "HAD")
            script_name="HAD_dart_aero1.py"
            ;;
        "race")
            script_name="race_dart_aero1.py"
            ;;
        "SLUE")
            script_name="SLUE_dart_aero1.py"
            ;;
        "TAU")
            script_name="TAU_dart_aero1.py"
            ;;
        "VESUS")
            script_name="VESUS_dart_aero1.py"
            ;;
        "Vox")
            script_name="Vox_dart_aero1.py"
            ;;
        "Vox_age")
            script_name="Vox_age_dart_aero1.py"
            ;;
        "LibriSpeech")
            script_name="LibriSpeech_dart_aero1.py"
            ;;
        "DESED")
            script_name="DESED_dart_aero1.py"
            ;;
        "GTZAN")
            script_name="GTZAN_dart_aero1.py"
            ;;
    esac
    
    # Check current directory
    if [[ -f "./$script_name" ]]; then
        echo "./$script_name"
        return 0
    fi
    
    # Check task subdirectory
    if [[ -f "./$task/$script_name" ]]; then
        echo "./$task/$script_name"
        return 0
    fi
    
    # Check parent directory's task subdirectory
    if [[ -f "../$task/$script_name" ]]; then
        echo "../$task/$script_name"
        return 0
    fi
    
    # Check special directories
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

# Run single test
run_single_test() {
    local ratio=$1
    local script_path=$2
    local test_name=""
    
    if (( $(echo "$ratio == 0.0" | bc -l) )); then
        test_name="base"
        sparse_flag="false"
    else
        test_name="sparse_${ratio}"
        sparse_flag="true"
    fi
    
    print_info "Starting test: $TASK, ratio: $ratio ($test_name)"
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi
    # Set results directory, excluding task name, let test script use its own default directory structure
    export RESULTS_DIR="$RESULTS_DIR"
    
    # Build command
    local cmd="python $script_path"
    cmd="$cmd --sparse $sparse_flag"
    cmd="$cmd --pruned_layer $PRUNED_LAYER"
    if [[ "$sparse_flag" == "true" ]]; then
        cmd="$cmd --reduction_ratio $ratio"
    fi
    
    # Create log file
    local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"
    mkdir -p "$(dirname "$log_file")"
    
    print_info "Executing command: $cmd"
    print_info "Log file: $log_file"
    
    # Run test
    local start_time=$(date +%s)
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Test finished: $test_name (Time: ${duration}s)"
        
        # Extract accuracy info if possible
        extract_accuracy_from_log "$log_file" "$test_name"
    else
        print_error "Test failed: $test_name"
        print_error "See details in: $log_file"
        return 1
    fi
}

# Extract accuracy info from log file
extract_accuracy_from_log() {
    local log_file=$1
    local test_name=$2
    
    # Try extracting different formats of accuracy
    local accuracy=""
    
    # Match "Overall Accuracy: XX.XX%"
    accuracy=$(grep -o "Overall Accuracy: [0-9.]*%" "$log_file" | tail -1)
    if [[ -z "$accuracy" ]]; then
        # Match "Total Accuracy: X.XXXX"
        accuracy=$(grep -o "Total Accuracy: [0-9.]*" "$log_file" | tail -1)
    fi
    if [[ -z "$accuracy" ]]; then
        # Match "Accuracy: X.XXXX"
        accuracy=$(grep -o "Accuracy: [0-9.]*" "$log_file" | tail -1)
    fi
    if [[ -z "$accuracy" ]]; then
        # Match "F1 Score: X.XXXX"
        accuracy=$(grep -o "F1 Score: [0-9.]*" "$log_file" | tail -1)
        if [[ -n "$accuracy" ]]; then
            accuracy="F1 $accuracy"
        fi
    fi
    
    if [[ -n "$accuracy" ]]; then
        print_info "  └─ $test_name: $accuracy"
    fi
}

# Generate summary report
generate_summary_report() {
    local summary_file="$RESULTS_DIR/logs/${TASK}_batch_test_summary.txt"
    
    print_info "Generating summary report: $summary_file"
    
    {
        echo "DART Batch Test Summary Report"
        echo "===================="
        echo "Task: $TASK"
        echo "GPU ID: $GPU_ID"
        echo "Sample Limit: $SAMPLE_LIMIT"
        echo "Pruned Layer Count: $PRUNED_LAYER"
        echo "Test Time: $(date)"
        echo ""
        echo "Test Results:"
        echo "--------"
        
        for ratio in "${RATIOS[@]}"; do
            if (( $(echo "$ratio == 0.0" | bc -l) )); then
                test_name="base"
            else
                test_name="sparse_${ratio}"
            fi
            
            # Find possible result file locations
            local result_file=""
            local possible_paths=(
                "$RESULTS_DIR/${TASK}_Results/${TASK}_results_dart_${test_name/_*}.json"
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
                
                # Try extracting accuracy from JSON
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
        print(f'F1 (macro): {summary[\"sklearn_metrics\"][\"f1_macro\"]:.4f}')
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
        echo "--------"
        find "$RESULTS_DIR" -name "*.json" -o -name "*.txt" | sort
        
    } > "$summary_file"
    
    print_success "Summary report generated: $summary_file"
}

# Main function
main() {
    print_info "DART Batch Test Script Started"
    
    # Parse arguments
    parse_args "$@"
    
    # Show configuration
    print_info "Test configuration:"
    print_info "  Task: $TASK"
    print_info "  GPU ID: $GPU_ID"
    print_info "  Sample Limit: $SAMPLE_LIMIT"
    print_info "  Pruned Layer Count: $PRUNED_LAYER"
    print_info "  Sparsity Ratios: ${RATIOS[*]}"
    print_info "  Output Directory: $RESULTS_DIR"
    
    # Check script file
    print_info "Searching for script file..."
    script_path=$(check_script_exists "$TASK")
    if [[ $? -ne 0 ]]; then
        print_error "Script file for task $TASK not found"
        print_error "Please make sure the script file exists in the current or task subdirectory"
        exit 1
    fi
    print_success "Script file found: $script_path"
    
    # Check Python and dependencies
    if ! command -v python &> /dev/null; then
        print_error "Python not found, please make sure Python is installed and in PATH"
        exit 1
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_DIR/logs"
    
    # Record start time
    local total_start_time=$(date +%s)
    
    # Run tests
    print_info "Starting batch tests, total ${#RATIOS[@]} ratios"
    local failed_tests=0
    
    for i in "${!RATIOS[@]}"; do
        local ratio=${RATIOS[$i]}
        print_info "Progress: $((i+1))/${#RATIOS[@]}"
        
        if ! run_single_test "$ratio" "$script_path"; then
            ((failed_tests++))
        fi
        
        # Add delay between tests to avoid GPU overheating
        if [[ $i -lt $((${#RATIOS[@]}-1)) ]]; then
            print_info "Waiting 5 seconds before next test..."
            sleep 5
        fi
    done
    
    # Calculate total duration
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    # Generate summary report
    generate_summary_report
    
    # Show final results
    print_info "Batch testing complete!"
    print_info "Total duration: ${total_duration}s"
    if [[ $failed_tests -eq 0 ]]; then
        print_success "All tests completed successfully"
    else
        print_warning "$failed_tests tests failed"
    fi
    print_info "Logs saved in: $RESULTS_DIR/logs"
    print_info "Result files saved in: $RESULTS_DIR (according to each test script's default directory structure)"
}

# Check if bc command is available (for float comparisons)
if ! command -v bc &> /dev/null; then
    print_warning "bc command not found, using simplified float comparison"
    # Define simplified float comparison function
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

# Run main function
main "$@"