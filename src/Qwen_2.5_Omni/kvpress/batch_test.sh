#!/bin/bash

# KV Press Batch Test Script
# Used to test KV Press performance under different compression ratios and types

# Set environment variables (keep consistent with dart_batch_test.sh)
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
DEFAULT_SAMPLE_LIMIT=0  # 0 means unlimited
DEFAULT_RESULTS_DIR="/data/to/your/results/path"

# Supported task list (corresponds to scripts in Qwen_kvpress directory)
SUPPORTED_TASKS=("race" "DESED" "TAU" "Vox" "LibriSpeech" "GTZAN" "SLUE" "VESUS" "HAD" "Vox_age")

# Default compression ratio list
DEFAULT_COMPRESSION_RATIOS=(0.1 0.2 0.4 0.5 0.6 0.8)

# Default compression type list
DEFAULT_PRESS_TYPES=("knorm" "random" "tova" "snap")

# Usage information
show_usage() {
    echo "Qwen2.5-Omni + KV Press Batch Test Script"
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
    echo "  -s, --sample-limit <num>    Sample limit (default: $DEFAULT_SAMPLE_LIMIT, 0=unlimited)"
    echo "  -r, --ratios <ratios>       Comma-separated list of compression ratios (default: ${DEFAULT_COMPRESSION_RATIOS[*]})"
    echo "  -t, --types <types>         Comma-separated list of compression types (default: ${DEFAULT_PRESS_TYPES[*]})"
    echo "  -c, --single-ratio <ratio>  Test single compression ratio"
    echo "  -p, --single-type <type>    Test single compression type"
    echo "  -b, --baseline              Include baseline test (no compression)"
    echo "  -o, --output-dir <dir>      Results output directory (default: $DEFAULT_RESULTS_DIR)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Supported compression types:"
    echo "  - knorm       : Key-norm based attention compression (default)"
    echo "  - expected    : RoPE-based expected attention compression"
    echo "  - observed    : Observed attention score based"
    echo "  - random      : Randomly drop tokens"
    echo "  - tova        : Time-order based attention value analysis"
    echo "  - snap        : SnapKV compression"
    echo "  - streaming   : StreamingLLM compression (for long text)"
    echo ""
    echo "Task descriptions:"
    echo "  - race        : RACE reading comprehension"
    echo "  - DESED       : Sound event detection"
    echo "  - TAU         : Urban sound scene classification"
    echo "  - Vox         : Speaker verification"
    echo "  - LibriSpeech : Speech recognition (ASR)"
    echo "  - GTZAN       : Music genre classification"
    echo "  - SLUE        : Speech understanding"
    echo "  - VESUS       : Video understanding"
    echo "  - HAD         : Hearing aid assessment"
    echo "  - Vox_age     : Age estimation"
    echo ""
    echo "Examples:"
    echo "  $0 race                                   # Test RACE task, all combinations"
    echo "  $0 -g 1 -s 100 TAU                        # Test TAU on GPU 1, limit 100 samples"
    echo "  $0 -r 0.3,0.5,0.7 -t knorm,expected DESED # Test DESED with specific ratios/types"
    echo "  $0 -c 0.5 -p knorm SLUE                   # Test SLUE with a single ratio/type"
    echo "  $0 -b -o /data/to/your/custom_results/path VESUS # Test VESUS with baseline, custom output dir"
    echo "  $0 LibriSpeech                            # Test LibriSpeech ASR"
    echo "  $0 GTZAN                                  # Test GTZAN music genre classification"
}

# Parse command line arguments
parse_args() {
    GPU_ID=$DEFAULT_GPU_ID
    SAMPLE_LIMIT=$DEFAULT_SAMPLE_LIMIT
    RESULTS_DIR=$DEFAULT_RESULTS_DIR
    TASK=""
    SINGLE_RATIO=""
    SINGLE_TYPE=""
    INCLUDE_BASELINE=false

    # Convert default ratio/type arrays to strings
    IFS=','
    RATIOS_STRING="${DEFAULT_COMPRESSION_RATIOS[*]}"
    TYPES_STRING="${DEFAULT_PRESS_TYPES[*]}"
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
            -r|--ratios)
                RATIOS_STRING="$2"
                shift 2
                ;;
            -t|--types)
                TYPES_STRING="$2"
                shift 2
                ;;
            -c|--single-ratio)
                SINGLE_RATIO="$2"
                shift 2
                ;;
            -p|--single-type)
                SINGLE_TYPE="$2"
                shift 2
                ;;
            -b|--baseline)
                INCLUDE_BASELINE=true
                shift
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

    # Check if supported
    if [[ ! " ${SUPPORTED_TASKS[@]} " =~ " ${TASK} " ]]; then
        print_error "Unsupported task: $TASK"
        print_info "Supported tasks: ${SUPPORTED_TASKS[*]}"
        exit 1
    fi

    # Handle single ratio/type
    if [[ -n "$SINGLE_RATIO" ]]; then
        RATIOS_STRING="$SINGLE_RATIO"
    fi

    if [[ -n "$SINGLE_TYPE" ]]; then
        TYPES_STRING="$SINGLE_TYPE"
    fi

    # Parse ratio/type strings to arrays
    IFS=',' read -ra COMPRESSION_RATIOS <<< "$RATIOS_STRING"
    IFS=',' read -ra PRESS_TYPES <<< "$TYPES_STRING"
    IFS=' '

    # Validate ratios
    for ratio in "${COMPRESSION_RATIOS[@]}"; do
        if ! [[ "$ratio" =~ ^0\.[0-9]+$|^1\.0$|^0$ ]]; then
            print_error "Invalid compression ratio: $ratio (should be between 0.0-1.0)"
            exit 1
        fi
    done

    # Validate types
    valid_types=("knorm" "expected" "observed" "random" "tova" "snap" "streaming")
    for press_type in "${PRESS_TYPES[@]}"; do
        if [[ ! " ${valid_types[@]} " =~ " ${press_type} " ]]; then
            print_error "Invalid compression type: $press_type"
            print_info "Supported types: ${valid_types[*]}"
            exit 1
        fi
    done
}

# Check if script file exists
check_script_exists() {
    local task=$1
    local script_name=""

    # Determine script name by task
    case $task in
        "race")
            script_name="race_qwen_kvpress.py"
            ;;
        "DESED")
            script_name="DESED_qwen_kvpress.py"
            ;;
        "TAU")
            script_name="TAU_qwen_kvpress.py"
            ;;
        "Vox")
            script_name="Vox_qwen_kvpress.py"
            ;;
        "LibriSpeech")
            script_name="LibriSpeech_qwen_kvpress.py"
            ;;
        "GTZAN")
            script_name="GTZAN_qwen_kvpress.py"
            ;;
        "SLUE")
            script_name="SLUE_qwen_kvpress.py"
            ;;
        "VESUS")
            script_name="VESUS_qwen_kvpress.py"
            ;;
        "HAD")
            script_name="HAD_qwen_kvpress.py"
            ;;
        "Vox_age")
            script_name="Vox_age_qwen_kvpress.py"
            ;;
    esac

    # Check current dir
    if [[ -f "./$script_name" ]]; then
        echo "./$script_name"
        return 0
    fi

    # Check Qwen_kvpress dir
    if [[ -f "/data/to/your/qwen_kvpress/path/$script_name" ]]; then
        echo "/data/to/your/qwen_kvpress/path/$script_name"
        return 0
    fi

    # Check parent Qwen_kvpress dir
    if [[ -f "/data/to/your/parent_qwen_kvpress/path/$script_name" ]]; then
        echo "/data/to/your/parent_qwen_kvpress/path/$script_name"
        return 0
    fi

    # Check task subdir
    if [[ -f "/data/to/your/$task/path/$script_name" ]]; then
        echo "/data/to/your/$task/path/$script_name"
        return 0
    fi

    # Check parent task subdir
    if [[ -f "/data/to/your/parent_$task/path/$script_name" ]]; then
        echo "/data/to/your/parent_$task/path/$script_name"
        return 0
    fi

    return 1
}

# Run baseline test (no compression)
run_baseline_test() {
    local script_path=$1
    local test_name="baseline_no_compression"

    print_info "Starting baseline test: $TASK (no compression)"

    # Set env variables
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    unset COMPRESSION_RATIO
    unset PRESS_TYPE

    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi

    # Set results dir
    export RESULTS_DIR="$RESULTS_DIR"

    # Build command (add --no-compress flag)
    local cmd="python $script_path --no-compress"

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
        print_success "Baseline test completed: $test_name (Duration: ${duration}s)"

        # Extract accuracy info
        extract_accuracy_from_log "$log_file" "$test_name"
    else
        print_error "Baseline test failed: $test_name"
        print_error "See details in: $log_file"
        return 1
    fi
}

# Run single test
run_single_test() {
    local ratio=$1
    local press_type=$2
    local script_path=$3
    local test_name="${press_type}_ratio_${ratio}"

    print_info "Starting test: $TASK, type: $press_type, ratio: $ratio"

    # Set env variables
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export COMPRESSION_RATIO=$ratio
    export PRESS_TYPE=$press_type

    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi

    # Set results dir
    export RESULTS_DIR="$RESULTS_DIR"

    # Build command
    local cmd="python $script_path"

    # Create log file
    local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"
    mkdir -p "$(dirname "$log_file")"

    print_info "Executing command: $cmd"
    print_info "Env: COMPRESSION_RATIO=$ratio, PRESS_TYPE=$press_type"
    print_info "Log file: $log_file"

    # Run test
    local start_time=$(date +%s)
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Test completed: $test_name (Duration: ${duration}s)"

        # Extract accuracy info
        extract_accuracy_from_log "$log_file" "$test_name"
    else
        print_error "Test failed: $test_name"
        print_error "See details in: $log_file"
        return 1
    fi
}

# Extract accuracy info from log
extract_accuracy_from_log() {
    local log_file=$1
    local test_name=$2

    # Try extracting different accuracy metrics
    local metrics=""

    # Different metric patterns
    local patterns=(
        "Overall Accuracy: [0-9.]*[%]*"
        "Overall WER: [0-9.]*"
        "Exact Accuracy: [0-9.]*"
        "Fuzzy Accuracy: [0-9.]*"
        "Mean Absolute Error: [0-9.]*"
        "F1 Score: [0-9.]*"
        "Accuracy: [0-9.]*"
        "WER: [0-9.]*"
        "Mean Throughput: [0-9.]* tok/s"
    )

    for pattern in "${patterns[@]}"; do
        local metric=$(grep -o "$pattern" "$log_file" | tail -1)
        if [[ -n "$metric" ]]; then
            if [[ -z "$metrics" ]]; then
                metrics="$metric"
            else
                metrics="$metrics, $metric"
            fi
        fi
    done

    if [[ -n "$metrics" ]]; then
        print_info "  └─ $test_name: $metrics"
    else
        # Try extracting GPU memory usage
        local gpu_mem=$(grep -o "Mean GPU Peak Memory: [0-9.]* GB" "$log_file" | tail -1)
        if [[ -n "$gpu_mem" ]]; then
            print_info "  └─ $test_name: $gpu_mem"
        fi
    fi
}

# Generate summary report
generate_summary_report() {
    local summary_file="$RESULTS_DIR/logs/${TASK}_qwen_kvpress_batch_summary.txt"

    print_info "Generating summary report: $summary_file"

    {
        echo "Qwen2.5-Omni + KV Press Batch Test Summary Report"
        echo "========================================"
        echo "Task: $TASK"
        echo "GPU ID: $GPU_ID"
        echo "Sample limit: $SAMPLE_LIMIT"
        echo "Test time: $(date)"
        echo ""
        echo "Test configuration:"
        echo "--------"
        echo "Compression ratios: ${COMPRESSION_RATIOS[*]}"
        echo "Compression types: ${PRESS_TYPES[*]}"
        echo "Include baseline: $INCLUDE_BASELINE"
        echo ""
        echo "Test results:"
        echo "--------"

        # Baseline test result
        if [[ "$INCLUDE_BASELINE" == "true" ]]; then
            echo "Baseline test (no compression):"
            local baseline_log="$RESULTS_DIR/logs/${TASK}_log_baseline_no_compression.txt"
            if [[ -f "$baseline_log" ]]; then
                echo "  Baseline: completed"
                # Extract baseline metrics
                extract_accuracy_from_log "$baseline_log" "baseline" >> /tmp/baseline_metrics.txt 2>/dev/null
                if [[ -f "/tmp/baseline_metrics.txt" ]]; then
                    cat /tmp/baseline_metrics.txt
                    rm -f /tmp/baseline_metrics.txt
                fi
            else
                echo "  Baseline: failed or incomplete"
            fi
            echo ""
        fi

        # KV Press test results
        for press_type in "${PRESS_TYPES[@]}"; do
            echo "Compression type: $press_type"
            for ratio in "${COMPRESSION_RATIOS[@]}"; do
                local test_name="${press_type}_ratio_${ratio}"
                local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"

                if [[ -f "$log_file" ]]; then
                    echo "  Ratio $ratio: completed"
                    # Extract metrics
                    extract_accuracy_from_log "$log_file" "$test_name" >> /tmp/test_metrics.txt 2>/dev/null
                    if [[ -f "/tmp/test_metrics.txt" ]]; then
                        tail -1 /tmp/test_metrics.txt | sed 's/^.*└─[^:]*: /    /'
                        rm -f /tmp/test_metrics.txt
                    fi
                else
                    echo "  Ratio $ratio: failed or incomplete"
                fi
            done
            echo ""
        done

        echo ""
        echo "Result file location:"
        echo "------------"
        echo "JSON result files:"
        find "$RESULTS_DIR" -name "*.json" -not -path "*/logs/*" | sort
        echo ""
        echo "Log files:"
        find "$RESULTS_DIR/logs" -name "*.txt" | sort

        echo ""
        echo "Performance summary (if available):"
        echo "----------------"
        # Try extracting performance stats from result files
        for json_file in $(find "$RESULTS_DIR" -name "*.json" -not -path "*/logs/*"); do
            if [[ -f "$json_file" ]]; then
                local filename=$(basename "$json_file")
                echo "File: $filename"

                # Use python to extract key metrics (if available)
                if command -v python3 &> /dev/null; then
                    python3 -c "
import json
import sys
try:
    with open('$json_file', 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary = data.get('summary', {})
    timing = summary.get('timing', {})

    # Extract key metrics
    metrics = {}
    if 'overall_accuracy' in summary:
        metrics['accuracy'] = summary['overall_accuracy']
    elif 'exact_accuracy' in summary:
        metrics['exact_acc'] = summary['exact_accuracy']
    elif 'avg_age_error' in summary:
        metrics['mae'] = summary['avg_age_error']

    if 'avg_tokens_per_sec' in timing:
        metrics['throughput'] = timing['avg_tokens_per_sec']
    if 'avg_gpu_peak_mem_gb' in timing:
        metrics['gpu_mem'] = timing['avg_gpu_peak_mem_gb']

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.4f}')
        else:
            print(f'  {key}: {value}')

except Exception as e:
    print(f'  Parse failed: {str(e)}')
" 2>/dev/null
                fi
                echo ""
            fi
        done

    } > "$summary_file"

    print_success "Summary report generated: $summary_file"
}

# Main function
main() {
    print_info "Qwen2.5-Omni + KV Press Batch Test Script started"

    # Parse arguments
    parse_args "$@"

    # Show config
    print_info "Test configuration:"
    print_info "  Task: $TASK"
    print_info "  GPU ID: $GPU_ID"
    print_info "  Sample limit: $SAMPLE_LIMIT"
    print_info "  Compression ratios: ${COMPRESSION_RATIOS[*]}"
    print_info "  Compression types: ${PRESS_TYPES[*]}"
    print_info "  Include baseline: $INCLUDE_BASELINE"
    print_info "  Output directory: $RESULTS_DIR"

    # Check script file
    print_info "Finding script file..."
    script_path=$(check_script_exists "$TASK")
    if [[ $? -ne 0 ]]; then
        print_error "Cannot find Qwen KV Press script file for task $TASK"
        print_error "Please make sure the script file exists in Qwen_kvpress directory"
        case $TASK in
            "race") print_error "Expected file: race_qwen_kvpress.py" ;;
            "DESED") print_error "Expected file: DESED_qwen_kvpress.py" ;;
            "TAU") print_error "Expected file: TAU_qwen_kvpress.py" ;;
            "Vox") print_error "Expected file: Vox_qwen_kvpress.py" ;;
            "LibriSpeech") print_error "Expected file: LibriSpeech_qwen_kvpress.py" ;;
            "GTZAN") print_error "Expected file: GTZAN_qwen_kvpress.py" ;;
            "SLUE") print_error "Expected file: SLUE_qwen_kvpress.py" ;;
            "VESUS") print_error "Expected file: VESUS_qwen_kvpress.py" ;;
            "HAD") print_error "Expected file: HAD_qwen_kvpress.py" ;;
            "Vox_age") print_error "Expected file: Vox_age_qwen_kvpress.py" ;;
        esac
        exit 1
    fi
    print_success "Script file found: $script_path"

    # Check Python and dependencies
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please ensure Python is installed and in PATH"
        exit 1
    fi

    # Create results dir
    mkdir -p "$RESULTS_DIR/logs"

    # Record start time
    local total_start_time=$(date +%s)

    # Calculate total tests
    local total_tests=$((${#COMPRESSION_RATIOS[@]} * ${#PRESS_TYPES[@]}))
    if [[ "$INCLUDE_BASELINE" == "true" ]]; then
        total_tests=$((total_tests + 1))
    fi

    local failed_tests=0
    local test_count=0

    print_info "Starting batch test, total $total_tests tests"

    # Run baseline test
    if [[ "$INCLUDE_BASELINE" == "true" ]]; then
        ((test_count++))
        print_info "Progress: $test_count/$total_tests (baseline test)"

        if ! run_baseline_test "$script_path"; then
            ((failed_tests++))
        fi

        # Test interval
        if [[ $test_count -lt $total_tests ]]; then
            print_info "Wait 5 seconds before next test..."
            sleep 5
        fi
    fi

    # Run KV Press tests
    for press_type in "${PRESS_TYPES[@]}"; do
        for ratio in "${COMPRESSION_RATIOS[@]}"; do
            ((test_count++))
            print_info "Progress: $test_count/$total_tests ($press_type, ratio $ratio)"

            if ! run_single_test "$ratio" "$press_type" "$script_path"; then
                ((failed_tests++))
            fi

            # Test interval
            if [[ $test_count -lt $total_tests ]]; then
                print_info "Wait 5 seconds before next test..."
                sleep 5
            fi
        done
    done

    # Calculate total duration
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))

    # Generate summary report
    generate_summary_report

    # Show final result
    print_info "Qwen2.5-Omni + KV Press batch test finished!"
    print_info "Total tests: $total_tests"
    print_info "Total duration: ${total_duration}s"
    print_info "Average per test: $((total_duration / total_tests))s"

    if [[ $failed_tests -eq 0 ]]; then
        print_success "All tests completed successfully"
    else
        print_warning "$failed_tests tests failed"
    fi

    print_info "Detailed results:"
    print_info "  Log files: $RESULTS_DIR/logs"
    print_info "  Result files: $RESULTS_DIR"
    print_info "  Summary report: $RESULTS_DIR/logs/${TASK}_qwen_kvpress_batch_summary.txt"
}

# Run main function
main "$@"