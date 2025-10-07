#!/bin/bash

# DART Batch Test Script - Multi-dataset Support Version
# Used to test DART performance at different sparsity ratios, supports testing multiple datasets at the same time
# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_task() {
    echo -e "${PURPLE}[TASK]${NC} $1"
}

print_progress() {
    echo -e "${CYAN}[PROGRESS]${NC} $1"
}

# Default configuration
DEFAULT_GPU_ID=0
DEFAULT_SAMPLE_LIMIT=0  # 0 means no limit
DEFAULT_PRUNED_LAYER=2
DEFAULT_RESULTS_DIR="/data/to/your/DART_Batch_Results/path"
DEFAULT_PARALLEL_TASKS=1  # Default serial execution
DEFAULT_TASK_INTERVAL=10  # Task interval time (seconds)

# Supported task list
SUPPORTED_TASKS=("HAD" "race" "SLUE" "TAU" "VESUS" "Vox" "Vox_age" "LibriSpeech" "DESED" "GTZAN")

# Task classification
AUDIO_TASKS=("HAD" "TAU" "VESUS" "Vox" "Vox_age" "LibriSpeech" "DESED" "GTZAN")
TEXT_TASKS=("race" "SLUE")

# Default sparsity ratios
DEFAULT_RATIOS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Usage instructions
show_usage() {
    echo "DART Batch Test Script - Multi-dataset Support Version"
    echo ""
    echo "Usage: $0 [options] <task_list>"
    echo ""
    echo "Supported tasks:"
    echo "Audio tasks: ${AUDIO_TASKS[*]}"
    echo "Text tasks: ${TEXT_TASKS[*]}"
    echo ""
    echo "Options:"
    echo "  -g, --gpu-id <id>           GPU ID (default: $DEFAULT_GPU_ID)"
    echo "  -s, --sample-limit <num>    Sample limit (default: $DEFAULT_SAMPLE_LIMIT, 0 means unlimited)"
    echo "  -l, --pruned-layer <num>    Pruned layers (default: $DEFAULT_PRUNED_LAYER)"
    echo "  -r, --ratios <ratios>       Sparsity ratios, separated by commas (default: ${DEFAULT_RATIOS[*]})"
    echo "  -o, --output-dir <dir>      Output directory for results (default: $DEFAULT_RESULTS_DIR)"
    echo "  -p, --parallel <num>        Number of parallel tasks (default: $DEFAULT_PARALLEL_TASKS)"
    echo "  -i, --interval <seconds>    Task interval time (default: $DEFAULT_TASK_INTERVAL seconds)"
    echo "  -t, --task-filter <type>    Task type filter (audio/text/all, default: all)"
    echo "  --skip-base                 Skip base test (ratio=0.0)"
    echo "  --continue-on-error         Continue other tasks if error occurs"
    echo "  --dry-run                   Only show commands to be executed, do not actually execute"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Task list format:"
    echo "  - Single task: HAD"
    echo "  - Multiple tasks: HAD,race,SLUE"
    echo "  - All tasks: all"
    echo "  - Task type: audio (all audio tasks) or text (all text tasks)"
    echo ""
    echo "Examples:"
    echo "  $0 all                                    # Test all tasks"
    echo "  $0 HAD,race,SLUE                         # Test specified multiple tasks"
    echo "  $0 -t audio                               # Only test audio tasks"
    echo "  $0 -g 1 -s 100 -p 2 HAD,race             # Test 2 tasks in parallel on GPU1, limit to 100 samples"
    echo "  $0 -r 0.0,0.5,0.8 --skip-base TAU,Vox    # Test specific ratios, skip base test"
    echo "  $0 --dry-run LibriSpeech,DESED           # Preview commands to be executed"
    echo "  $0 --continue-on-error all               # Test all tasks, continue on error"
}

# Parse command-line arguments
parse_args() {
    GPU_ID=$DEFAULT_GPU_ID
    SAMPLE_LIMIT=$DEFAULT_SAMPLE_LIMIT
    PRUNED_LAYER=$DEFAULT_PRUNED_LAYER
    RESULTS_DIR=$DEFAULT_RESULTS_DIR
    PARALLEL_TASKS=$DEFAULT_PARALLEL_TASKS
    TASK_INTERVAL=$DEFAULT_TASK_INTERVAL
    TASK_FILTER="all"
    TASKS=""
    SKIP_BASE=false
    CONTINUE_ON_ERROR=false
    DRY_RUN=false
    
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
            -p|--parallel)
                PARALLEL_TASKS="$2"
                shift 2
                ;;
            -i|--interval)
                TASK_INTERVAL="$2"
                shift 2
                ;;
            -t|--task-filter)
                TASK_FILTER="$2"
                shift 2
                ;;
            --skip-base)
                SKIP_BASE=true
                shift
                ;;
            --continue-on-error)
                CONTINUE_ON_ERROR=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
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
                if [[ -z "$TASKS" ]]; then
                    TASKS="$1"
                else
                    print_error "Only one task list can be specified"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if tasks are specified
    if [[ -z "$TASKS" ]]; then
        print_error "Task list must be specified"
        show_usage
        exit 1
    fi
    
    # Parse task list
    parse_task_list "$TASKS"
    
    # Parse ratio string to array
    IFS=',' read -ra RATIOS <<< "$RATIOS_STRING"
    IFS=' '
    
    # If skipping base test, remove 0.0 from ratios
    if [[ "$SKIP_BASE" == true ]]; then
        local temp_ratios=()
        for ratio in "${RATIOS[@]}"; do
            if (( $(echo "$ratio != 0.0" | bc -l 2>/dev/null || echo "1") )); then
                temp_ratios+=("$ratio")
            fi
        done
        RATIOS=("${temp_ratios[@]}")
    fi
    
    # Validate parallel task number
    if [[ $PARALLEL_TASKS -lt 1 ]]; then
        print_error "Parallel tasks number must be greater than or equal to 1"
        exit 1
    fi
}

# Parse task list
parse_task_list() {
    local task_input="$1"
    TASK_LIST=()
    
    case "$task_input" in
        "all")
            TASK_LIST=("${SUPPORTED_TASKS[@]}")
            ;;
        "audio")
            TASK_LIST=("${AUDIO_TASKS[@]}")
            ;;
        "text")
            TASK_LIST=("${TEXT_TASKS[@]}")
            ;;
        *)
            # Parse comma-separated task list
            IFS=',' read -ra temp_tasks <<< "$task_input"
            for task in "${temp_tasks[@]}"; do
                task=$(echo "$task" | xargs)  # Remove spaces
                if [[ " ${SUPPORTED_TASKS[@]} " =~ " ${task} " ]]; then
                    TASK_LIST+=("$task")
                else
                    print_error "Unsupported task: $task"
                    print_info "Supported tasks: ${SUPPORTED_TASKS[*]}"
                    exit 1
                fi
            done
            ;;
    esac
    
    # Further filter by task filter
    if [[ "$TASK_FILTER" != "all" ]]; then
        local filtered_tasks=()
        for task in "${TASK_LIST[@]}"; do
            case "$TASK_FILTER" in
                "audio")
                    if [[ " ${AUDIO_TASKS[@]} " =~ " ${task} " ]]; then
                        filtered_tasks+=("$task")
                    fi
                    ;;
                "text")
                    if [[ " ${TEXT_TASKS[@]} " =~ " ${task} " ]]; then
                        filtered_tasks+=("$task")
                    fi
                    ;;
            esac
        done
        TASK_LIST=("${filtered_tasks[@]}")
    fi
    
    if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
        print_error "No matching tasks found"
        exit 1
    fi
}

# Check if the script file exists
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
    if [[ -f "/data/to/your/current/path/$script_name" ]]; then
        echo "/data/to/your/current/path/$script_name"
        return 0
    fi
    
    # Check task subdirectory
    if [[ -f "/data/to/your/$task/path/$script_name" ]]; then
        echo "/data/to/your/$task/path/$script_name"
        return 0
    fi
    
    # Check parent directory's task subdirectory
    if [[ -f "/data/to/your/parent/$task/path/$script_name" ]]; then
        echo "/data/to/your/parent/$task/path/$script_name"
        return 0
    fi
    
    # Check special directories
    case $task in
        "LibriSpeech")
            if [[ -f "/data/to/your/LibriSpeech-Long/path/$script_name" ]]; then
                echo "/data/to/your/LibriSpeech-Long/path/$script_name"
                return 0
            fi
            if [[ -f "/data/to/your/parent/LibriSpeech-Long/path/$script_name" ]]; then
                echo "/data/to/your/parent/LibriSpeech-Long/path/$script_name"
                return 0
            fi
            ;;
        "DESED")
            if [[ -f "/data/to/your/DESED_test/path/$script_name" ]]; then
                echo "/data/to/your/DESED_test/path/$script_name"
                return 0
            fi
            if [[ -f "/data/to/your/parent/DESED_test/path/$script_name" ]]; then
                echo "/data/to/your/parent/DESED_test/path/$script_name"
                return 0
            fi
            ;;
        "GTZAN")
            if [[ -f "/data/to/your/GTZAN/path/$script_name" ]]; then
                echo "/data/to/your/GTZAN/path/$script_name"
                return 0
            fi
            if [[ -f "/data/to/your/parent/GTZAN/path/$script_name" ]]; then
                echo "/data/to/your/parent/GTZAN/path/$script_name"
                return 0
            fi
            ;;
    esac
    
    return 1
}

# Run a single test
run_single_test() {
    local task=$1
    local ratio=$2
    local script_path=$3
    local test_name=""
    
    if (( $(echo "$ratio == 0.0" | bc -l 2>/dev/null || echo "0") )); then
        test_name="base"
        sparse_flag="false"
    else
        test_name="sparse_${ratio}"
        sparse_flag="true"
    fi
    
    print_info "Start test: $task, Ratio: $ratio ($test_name)"
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi
    # Set result directory
    export RESULTS_DIR="$RESULTS_DIR"
    
    # Build command
    local cmd="python $script_path"
    cmd="$cmd --sparse $sparse_flag"
    cmd="$cmd --pruned_layer $PRUNED_LAYER"
    if [[ "$sparse_flag" == "true" ]]; then
        cmd="$cmd --reduction_ratio $ratio"
    fi
    
    # Create log file
    local log_file="$RESULTS_DIR/logs/${task}_log_${test_name}.txt"
    mkdir -p "$(dirname "$log_file")"
    
    if [[ "$DRY_RUN" == true ]]; then
        print_info "[DRY RUN] Command: $cmd"
        print_info "[DRY RUN] Log file: $log_file"
        return 0
    fi
    
    print_info "Execute command: $cmd"
    print_info "Log file: $log_file"
    
    # Run test
    local start_time=$(date +%s)
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Test complete: $task - $test_name (Duration: ${duration}s)"
        
        # Extract accuracy info
        extract_accuracy_from_log "$log_file" "$task" "$test_name"
        return 0
    else
        print_error "Test failed: $task - $test_name"
        print_error "See detailed error in: $log_file"
        return 1
    fi
}

# Extract accuracy from log file
extract_accuracy_from_log() {
    local log_file=$1
    local task=$2
    local test_name=$3
    
    # Try to extract accuracy info of different formats
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
        print_info "  └─ $task - $test_name: $accuracy"
    fi
}

# Run all ratio tests for a task
run_task_tests() {
    local task=$1
    local script_path=$2
    local task_start_time=$(date +%s)
    local failed_ratios=0
    
    print_task "Start task: $task (Total ${#RATIOS[@]} ratios)"
    
    for i in "${!RATIOS[@]}"; do
        local ratio=${RATIOS[$i]}
        print_progress "Task $task - Progress: $((i+1))/${#RATIOS[@]} (Ratio: $ratio)"
        
        if ! run_single_test "$task" "$ratio" "$script_path"; then
            ((failed_ratios++))
            if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                print_error "Task $task failed at ratio $ratio, stopping subsequent tests for this task"
                break
            fi
        fi
        
        # Add small delay between ratio tests
        if [[ $i -lt $((${#RATIOS[@]}-1)) ]]; then
            sleep 2
        fi
    done
    
    local task_end_time=$(date +%s)
    local task_duration=$((task_end_time - task_start_time))
    
    if [[ $failed_ratios -eq 0 ]]; then
        print_success "Task $task completed, all ${#RATIOS[@]} ratio tests succeeded (Duration: ${task_duration}s)"
        return 0
    else
        print_warning "Task $task completed, $failed_ratios ratio tests failed (Duration: ${task_duration}s)"
        return 1
    fi
}

# Run tasks in parallel
run_tasks_parallel() {
    local pids=()
    local task_results=()
    local running_tasks=0
    local completed_tasks=0
    local failed_tasks=0
    
    print_info "Start parallel execution of tasks (Max parallel: $PARALLEL_TASKS)"
    
    for task in "${TASK_LIST[@]}"; do
        # Wait until there is an available parallel slot
        while [[ $running_tasks -ge $PARALLEL_TASKS ]]; do
            check_completed_tasks pids task_results running_tasks completed_tasks failed_tasks
            sleep 1
        done
        
        # Check script file
        script_path=$(check_script_exists "$task")
        if [[ $? -ne 0 ]]; then
            print_error "Cannot find script file for $task, skipping this task"
            ((failed_tasks++))
            continue
        fi
        
        # Start new task
        print_info "Start task: $task"
        (run_task_tests "$task" "$script_path") &
        local pid=$!
        pids+=("$pid:$task")
        ((running_tasks++))
        
        # Task interval
        if [[ $TASK_INTERVAL -gt 0 ]]; then
            sleep $TASK_INTERVAL
        fi
    done
    
    # Wait for all tasks to complete
    print_info "Waiting for all tasks to complete..."
    while [[ $running_tasks -gt 0 ]]; do
        check_completed_tasks pids task_results running_tasks completed_tasks failed_tasks
        sleep 1
    done
    
    # Show final result
    print_info "All tasks execution completed"
    print_info "Success tasks: $completed_tasks"
    print_info "Failed tasks: $failed_tasks"
    
    return $failed_tasks
}

# Check completed tasks
check_completed_tasks() {
    local -n pids_ref=$1
    local -n results_ref=$2
    local -n running_ref=$3
    local -n completed_ref=$4
    local -n failed_ref=$5
    
    local new_pids=()
    
    for pid_task in "${pids_ref[@]}"; do
        IFS=':' read -ra pid_info <<< "$pid_task"
        local pid=${pid_info[0]}
        local task=${pid_info[1]}
        
        if ! kill -0 "$pid" 2>/dev/null; then
            # Task completed
            wait "$pid"
            local exit_code=$?
            
            if [[ $exit_code -eq 0 ]]; then
                print_success "Task completed: $task"
                ((completed_ref++))
            else
                print_error "Task failed: $task (exit code: $exit_code)"
                ((failed_ref++))
            fi
            
            ((running_ref--))
            results_ref+=("$task:$exit_code")
        else
            # Task still running
            new_pids+=("$pid_task")
        fi
    done
    
    pids_ref=("${new_pids[@]}")
}

# Run tasks serially
run_tasks_serial() {
    local failed_tasks=0
    local completed_tasks=0
    
    print_info "Start serial execution of tasks"
    
    for i in "${!TASK_LIST[@]}"; do
        local task=${TASK_LIST[$i]}
        print_progress "Overall progress: $((i+1))/${#TASK_LIST[@]} - Current task: $task"
        
        # Check script file
        script_path=$(check_script_exists "$task")
        if [[ $? -ne 0 ]]; then
            print_error "Cannot find script file for $task, skipping this task"
            ((failed_tasks++))
            continue
        fi
        
        # Execute task
        if run_task_tests "$task" "$script_path"; then
            ((completed_tasks++))
        else
            ((failed_tasks++))
            if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                print_error "Task $task failed, stopping subsequent tasks"
                break
            fi
        fi
        
        # Task interval
        if [[ $i -lt $((${#TASK_LIST[@]}-1)) && $TASK_INTERVAL -gt 0 ]]; then
            print_info "Wait ${TASK_INTERVAL}s before next task..."
            sleep $TASK_INTERVAL
        fi
    done
    
    print_info "Serial execution completed"
    print_info "Success tasks: $completed_tasks"
    print_info "Failed tasks: $failed_tasks"
    
    return $failed_tasks
}

# Generate multi-task summary report
generate_multi_task_summary_report() {
    local summary_file="$RESULTS_DIR/logs/multi_task_batch_test_summary.txt"
    local csv_file="$RESULTS_DIR/logs/multi_task_results_summary.csv"
    
    print_info "Generate multi-task summary report: $summary_file"
    
    {
        echo "DART Multi-task Batch Test Summary Report"
        echo "=========================="
        echo "Test tasks: ${TASK_LIST[*]}"
        echo "GPU ID: $GPU_ID"
        echo "Sample limit: $SAMPLE_LIMIT"
        echo "Pruned layers: $PRUNED_LAYER"
        echo "Sparsity ratios: ${RATIOS[*]}"
        echo "Parallel tasks: $PARALLEL_TASKS"
        echo "Task interval: ${TASK_INTERVAL}s"
        echo "Test time: $(date)"
        echo ""
        echo "Test results:"
        echo "--------"
        
        for task in "${TASK_LIST[@]}"; do
            echo ""
            echo "Task: $task"
            echo "============"
            
            for ratio in "${RATIOS[@]}"; do
                if (( $(echo "$ratio == 0.0" | bc -l 2>/dev/null || echo "0") )); then
                    test_name="base"
                else
                    test_name="sparse_${ratio}"
                fi
                
                # Find log file
                local log_file="$RESULTS_DIR/logs/${task}_log_${test_name}.txt"
                
                if [[ -f "$log_file" ]]; then
                    echo "Ratio $ratio ($test_name): Completed"
                    
                    # Extract accuracy
                    local accuracy=$(grep -o "Overall Accuracy: [0-9.]*%" "$log_file" | tail -1)
                    if [[ -z "$accuracy" ]]; then
                        accuracy=$(grep -o "Total Accuracy: [0-9.]*" "$log_file" | tail -1)
                    fi
                    if [[ -z "$accuracy" ]]; then
                        accuracy=$(grep -o "Accuracy: [0-9.]*" "$log_file" | tail -1)
                    fi
                    
                    if [[ -n "$accuracy" ]]; then
                        echo "  └─ $accuracy"
                    fi
                else
                    echo "Ratio $ratio ($test_name): Not completed or failed"
                fi
            done
        done
        
        echo ""
        echo "File locations:"
        echo "--------"
        find "$RESULTS_DIR" -name "*.json" -o -name "*.txt" | sort
        
    } > "$summary_file"
    
    # Generate CSV summary
    {
        echo "Task,Ratio,Test_Name,Status,Accuracy,Log_File"
        
        for task in "${TASK_LIST[@]}"; do
            for ratio in "${RATIOS[@]}"; do
                if (( $(echo "$ratio == 0.0" | bc -l 2>/dev/null || echo "0") )); then
                    test_name="base"
                else
                    test_name="sparse_${ratio}"
                fi
                
                local log_file="$RESULTS_DIR/logs/${task}_log_${test_name}.txt"
                local status="Failed"
                local accuracy="N/A"
                
                if [[ -f "$log_file" ]]; then
                    status="Completed"
                    accuracy=$(grep -o "Overall Accuracy: [0-9.]*%" "$log_file" | tail -1)
                    if [[ -z "$accuracy" ]]; then
                        accuracy=$(grep -o "Total Accuracy: [0-9.]*" "$log_file" | tail -1)
                    fi
                    if [[ -z "$accuracy" ]]; then
                        accuracy=$(grep -o "Accuracy: [0-9.]*" "$log_file" | tail -1)
                    fi
                    if [[ -z "$accuracy" ]]; then
                        accuracy="N/A"
                    fi
                fi
                
                echo "$task,$ratio,$test_name,$status,$accuracy,$log_file"
            done
        done
        
    } > "$csv_file"
    
    print_success "Summary report generated: $summary_file"
    print_success "CSV summary generated: $csv_file"
}

# Preview execution plan
preview_execution_plan() {
    print_info "Execution plan preview:"
    print_info "============="
    print_info "Tasks to be tested (${#TASK_LIST[@]}): ${TASK_LIST[*]}"
    print_info "Sparsity ratios (${#RATIOS[@]}): ${RATIOS[*]}"
    print_info "Total test count: $((${#TASK_LIST[@]} * ${#RATIOS[@]}))"
    print_info "Parallel tasks: $PARALLEL_TASKS"
    print_info "Task interval: ${TASK_INTERVAL}s"
    print_info "Estimated total time: Depends on specific task complexity"
    
    if [[ "$DRY_RUN" == true ]]; then
        print_warning "This is preview mode, tests will not actually be executed"
    fi
    
    echo ""
    print_info "Detailed test list:"
    for task in "${TASK_LIST[@]}"; do
        echo "  Task: $task"
        for ratio in "${RATIOS[@]}"; do
            if (( $(echo "$ratio == 0.0" | bc -l 2>/dev/null || echo "0") )); then
                echo "    - Base test (ratio=0.0)"
            else
                echo "    - Sparse test (ratio=$ratio)"
            fi
        done
    done
}

# Main function
main() {
    print_info "DART multi-task batch test script started"
    
    # Parse arguments
    parse_args "$@"
    
    # Show config
    print_info "Test config:"
    print_info "  Task list (${#TASK_LIST[@]}): ${TASK_LIST[*]}"
    print_info "  GPU ID: $GPU_ID"
    print_info "  Sample limit: $SAMPLE_LIMIT"
    print_info "  Pruned layers: $PRUNED_LAYER"
    print_info "  Sparsity ratios (${#RATIOS[@]}): ${RATIOS[*]}"
    print_info "  Parallel tasks: $PARALLEL_TASKS"
    print_info "  Task interval: ${TASK_INTERVAL}s"
    print_info "  Output directory: $RESULTS_DIR"
    print_info "  Skip base test: $SKIP_BASE"
    print_info "  Continue on error: $CONTINUE_ON_ERROR"
    print_info "  Preview mode: $DRY_RUN"
    
    # Preview execution plan
    preview_execution_plan
    
    # Check Python and dependencies
    if ! command -v python &> /dev/null; then
        print_error "Python not found, please ensure Python is installed and in PATH"
        exit 1
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_DIR/logs"
    
    # Record start time
    local total_start_time=$(date +%s)
    
    # Execute tests
    local failed_tasks=0
    if [[ $PARALLEL_TASKS -gt 1 ]]; then
        run_tasks_parallel
        failed_tasks=$?
    else
        run_tasks_serial
        failed_tasks=$?
    fi
    
    # Calculate total duration
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    # Generate summary report
    if [[ "$DRY_RUN" != true ]]; then
        generate_multi_task_summary_report
    fi
    
    # Show final result
    print_info "Multi-task batch test completed!"
    print_info "Total duration: ${total_duration}s"
    print_info "Success tasks: $((${#TASK_LIST[@]} - failed_tasks))"
    if [[ $failed_tasks -eq 0 ]]; then
        print_success "All tasks completed successfully"
    else
        print_warning "$failed_tasks tasks failed"
    fi
    
    if [[ "$DRY_RUN" != true ]]; then
        print_info "Logs saved in: $RESULTS_DIR/logs"
        print_info "Result files saved in: $RESULTS_DIR (by each test script's default directory structure)"
    fi
    
    return $failed_tasks
}

# Check if bc command is available (for float comparisons)
if ! command -v bc &> /dev/null; then
    print_warning "bc command not found, will use simplified float handling"
    # Define simplified float comparison function
    bc() {
        if [[ "$1" =~ ^.*==.*0\.0.*$ ]]; then
            local num=$(echo "$1" | sed 's/.*== *\([0-9.]*\).*/\1/')
            if [[ "$num" == "0.0" || "$num" == "0" ]]; then
                echo "1"
            else
                echo "0"
            fi
        elif [[ "$1" =~ ^.*!=.*0\.0.*$ ]]; then
            local num=$(echo "$1" | sed 's/.*!= *\([0-9.]*\).*/\1/')
            if [[ "$num" == "0.0" || "$num" == "0" ]]; then
                echo "0"
            else
                echo "1"
            fi
        else
            echo "0"
        fi
    }
fi

# Run main function
main "$@"