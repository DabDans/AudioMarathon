#!/bin/bash

run_task() {
    local gpu_id=$1
    local task_name=$2
    
    echo "Executing task: $task_name (GPU $gpu_id)"
    
    bash batch_test.sh -g $gpu_id $task_name 
    
    if [ $? -eq 0 ]; then
        echo "Task $task_name executed successfully"
    else
        echo "Task $task_name execution failed"
        return 1
    fi
}

echo "Starting KV Press batch test tasks..."

run_task 4 SLUE
if [ $? -ne 0 ]; then
    echo "SLUE task failed, terminating script"
    exit 1
fi

run_task 4 race
if [ $? -ne 0 ]; then
    echo "race task failed, continuing to next task"
fi

echo "All tasks completed!"