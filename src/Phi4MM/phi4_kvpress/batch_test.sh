#!/bin/bash

# KV Press批量测试脚本
# 用于测试不同压缩比例和压缩类型的KV Press性能

# 设置环境变量 (保持与dart_batch_test.sh一致)
export HUGGINGFACE_HUB_CACHE='/data/hepeize05/Audio_Longbench/Code/Model'
export HF_HOME='/data/hepeize05/Audio_Longbench/Code/Model'
export XDG_CACHE_HOME='/data/hepeize05/Audio_Longbench/Code/Model'
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
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

# 默认配置
DEFAULT_GPU_ID=0
DEFAULT_SAMPLE_LIMIT=0  # 0表示不限制
DEFAULT_RESULTS_DIR="./KVPress_Batch_Results"

# 支持的任务列表
SUPPORTED_TASKS=("HAD" "race" "SLUE" "TAU" "VESUS" "Vox" "Vox_age" "LibriSpeech" "DESED" "GTZAN")

# 默认压缩比例列表
DEFAULT_COMPRESSION_RATIOS=(0.1 0.3 0.5 0.7 0.8 0.9)

# 默认压缩类型列表
DEFAULT_PRESS_TYPES=("knorm"  "random" "tova" "snap")

# 使用说明
show_usage() {
    echo "KV Press批量测试脚本"
    echo ""
    echo "用法: $0 [选项] <任务名>"
    echo ""
    echo "支持的任务:"
    for task in "${SUPPORTED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "选项:"
    echo "  -g, --gpu-id <id>           GPU ID (默认: $DEFAULT_GPU_ID)"
    echo "  -s, --sample-limit <num>    样本数量限制 (默认: $DEFAULT_SAMPLE_LIMIT, 0表示不限制)"
    echo "  -r, --ratios <ratios>       压缩比例列表，用逗号分隔 (默认: ${DEFAULT_COMPRESSION_RATIOS[*]})"
    echo "  -t, --types <types>         压缩类型列表，用逗号分隔 (默认: ${DEFAULT_PRESS_TYPES[*]})"
    echo "  -c, --single-ratio <ratio>  单个压缩比例测试"
    echo "  -p, --single-type <type>    单个压缩类型测试"
    echo "  -o, --output-dir <dir>      结果输出目录 (默认: $DEFAULT_RESULTS_DIR)"
    echo "  -h, --help                  显示此帮助信息"
    echo ""
    echo "支持的压缩类型:"
    echo "  - knorm       : 基于Key-norm的注意力压缩 (默认)"
    echo "  - expected    : 基于RoPE的期望注意力压缩"
    echo "  - observed    : 基于观察到的注意力分数"
    echo "  - random      : 随机丢弃tokens"
    echo "  - tova        : 基于时间顺序的注意力值分析"
    echo "  - snap        : 适用于LoRA微调模型"
    echo "  - streaming   : 适用于长文本流式处理"
    echo ""
    echo "示例:"
    echo "  $0 HAD                                    # 测试HAD任务，所有比例和类型组合"
    echo "  $0 -g 1 -s 100 race                      # 在GPU1上测试race任务，限制100个样本"
    echo "  $0 -r 0.3,0.5,0.7 -t knorm,expected TAU  # 测试TAU任务，指定特定比例和类型"
    echo "  $0 -c 0.5 -p knorm SLUE                  # 测试SLUE任务，单个比例和类型"
    echo "  $0 -o ./my_results DESED                 # 测试DESED任务，自定义输出目录"
    echo "  $0 VESUS                                  # 测试VESUS情感识别任务"
    echo "  $0 GTZAN                                  # 测试GTZAN音乐风格分类任务"
}

# 解析命令行参数
parse_args() {
    GPU_ID=$DEFAULT_GPU_ID
    SAMPLE_LIMIT=$DEFAULT_SAMPLE_LIMIT
    RESULTS_DIR=$DEFAULT_RESULTS_DIR
    TASK=""
    SINGLE_RATIO=""
    SINGLE_TYPE=""
    
    # 将默认比例数组转换为字符串
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
            -o|--output-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                print_error "未知选项: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$TASK" ]]; then
                    TASK="$1"
                else
                    print_error "只能指定一个任务"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 检查任务是否指定
    if [[ -z "$TASK" ]]; then
        print_error "必须指定一个任务"
        show_usage
        exit 1
    fi
    
    # 检查任务是否支持
    if [[ ! " ${SUPPORTED_TASKS[@]} " =~ " ${TASK} " ]]; then
        print_error "不支持的任务: $TASK"
        print_info "支持的任务: ${SUPPORTED_TASKS[*]}"
        exit 1
    fi
    
    # 处理单个比例和类型
    if [[ -n "$SINGLE_RATIO" ]]; then
        RATIOS_STRING="$SINGLE_RATIO"
    fi
    
    if [[ -n "$SINGLE_TYPE" ]]; then
        TYPES_STRING="$SINGLE_TYPE"
    fi
    
    # 解析比例和类型字符串为数组
    IFS=',' read -ra COMPRESSION_RATIOS <<< "$RATIOS_STRING"
    IFS=',' read -ra PRESS_TYPES <<< "$TYPES_STRING"
    IFS=' '
    
    # 验证压缩比例
    for ratio in "${COMPRESSION_RATIOS[@]}"; do
        if ! [[ "$ratio" =~ ^0\.[0-9]+$|^1\.0$|^0$ ]]; then
            print_error "无效的压缩比例: $ratio (应该在0.0-1.0之间)"
            exit 1
        fi
    done
    
    # 验证压缩类型
    valid_types=("knorm" "expected" "observed" "random" "tova" "snap" "streaming")
    for press_type in "${PRESS_TYPES[@]}"; do
        if [[ ! " ${valid_types[@]} " =~ " ${press_type} " ]]; then
            print_error "无效的压缩类型: $press_type"
            print_info "支持的类型: ${valid_types[*]}"
            exit 1
        fi
    done
}

# 检查脚本文件是否存在
check_script_exists() {
    local task=$1
    local script_name=""
    
    case $task in
        "HAD")
            script_name="HAD_kvpress.py"
            ;;
        "race")
            script_name="race_kvpress.py"
            ;;
        "SLUE")
            script_name="SLUE_kvpress.py"
            ;;
        "TAU")
            script_name="TAU_kvpress.py"
            ;;
        "VESUS")
            script_name="VESUS_kvpress.py"
            ;;
        "Vox")
            script_name="Vox_kvpress.py"
            ;;
        "Vox_age")
            script_name="Vox_age_kvpress.py"
            ;;
        "LibriSpeech")
            script_name="LibriSpeech_kvpress.py"
            ;;
        "DESED")
            script_name="DESED_kvpress.py"
            ;;
        "GTZAN")
            script_name="GTZAN_kvpress.py"
            ;;
    esac
    
    # 检查当前目录
    if [[ -f "./$script_name" ]]; then
        echo "./$script_name"
        return 0
    fi
    
    # 检查任务子目录
    if [[ -f "./$task/$script_name" ]]; then
        echo "./$task/$script_name"
        return 0
    fi
    
    # 检查上级目录的任务子目录
    if [[ -f "../$task/$script_name" ]]; then
        echo "../$task/$script_name"
        return 0
    fi
    
    # 检查KV Press目录
    if [[ -f "./kv_press/$script_name" ]]; then
        echo "./kv_press/$script_name"
        return 0
    fi
    
    if [[ -f "../kv_press/$script_name" ]]; then
        echo "../kv_press/$script_name"
        return 0
    fi
    
    # 检查特殊目录
    case $task in
        "LibriSpeech")
            local paths=(
                "./LibriSpeech-Long/$script_name"
                "../LibriSpeech-Long/$script_name"
                "./Librispeech/$script_name"
                "../Librispeech/$script_name"
            )
            ;;
        "DESED")
            local paths=(
                "./DESED_test/$script_name"
                "../DESED_test/$script_name"
                "./DESED/$script_name"
                "../DESED/$script_name"
            )
            ;;
        "GTZAN")
            local paths=(
                "./GTZAN/$script_name"
                "../GTZAN/$script_name"
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

# 运行单个测试
run_single_test() {
    local ratio=$1
    local press_type=$2
    local script_path=$3
    local test_name="${press_type}_ratio_${ratio}"
    
    print_info "开始测试: $TASK, 压缩类型: $press_type, 压缩比例: $ratio"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export COMPRESSION_RATIO=$ratio
    export PRESS_TYPE=$press_type
    
    if [[ $SAMPLE_LIMIT -gt 0 ]]; then
        export SAMPLE_LIMIT=$SAMPLE_LIMIT
    else
        unset SAMPLE_LIMIT
    fi
    
    # 设置结果目录
    export RESULTS_DIR="$RESULTS_DIR"
    
    # 构建命令
    local cmd="python $script_path"
    
    # 创建日志文件
    local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"
    mkdir -p "$(dirname "$log_file")"
    
    print_info "执行命令: $cmd"
    print_info "环境变量: COMPRESSION_RATIO=$ratio, PRESS_TYPE=$press_type"
    print_info "日志文件: $log_file"
    
    # 执行测试
    local start_time=$(date +%s)
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "测试完成: $test_name (耗时: ${duration}秒)"
        
        # 提取准确率信息（如果可能）
        extract_accuracy_from_log "$log_file" "$test_name"
    else
        print_error "测试失败: $test_name"
        print_error "详细错误信息请查看: $log_file"
        return 1
    fi
}

# 从日志文件提取准确率信息
extract_accuracy_from_log() {
    local log_file=$1
    local test_name=$2
    
    # 尝试提取不同格式的准确率信息
    local accuracy=""
    
    # 匹配 "总体准确率: XX.XX%" 或 "总体准确率: X.XXXX"
    accuracy=$(grep -o "总体准确率: [0-9.]*[%]*" "$log_file" | tail -1)
    if [[ -z "$accuracy" ]]; then
        # 匹配 "Accuracy: X.XXXX"
        accuracy=$(grep -o "Accuracy: [0-9.]*" "$log_file" | tail -1)
    fi
    if [[ -z "$accuracy" ]]; then
        # 匹配 "F1 Score: X.XXXX"
        accuracy=$(grep -o "F1 Score: [0-9.]*" "$log_file" | tail -1)
        if [[ -n "$accuracy" ]]; then
            accuracy="F1 $accuracy"
        fi
    fi
    if [[ -z "$accuracy" ]]; then
        # 匹配 "平均吞吐量: XX.XX tokens/秒"
        accuracy=$(grep -o "平均吞吐量: [0-9.]* tokens/秒" "$log_file" | tail -1)
        if [[ -n "$accuracy" ]]; then
            accuracy="吞吐量 $accuracy"
        fi
    fi
    
    if [[ -n "$accuracy" ]]; then
        print_info "  └─ $test_name: $accuracy"
    fi
}

# 生成汇总报告
generate_summary_report() {
    local summary_file="$RESULTS_DIR/logs/${TASK}_kvpress_batch_summary.txt"
    
    print_info "生成汇总报告: $summary_file"
    
    {
        echo "KV Press批量测试汇总报告"
        echo "========================"
        echo "任务: $TASK"
        echo "GPU ID: $GPU_ID"
        echo "样本限制: $SAMPLE_LIMIT"
        echo "测试时间: $(date)"
        echo ""
        echo "测试配置:"
        echo "--------"
        echo "压缩比例: ${COMPRESSION_RATIOS[*]}"
        echo "压缩类型: ${PRESS_TYPES[*]}"
        echo ""
        echo "测试结果:"
        echo "--------"
        
        # 批量测试结果
        for press_type in "${PRESS_TYPES[@]}"; do
            echo "压缩类型: $press_type"
            for ratio in "${COMPRESSION_RATIOS[@]}"; do
                local test_name="${press_type}_ratio_${ratio}"
                local log_file="$RESULTS_DIR/logs/${TASK}_log_${test_name}.txt"
                
                if [[ -f "$log_file" ]]; then
                    echo "  比例 $ratio: 测试完成"
                    local accuracy=$(extract_accuracy_from_log "$log_file" "$test_name")
                else
                    echo "  比例 $ratio: 测试失败或未完成"
                fi
            done
            echo ""
        done
        
        echo ""
        echo "文件位置:"
        echo "--------"
        find "$RESULTS_DIR" -name "*.json" -o -name "*.txt" | sort
        
    } > "$summary_file"
    
    print_success "汇总报告已生成: $summary_file"
}

# 主函数
main() {
    print_info "KV Press批量测试脚本启动"
    
    # 解析参数
    parse_args "$@"
    
    # 显示配置
    print_info "测试配置:"
    print_info "  任务: $TASK"
    print_info "  GPU ID: $GPU_ID"
    print_info "  样本限制: $SAMPLE_LIMIT"
    print_info "  压缩比例: ${COMPRESSION_RATIOS[*]}"
    print_info "  压缩类型: ${PRESS_TYPES[*]}"
    print_info "  输出目录: $RESULTS_DIR"
    
    # 检查脚本文件
    print_info "查找脚本文件..."
    script_path=$(check_script_exists "$TASK")
    if [[ $? -ne 0 ]]; then
        print_error "找不到任务 $TASK 的KV Press脚本文件"
        print_error "请确保脚本文件存在于当前目录或任务子目录中"
        print_error "预期文件名: ${TASK}_kvpress.py"
        exit 1
    fi
    print_success "找到脚本文件: $script_path"
    
    # 检查Python和依赖
    if ! command -v python &> /dev/null; then
        print_error "Python未找到，请确保Python已安装并在PATH中"
        exit 1
    fi
    
    # 创建结果目录
    mkdir -p "$RESULTS_DIR/logs"
    
    # 记录开始时间
    local total_start_time=$(date +%s)
    
    # 运行测试
    local total_tests=0
    local failed_tests=0
    
    # 测试所有压缩比例和类型的组合
    total_tests=$((${#COMPRESSION_RATIOS[@]} * ${#PRESS_TYPES[@]}))
    print_info "开始批量测试，共 $total_tests 个组合"
    
    local test_count=0
    for press_type in "${PRESS_TYPES[@]}"; do
        for ratio in "${COMPRESSION_RATIOS[@]}"; do
            ((test_count++))
            print_info "进度: $test_count/$total_tests"
            
            if ! run_single_test "$ratio" "$press_type" "$script_path"; then
                ((failed_tests++))
            fi
            
            # 在测试之间添加小延迟，避免GPU过热
            if [[ $test_count -lt $total_tests ]]; then
                print_info "等待5秒后继续下一个测试..."
                sleep 5
            fi
        done
    done
    
    # 计算总耗时
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    # 生成汇总报告
    generate_summary_report
    
    # 显示最终结果
    print_info "KV Press批量测试完成！"
    print_info "总测试数: $total_tests"
    print_info "总耗时: ${total_duration}秒"
    if [[ $failed_tests -eq 0 ]]; then
        print_success "所有测试均成功完成"
    else
        print_warning "有 $failed_tests 个测试失败"
    fi
    print_info "日志保存在: $RESULTS_DIR/logs"
    print_info "结果文件保存在: $RESULTS_DIR"
}

# 检查bc命令是否可用（用于浮点数比较）
if ! command -v bc &> /dev/null; then
    print_warning "bc命令未找到，将使用简化的浮点数处理"
fi

# 执行主函数
main "$@"
