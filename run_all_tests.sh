#!/bin/bash

# =================================================================
#   在 TMUX 中顺序执行并带休眠的自动化测试脚本
# =================================================================
#
# 功能:
#   1. (TMUX 集成) 自动创建或附加到一个 tmux 会话，确保测试在后台持续运行。
#   2. (顺序执行) 严格按照定义的命令列表，逐一执行测试。
#   3. (自动休眠) 每条命令执行后休眠指定时间，让硬件冷却。
#   4. (错误处理) 单个命令失败不会中断整个测试流程。
#   5. (日志记录) 所有操作的输出都会实时显示并记录到日志文件。
#
# 如何使用:
#   1. 将此脚本与 stress_test.py 放在同一目录。
#   2. 根据需要，修改下面的 "配置" 部分 (如 tmux 会话名、休眠时间等)。
#   3. 在终端中给予执行权限: chmod +x run_tests_in_tmux.sh
#   4. 直接运行脚本 (不再需要 sudo):
#      ./run_tests_in_tmux.sh
#   5. 脚本会自动启动 tmux 并在其中运行测试。您可以随时关闭终端。
#   6. 若要重新查看进度，请在新终端中运行: tmux attach -t rag_stress_test
#   7. 查看完整的日志文件: cat stress_test_final_log.txt
#
# =================================================================

# --- 配置 ---
# 脚本和日志文件
PYTHON_SCRIPT="stress_test.py"
OUTPUT_FILE="stress_test_log.txt"

# TMUX 配置
TMUX_SESSION_NAME="rag_stress_test"

# 休眠时间 (单位: 秒), 15分钟 = 900 秒
SLEEP_DURATION=10

# --- 您要执行的命令列表 ---
COMMANDS_TO_RUN=(
    # "python ${PYTHON_SCRIPT} --mode end2end --batch_sizes 1000 --repeats 1"
    # "python ${PYTHON_SCRIPT} --mode end2end --batch_sizes 1000 2000 3000 4000 5000 6000 7000 8000"
    # "python ${PYTHON_SCRIPT} --mode embedding  --batch_sizes 13000 14000 15000 16000 --repeats 50"
    # "python ${PYTHON_SCRIPT} --mode end2end --query_type image --batch_sizes 1000 2000"
    # "python ${PYTHON_SCRIPT} --mode end2end --query_type image --batch_sizes 3000 --repeats 10"
    # "python ${PYTHON_SCRIPT} --mode embedding --query_type image --batch_sizes 1000 2000"
    # "python ${PYTHON_SCRIPT} --mode embedding --query_type image --batch_sizes 3000 --repeats 10"

    "python test_memory.py --mode embedding --batch_sizes [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]  "
    "python test_memory.py --mode end2end --batch_sizes [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]  "
    "python test_memory.py --mode search --batch_sizes [1000,  3000,  5000, 10000, 20000, 30000,40000, 50000, 60000, 70000]"
    "python test_memory.py --mode end2end --query_type image --batch_sizes 1000 2000"
    "python test_memory.py --mode end2end --query_type image --batch_sizes 3000 "
    "python test_memory.py --mode embedding --query_type image --batch_sizes 1000 2000 "
    "python test_memory.py --mode embedding --query_type image --batch_sizes 3000 "


)


# --- 核心执行函数 (在 tmux 内部运行) ---
run_tests_inside_tmux() {
    # 初始化日志文件
    echo "自动化测试已在 TMUX 会话 '$TMUX_SESSION_NAME' 中启动。" > "$OUTPUT_FILE"
    echo "测试开始时间: $(date)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    total_commands=${#COMMANDS_TO_RUN[@]}

    for i in "${!COMMANDS_TO_RUN[@]}"; do
        cmd="${COMMANDS_TO_RUN[$i]}"
        current_index=$i

        # --- 执行单个命令 ---
        echo "======================================================================" | tee -a "$OUTPUT_FILE"
        echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] EXECUTING COMMAND $(($i + 1)) / $total_commands:" | tee -a "$OUTPUT_FILE"
        echo ">>> $cmd" | tee -a "$OUTPUT_FILE"
        echo "----------------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
        
        eval "$cmd" 2>&1 | tee -a "$OUTPUT_FILE"
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -ne 0 ]; then
            echo "----------------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
            echo "!!! WARNING: 上一条命令执行失败 (退出码: $exit_code)。" | tee -a "$OUTPUT_FILE"
        else
            echo "--- Command completed successfully (exit code: 0) ---" | tee -a "$OUTPUT_FILE"
        fi

        # --- 执行休眠 (如果不是最后一条命令) ---
        if [ $current_index -lt $(($total_commands - 1)) ]; then
            sleep_minutes=$((SLEEP_DURATION / 60))
            echo "" | tee -a "$OUTPUT_FILE"
            echo "----------------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
            echo ">>> 命令执行完毕，将休眠 ${sleep_minutes} 分钟..." | tee -a "$OUTPUT_FILE"
            echo ">>> 下一条命令将在 $(date -d "+${SLEEP_DURATION} seconds" +'%Y-%m-%d %H:%M:%S') 左右开始执行。" | tee -a "$OUTPUT_FILE"
            echo "----------------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
            sleep "$SLEEP_DURATION"
        fi
    done

    # --- 所有测试完成 ---
    echo "======================================================================" | tee -a "$OUTPUT_FILE"
    echo "所有测试已全部执行完毕！" | tee -a "$OUTPUT_FILE"
    echo "您可以随时使用 'exit' 命令或按 Ctrl+D 来关闭此 TMUX 会话。" | tee -a "$OUTPUT_FILE"
    echo "完整日志已保存在 '$OUTPUT_FILE' 文件中。" | tee -a "$OUTPUT_FILE"
    echo "======================================================================" | tee -a "$OUTPUT_FILE"
    
    exec bash
}

# --- 主程序入口 ---

# 检查是否已经在 tmux 中。如果是，则直接执行测试函数。
if [ "$1" = "--in-tmux" ]; then
    run_tests_inside_tmux
    exit 0
fi

# --- 如果不在 tmux 中，则执行以下设置和启动流程 ---

# 1. 检查依赖
command -v tmux &> /dev/null || { echo "错误: 未找到 tmux。请先安装 tmux (例如: sudo apt-get install tmux)。"; exit 1; }
[ -f "$PYTHON_SCRIPT" ] || { echo "错误: Python 脚本 '$PYTHON_SCRIPT' 未找到。"; exit 1; }

# 2. 启动或附加到 tmux 会话
echo "正在启动或附加到 TMUX 会话: '$TMUX_SESSION_NAME'..."
tmux new-session -As "$TMUX_SESSION_NAME" "bash '$0' --in-tmux"

echo ""
echo "-----------------------------------------------------------------"
echo "TMUX 会话已启动！测试正在后台运行。"
echo "您现在可以安全地关闭此终端窗口。"
echo ""
echo "要重新连接到会话以查看进度，请打开新终端并运行:"
echo "  tmux attach -t $TMUX_SESSION_NAME"
echo "-----------------------------------------------------------------"

exit 0
