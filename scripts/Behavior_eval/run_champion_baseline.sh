#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

TASK_NAME="${1:-}"
MODE="public_test"
HOST="127.0.0.1"
PORT="8011"
BEHAVIOR_ROOT="${REPO_ROOT}/third_party/behavior-1k"
BEHAVIOR_PYTHON="${WORKSPACE_ROOT}/envs/behavior/bin/python"
ROBOT_CONFIG="${REPO_ROOT}/dev/behavior_r1pro_openpi.yaml"
OUTPUT_DIR="${WORKSPACE_ROOT}/outputs/champion_baseline/${TASK_NAME}"

if [[ -z "${TASK_NAME}" ]]; then
    echo "用法: bash $0 <task_name>" >&2
    exit 2
fi

if [[ ! -x "${BEHAVIOR_PYTHON}" ]]; then
    echo "BEHAVIOR Python 不存在或不可执行: ${BEHAVIOR_PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${BEHAVIOR_ROOT}" ]]; then
    echo "BEHAVIOR-1K 仓库不存在: ${BEHAVIOR_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${ROBOT_CONFIG}" ]]; then
    echo "机器人配置不存在: ${ROBOT_CONFIG}" >&2
    exit 1
fi

if ! curl --noproxy '*' --silent --show-error --fail \
    "http://${HOST}:${PORT}/healthz" >/dev/null; then
    echo "冠军策略服务未就绪: http://${HOST}:${PORT}/healthz" >&2
    echo "请先启动冠军策略服务，再运行本脚本。" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

FAILED_INDICES=()
INTERRUPT_STATUS=0

# Bash waits for the foreground evaluator pipeline before running signal
# traps. Remember the requested stop so that, once the evaluator exits, this
# runner propagates the conventional status to the checkpoint-level runner
# instead of treating the interrupted instance as an ordinary failure and
# continuing with the next one.
trap 'INTERRUPT_STATUS=130' INT
trap 'INTERRUPT_STATUS=143' TERM

echo "任务: ${TASK_NAME}"
echo "实例: public_test index 0-9（官方 10 个实例）"
echo "输出目录: ${OUTPUT_DIR}"

for INDEX in $(seq 0 9); do
    INSTANCE_ID=$((301 + INDEX))
    RESULT_PATH="${OUTPUT_DIR}/json/${TASK_NAME}_${INSTANCE_ID}_0.json"
    LOG_PATH="${OUTPUT_DIR}/logs/${TASK_NAME}_${INSTANCE_ID}.log"

    if [[ -s "${RESULT_PATH}" ]]; then
        echo "[index ${INDEX}, instance ${INSTANCE_ID}] 已有官方结果，跳过。"
        continue
    fi

    echo
    echo "[index ${INDEX}, instance ${INSTANCE_ID}] 开始评测"

    (
        cd "${BEHAVIOR_ROOT}" || exit 1
        env \
            -u ISAAC_PATH \
            -u ISAACSIM_PATH \
            -u EXP_PATH \
            -u CARB_APP_PATH \
            OMNI_KIT_ACCEPT_EULA=YES \
            "${BEHAVIOR_PYTHON}" -m omnigibson.eval.eval \
                --task-name "${TASK_NAME}" \
                --mode "${MODE}" \
                --host "${HOST}" \
                --port "${PORT}" \
                --instance-indices "${INDEX}" \
                --num-envs 1 \
                --num-rollouts 1 \
                --replay-action-chunk-size 0 \
                --robot-config "${ROBOT_CONFIG}" \
                --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
                --output-dir "${OUTPUT_DIR}" \
                --video-fps 30 \
                --headless \
                --write-video
    ) 2>&1 | tee "${LOG_PATH}"

    STATUS=${PIPESTATUS[0]}
    if [[ ${INTERRUPT_STATUS} -ne 0 ]]; then
        STATUS=${INTERRUPT_STATUS}
    fi
    if [[ ${STATUS} -eq 130 || ${STATUS} -eq 143 ]]; then
        echo "[index ${INDEX}, instance ${INSTANCE_ID}] 收到中断信号；停止当前任务。" >&2
        exit "${STATUS}"
    fi
    if [[ ${STATUS} -ne 0 || ! -s "${RESULT_PATH}" ]]; then
        echo "[index ${INDEX}, instance ${INSTANCE_ID}] 失败，日志: ${LOG_PATH}" >&2
        FAILED_INDICES+=("${INDEX}")
    else
        echo "[index ${INDEX}, instance ${INSTANCE_ID}] 完成，结果: ${RESULT_PATH}"
    fi
done

echo
if [[ ${#FAILED_INDICES[@]} -eq 0 ]]; then
    echo "10 个 public_test 实例均已生成官方结果。"
    echo "结果目录: ${OUTPUT_DIR}"
    exit 0
fi

echo "以下 index 未成功生成官方结果: ${FAILED_INDICES[*]}" >&2
echo "重新执行同一脚本会跳过已完成实例，只重跑缺失部分。" >&2
exit 1
