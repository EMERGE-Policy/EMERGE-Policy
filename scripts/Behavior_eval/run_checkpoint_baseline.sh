#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

CHECKPOINT_NAME="${1:-}"
MODE="${2:-}"
CHECKPOINT_MAPPING="${OPENPI_CHAMPION_MAPPING:-${REPO_ROOT}/third_party/behavior-1k-solution/task_checkpoint_mapping.json}"
TASK_METADATA="${REPO_ROOT}/third_party/behavior-1k/datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv"
TASK_RUNNER="${SCRIPT_DIR}/run_champion_baseline.sh"

if [[ -z "${CHECKPOINT_NAME}" ]]; then
    echo "用法: bash $0 <checkpoint_name> [--list-only]" >&2
    echo "示例: bash $0 checkpoint_2" >&2
    exit 2
fi

if [[ -n "${MODE}" && "${MODE}" != "--list-only" ]]; then
    echo "未知参数: ${MODE}" >&2
    echo "用法: bash $0 <checkpoint_name> [--list-only]" >&2
    exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "缺少 jq，无法读取 checkpoint 映射。" >&2
    exit 1
fi

for REQUIRED in "${CHECKPOINT_MAPPING}" "${TASK_METADATA}" "${TASK_RUNNER}"; do
    if [[ ! -f "${REQUIRED}" ]]; then
        echo "缺少必需文件: ${REQUIRED}" >&2
        exit 1
    fi
done

if ! jq -e --arg checkpoint "${CHECKPOINT_NAME}" \
    '.checkpoints[$checkpoint].tasks | type == "array" and length > 0' \
    "${CHECKPOINT_MAPPING}" >/dev/null; then
    echo "映射中不存在 checkpoint 或任务列表为空: ${CHECKPOINT_NAME}" >&2
    echo "可用 checkpoint:" >&2
    jq -r '.checkpoints | keys[]' "${CHECKPOINT_MAPPING}" >&2
    exit 1
fi

mapfile -t TASK_IDS < <(
    jq -r --arg checkpoint "${CHECKPOINT_NAME}" \
        '.checkpoints[$checkpoint].tasks[]' "${CHECKPOINT_MAPPING}" | sort -n
)

TASK_NAMES=()
for TASK_ID in "${TASK_IDS[@]}"; do
    if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
        echo "checkpoint 映射包含非法 task ID: ${TASK_ID}" >&2
        exit 1
    fi

    TASK_NAME="$(
        awk -F, -v task_id="${TASK_ID}" \
            '$1 == task_id { gsub(/\r/, "", $2); print $2; exit }' \
            "${TASK_METADATA}"
    )"
    if [[ -z "${TASK_NAME}" ]]; then
        echo "官方任务元数据中找不到 task ID ${TASK_ID}。" >&2
        exit 1
    fi
    if ! [[ "${TASK_NAME}" =~ ^[A-Za-z0-9_]+$ ]]; then
        echo "task ID ${TASK_ID} 对应非法任务名: ${TASK_NAME}" >&2
        exit 1
    fi
    TASK_NAMES+=("${TASK_NAME}")
done

TASK_COUNT=${#TASK_NAMES[@]}
TOTAL_ROLLOUTS=$((TASK_COUNT * 10))

echo "Checkpoint: ${CHECKPOINT_NAME}"
echo "任务数量: ${TASK_COUNT}"
echo "每个任务: public_test index 0-9，共 10 个 rollout"
echo "计划总数: ${TOTAL_ROLLOUTS} 个 rollout"
echo "任务列表:"
for INDEX in "${!TASK_NAMES[@]}"; do
    printf '  [%s] task_id=%s  %s\n' \
        "$((INDEX + 1))" "${TASK_IDS[INDEX]}" "${TASK_NAMES[INDEX]}"
done

if [[ "${MODE}" == "--list-only" ]]; then
    exit 0
fi

FAILED_TASKS=()
COMPLETED_TASKS=0

for INDEX in "${!TASK_NAMES[@]}"; do
    TASK_ID="${TASK_IDS[INDEX]}"
    TASK_NAME="${TASK_NAMES[INDEX]}"

    echo
    echo "============================================================"
    echo "[$((INDEX + 1))/${TASK_COUNT}] ${TASK_NAME}（task_id=${TASK_ID}）"
    echo "============================================================"

    bash "${TASK_RUNNER}" "${TASK_NAME}"
    STATUS=$?

    if [[ ${STATUS} -eq 0 ]]; then
        COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
        continue
    fi
    if [[ ${STATUS} -eq 130 || ${STATUS} -eq 143 ]]; then
        echo "收到中断信号；停止 checkpoint baseline。" >&2
        exit "${STATUS}"
    fi

    FAILED_TASKS+=("${TASK_NAME}")
    echo "任务 ${TASK_NAME} 未完成；继续下一个任务。" >&2
done

echo
echo "Checkpoint baseline 结束。"
echo "完整任务: ${COMPLETED_TASKS}/${TASK_COUNT}"
echo "输出根目录: ${WORKSPACE_ROOT}/outputs/champion_baseline"

if [[ ${#FAILED_TASKS[@]} -eq 0 ]]; then
    echo "${TOTAL_ROLLOUTS} 个预期结果均已生成或此前已存在。"
    exit 0
fi

echo "未完整生成结果的任务: ${FAILED_TASKS[*]}" >&2
echo "重新执行同一命令会跳过已有 JSON，只重跑缺失实例。" >&2
exit 1
