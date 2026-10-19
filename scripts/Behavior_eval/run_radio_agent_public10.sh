#!/usr/bin/env bash

# Sequential orchestration only: each episode uses the unchanged Agent runner.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
AGENT_PYTHON="${EMERGE_AGENT_PYTHON:-/home/embodied/miniconda3/envs/phyagentos/bin/python}"
AGENT_MODEL="${EMERGE_AGENT_MODEL:-gpt-6.1-sol}"
OUTPUT_ROOT="${EMERGE_RADIO_OUTPUT_DIR:-${WORKSPACE_ROOT}/outputs/Emerge/turning_on_radio/public10}"
VIEW_MODE="${1:---no-headless}"

if [[ $# -gt 1 || ( "${VIEW_MODE}" != "--headless" && "${VIEW_MODE}" != "--no-headless" ) ]]; then
    echo "用法: bash $0 [--headless | --no-headless]" >&2
    exit 2
fi
if [[ ! -x "${AGENT_PYTHON}" ]] || ! command -v jq >/dev/null 2>&1; then
    echo "需要可执行的 EMERGE Python 和 jq: ${AGENT_PYTHON}" >&2
    exit 1
fi

INTERRUPT_STATUS=0
trap 'INTERRUPT_STATUS=130' INT
trap 'INTERRUPT_STATUS=143' TERM
INCOMPLETE=()
RESULTS=()

complete_result() {
    [[ -s "$1" && -s "$2" ]] || return 1
    # A clean official failure is a completed experiment too; never rerun only
    # unsuccessful episodes. Check identity and completion, not Agent prose.
    jq -e -s --argjson instance "$3" --argjson index "$4" '
        length == 2
        and .[0].task == "turning_on_radio" and .[0].instance_id == $instance
        and .[0].rollout_id == 0 and (.[0].success | type) == "boolean"
        and (.[0].steps | type) == "number" and .[0].steps >= 0
        and .[1].task == "turning_on_radio" and .[1].instance_index == $index
        and .[1].episode_terminated == true and .[1].infrastructure_error == null
        and .[1].evaluator_returncode == 0 and .[1].exit_code == 0
        and .[1].success == .[0].success
    ' "$1" "$2" >/dev/null 2>&1
}

echo "EMERGE turning_on_radio: public_test index 0-9 / instance 301-310"
echo "模型: ${AGENT_MODEL}; 每例墙钟上限 3600 秒; 官方默认仿真步数上限"
echo "输出目录: ${OUTPUT_ROOT}; 保留视频; ${VIEW_MODE}"
echo "开始后请保持代码、profile、checkpoint 和服务配置不变。"

for INDEX in 0 1 2 3 4 5 6 7 8 9; do
    if [[ ${INTERRUPT_STATUS} -ne 0 ]]; then
        exit "${INTERRUPT_STATUS}"
    fi
    INSTANCE_ID=$((301 + INDEX))
    EPISODE_DIR="${OUTPUT_ROOT}/instance_${INSTANCE_ID}"
    RESULT="${EPISODE_DIR}/json/turning_on_radio_${INSTANCE_ID}_0.json"
    SUMMARY="${EPISODE_DIR}/agent_episode_result.json"
    if complete_result "${RESULT}" "${SUMMARY}" "${INSTANCE_ID}" "${INDEX}"; then
        echo "[${INSTANCE_ID}] 已有完整官方结果，跳过。"
        RESULTS+=("${RESULT}")
        continue
    fi
    if [[ -e "${EPISODE_DIR}" ]]; then
        echo "[${INSTANCE_ID}] 目录已有未完成/无效结果，保留且不覆盖: ${EPISODE_DIR}" >&2
        echo "请检查并移走该实例目录后重跑；本次继续其它实例。" >&2
        INCOMPLETE+=("${INSTANCE_ID}")
        continue
    fi

    echo "[index ${INDEX}, instance ${INSTANCE_ID}] 开始"
    if [[ ${INTERRUPT_STATUS} -ne 0 ]]; then
        exit "${INTERRUPT_STATUS}"
    fi
    (
        cd "${REPO_ROOT}" || exit 1
        exec "${AGENT_PYTHON}" "${SCRIPT_DIR}/eval_behavior_agent.py" \
            --task-name turning_on_radio \
            --instance-index "${INDEX}" \
            --agent-model "${AGENT_MODEL}" \
            --episode-timeout 3600 \
            "${VIEW_MODE}" \
            --write-video \
            --show-agent-events \
            --output-dir "${EPISODE_DIR}"
    )
    STATUS=$?
    if [[ ${INTERRUPT_STATUS} -ne 0 ]]; then
        STATUS=${INTERRUPT_STATUS}
    fi
    if [[ ${STATUS} -eq 130 || ${STATUS} -eq 143 ]]; then
        echo "收到中断；停止整个 10 实例实验，不再启动下一例。" >&2
        exit "${STATUS}"
    fi
    if [[ ${STATUS} -ne 0 ]] || ! complete_result "${RESULT}" "${SUMMARY}" "${INSTANCE_ID}" "${INDEX}"; then
        INCOMPLETE+=("${INSTANCE_ID}")
        echo "[${INSTANCE_ID}] 未形成完整评测，请检查该实例 logs/。" >&2
        continue
    fi
    RESULTS+=("${RESULT}")
    echo "[${INSTANCE_ID}] 完成，success=$(jq -r '.success' "${RESULT}")"
done

echo "完整评测: ${#RESULTS[@]}/10"
if [[ ${#RESULTS[@]} -gt 0 ]]; then
    jq -s '{completed: length, successes: ([.[] | select(.success)] | length),
            mean_steps: (map(.steps) | add / length),
            mean_q_score: (map(.q_score.final) | add / length)}' "${RESULTS[@]}"
fi
if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
    echo "未完整完成的 instance: ${INCOMPLETE[*]}（不是任务失败率统计）" >&2
    exit 1
fi
