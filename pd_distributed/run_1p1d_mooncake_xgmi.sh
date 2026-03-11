#!/bin/bash
# =============================================================================
# 1P1D Disaggregated Serving Demo — MooncakeConnector (xGMI)
# =============================================================================
# GPU0 = Prefill (kv_producer), GPU1 = Decode (kv_consumer)
# Mooncake TransferEngine 使用 xGMI 协议进行 GPU-to-GPU KV cache 直传
# 在 MI308/MI300X 上走 xGMI 互联，带宽 ~250 GB/s (vs TCP ~3 GB/s)
#
# 需要使用 amd_xgmi 分支编译的 Mooncake:
#   git fetch zhyajie amd_xgmi && git checkout amd_xgmi
#   cmake .. -DUSE_XGMI=ON -DUSE_TCP=ON -DUSE_HTTP=ON -DUSE_CUDA=OFF
# =============================================================================

set -euo pipefail

MODEL="/mnt/raid0/pretrained_model/Qwen/Qwen3-8B-FP8-dynamic"
SERVED_MODEL="qwen3-8b-fp8"

PREFILL_GPU=0
DECODE_GPU=1
PREFILL_PORT=8010
DECODE_PORT=8020
BOOTSTRAP_PORT=8998
PROXY_PORT=8000

TIMEOUT_SECONDS=600

# Mooncake source-compiled .so 路径
MOONCAKE_LIB="/home/yajizhan/qwen_code/Mooncake/mooncake-wheel/mooncake"
export LD_LIBRARY_PATH="${MOONCAKE_LIB}:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_mooncake_xgmi"
mkdir -p "${LOG_DIR}"

PIDS=()

cleanup() {
    echo ""
    echo "[cleanup] Stopping all background processes..."
    for pid in "${PIDS[@]}"; do
        # Kill process group (includes EngineCore child processes)
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    # Also kill any orphaned EngineCore processes on our GPUs
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup INT TERM EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local start=$(date +%s)
    echo "[wait] Waiting for ${name} on port ${port}..."
    while true; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "[wait] ${name} is ready (port ${port})."
            return 0
        fi
        local now=$(date +%s)
        if (( now - start >= TIMEOUT_SECONDS )); then
            echo "[wait] TIMEOUT waiting for ${name} on port ${port}"
            return 1
        fi
        sleep 2
    done
}

echo "============================================================"
echo " 1P1D MooncakeConnector Demo (xGMI)"
echo "============================================================"
echo " Model:          ${MODEL}"
echo " Prefill:        GPU ${PREFILL_GPU}, port ${PREFILL_PORT}"
echo " Decode:         GPU ${DECODE_GPU}, port ${DECODE_PORT}"
echo " Bootstrap:      port ${BOOTSTRAP_PORT}"
echo " Proxy:          port ${PROXY_PORT}"
echo " Mooncake lib:   ${MOONCAKE_LIB}"
echo "============================================================"
echo ""

# ---- Prefill Server (kv_producer) ----
echo "[launch] Starting Prefill server (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})..."
VLLM_MOONCAKE_BOOTSTRAP_PORT=${BOOTSTRAP_PORT} \
HIP_VISIBLE_DEVICES=${PREFILL_GPU} \
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_MODEL}" \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"xgmi"}}' \
    > "${LOG_DIR}/prefill.log" 2>&1 &
PIDS+=($!)
echo "  PID=$!"

# ---- Decode Server (kv_consumer) ----
echo "[launch] Starting Decode server (GPU ${DECODE_GPU}, port ${DECODE_PORT})..."
HIP_VISIBLE_DEVICES=${DECODE_GPU} \
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_MODEL}" \
    --port ${DECODE_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"xgmi"}}' \
    > "${LOG_DIR}/decode.log" 2>&1 &
PIDS+=($!)
echo "  PID=$!"

# ---- Wait for both servers ----
wait_for_server ${PREFILL_PORT} "Prefill" || { echo "Prefill failed to start"; exit 1; }
wait_for_server ${DECODE_PORT} "Decode"   || { echo "Decode failed to start"; exit 1; }

# ---- Proxy ----
echo "[launch] Starting Mooncake xGMI proxy (port ${PROXY_PORT})..."
PROXY_SCRIPT="${VLLM_ROOT}/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py"

python "${PROXY_SCRIPT}" \
    --prefill "http://0.0.0.0:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
    --decode  "http://0.0.0.0:${DECODE_PORT}" \
    --port ${PROXY_PORT} \
    > "${LOG_DIR}/proxy.log" 2>&1 &
PIDS+=($!)
echo "  PID=$!"

sleep 3
echo ""
echo "============================================================"
echo " All components launched. Proxy at http://localhost:${PROXY_PORT}"
echo " Logs: ${LOG_DIR}/"
echo "============================================================"
echo ""
echo "Test with:"
echo "  curl http://localhost:${PROXY_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVED_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
echo ""
echo "Or run:  python pd_distributed/test_1p1d.py --port ${PROXY_PORT}"
echo ""
echo "Press Ctrl+C to stop all servers."
wait
