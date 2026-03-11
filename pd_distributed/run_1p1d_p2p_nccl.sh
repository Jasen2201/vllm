#!/bin/bash
# =============================================================================
# 1P1D Disaggregated Serving Demo — P2pNcclConnector (xGMI / NCCL)
# =============================================================================
# GPU0 = Prefill (kv_producer, kv_rank=0)
# GPU1 = Decode  (kv_consumer, kv_rank=1)
# P2pNcclConnector 使用 NCCL 进行 GPU-to-GPU KV cache 传输
# 在 MI308 上走 xGMI 互联，带宽高、延迟低
#
# P2pNcclConnector 的特殊要求:
# - 每个 vLLM 实例的 P2pNcclEngine 都会 bind ZMQ ROUTER 到 kv_ip:kv_port
# - Proxy 需要将 prefill/decode 的 ZMQ 地址编码到 request_id 中
#   格式: ___prefill_addr_{ip}:{port}___decode_addr_{ip}:{port}_{uuid}
# - 使用专用 proxy (p2p_nccl_proxy.py) 而非通用 disagg_proxy_demo.py
# =============================================================================

set -euo pipefail

MODEL="/mnt/raid0/pretrained_model/Qwen/Qwen3-8B-FP8-dynamic"
SERVED_MODEL="qwen3-8b-fp8"

PREFILL_GPU=0
DECODE_GPU=1
PREFILL_PORT=8010
DECODE_PORT=8020
PROXY_PORT=8000

# P2pNcclConnector 配置
# 每个实例需要独立的 kv_port (ZMQ ROUTER bind)
KV_IP="127.0.0.1"
KV_PORT_PREFILL=14579
KV_PORT_DECODE=14580
KV_PARALLEL_SIZE=2

TIMEOUT_SECONDS=600

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_p2p_nccl"
mkdir -p "${LOG_DIR}"

PIDS=()

cleanup() {
    echo ""
    echo "[cleanup] Stopping all background processes..."
    for pid in "${PIDS[@]}"; do
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
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

# 获取本机 IP (P2pNcclEngine 默认使用 get_ip() 获取)
HOST_IP=$(python -c "from vllm.utils.network_utils import get_ip; print(get_ip())")
echo "============================================================"
echo " 1P1D P2pNcclConnector Demo (xGMI / NCCL)"
echo "============================================================"
echo " Model:          ${MODEL}"
echo " Host IP:        ${HOST_IP}"
echo " Prefill:        GPU ${PREFILL_GPU}, port ${PREFILL_PORT}, kv_port=${KV_PORT_PREFILL}"
echo " Decode:         GPU ${DECODE_GPU}, port ${DECODE_PORT}, kv_port=${KV_PORT_DECODE}"
echo " Proxy:          port ${PROXY_PORT}"
echo "============================================================"
echo ""

# ---- Prefill Server (kv_producer, kv_rank=0) ----
echo "[launch] Starting Prefill server (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})..."
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
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":${KV_PARALLEL_SIZE},\"kv_ip\":\"${KV_IP}\",\"kv_port\":${KV_PORT_PREFILL}}" \
    > "${LOG_DIR}/prefill.log" 2>&1 &
PIDS+=($!)
echo "  PID=$!"

# ---- Decode Server (kv_consumer, kv_rank=1) ----
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
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":${KV_PARALLEL_SIZE},\"kv_ip\":\"${KV_IP}\",\"kv_port\":${KV_PORT_DECODE}}" \
    > "${LOG_DIR}/decode.log" 2>&1 &
PIDS+=($!)
echo "  PID=$!"

# ---- Wait for both servers ----
wait_for_server ${PREFILL_PORT} "Prefill" || { echo "Prefill failed to start"; exit 1; }
wait_for_server ${DECODE_PORT} "Decode"   || { echo "Decode failed to start"; exit 1; }

# ---- Proxy ----
echo "[launch] Starting P2pNccl proxy (port ${PROXY_PORT})..."
python "${SCRIPT_DIR}/p2p_nccl_proxy.py" \
    --prefill-http "localhost:${PREFILL_PORT}" \
    --decode-http  "localhost:${DECODE_PORT}" \
    --prefill-zmq  "${HOST_IP}:${KV_PORT_PREFILL}" \
    --decode-zmq   "${HOST_IP}:${KV_PORT_DECODE}" \
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
