#!/bin/bash
# Start vLLM OpenAI-compatible server for Qwen3-8B-FP8-dynamic (single GPU)
# Usage: bash pd_distributed/start_server.sh

export HIP_VISIBLE_DEVICES=0

MODEL_PATH="/mnt/raid0/pretrained_model/Qwen/Qwen3-8B-FP8-dynamic"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "qwen3-8b-fp8" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code \
    --port "$PORT" \
    --host 0.0.0.0
