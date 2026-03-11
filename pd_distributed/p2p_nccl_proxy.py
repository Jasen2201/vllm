"""
P2pNcclConnector 专用 Proxy

P2pNcclConnector 要求 request_id 包含 prefill/decode 的 ZMQ 地址:
  ___prefill_addr_{ip}:{port}___decode_addr_{ip}:{port}_{uuid}

该 proxy 接收客户端请求后:
1. 构造包含 ZMQ 地址的 request_id
2. 发送 prefill 请求 (max_tokens=1) 到 prefill 实例
3. 发送 decode 请求 (原始 max_tokens) 到 decode 实例
4. 返回 decode 的响应给客户端
"""

import argparse
import os
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = FastAPI()
config = {}


def make_request_id(prefill_zmq: str, decode_zmq: str) -> str:
    return (
        f"___prefill_addr_{prefill_zmq}___decode_addr_"
        f"{decode_zmq}_{uuid.uuid4().hex}"
    )


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            async for chunk in response.content.iter_chunked(1024):
                yield chunk


async def _handle(api: str, request: Request):
    req_data = await request.json()

    prefill_http = config["prefill_http"]
    decode_http = config["decode_http"]
    prefill_zmq = config["prefill_zmq"]
    decode_zmq = config["decode_zmq"]

    request_id = make_request_id(prefill_zmq, decode_zmq)

    # Prefill: max_tokens=1
    prefill_req = req_data.copy()
    prefill_req["max_tokens"] = 1
    if "max_completion_tokens" in prefill_req:
        prefill_req["max_completion_tokens"] = 1

    async for _ in forward_request(
        f"http://{prefill_http}{api}", prefill_req, request_id
    ):
        pass

    # Decode: original request
    async def generate():
        async for chunk in forward_request(
            f"http://{decode_http}{api}", req_data, request_id
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="application/json")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/v1/chat/completions", request)


def parse_args():
    parser = argparse.ArgumentParser("P2pNccl Proxy")
    parser.add_argument("--prefill-http", required=True, help="Prefill HTTP addr (host:port)")
    parser.add_argument("--decode-http", required=True, help="Decode HTTP addr (host:port)")
    parser.add_argument("--prefill-zmq", required=True, help="Prefill ZMQ addr (ip:port)")
    parser.add_argument("--decode-zmq", required=True, help="Decode ZMQ addr (ip:port)")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config["prefill_http"] = args.prefill_http
    config["decode_http"] = args.decode_http
    config["prefill_zmq"] = args.prefill_zmq
    config["decode_zmq"] = args.decode_zmq
    uvicorn.run(app, host="0.0.0.0", port=args.port)
