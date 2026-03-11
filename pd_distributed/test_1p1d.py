"""
1P1D Disaggregated Serving 验证脚本

通过 Proxy 端口发送请求，验证 Prefill-Decode 分离推理是否正常工作。
适用于 MooncakeConnector 和 P2pNcclConnector 两种模式。

用法:
    python pd_distributed/test_1p1d.py --port 8000
    python pd_distributed/test_1p1d.py --port 8000 --model qwen3-8b-fp8
"""

import argparse
import json
import re
import time
import requests


def strip_think(text: str) -> str:
    """移除 <think>...</think> 标签及其内容，返回最终回答"""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return result if result else text


def test_chat_basic(base_url: str, model: str):
    """基础 chat 请求"""
    print("=" * 60)
    print("Test 1: Basic Chat Completion")
    print("=" * 60)

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "What is 2+3? Answer with just the number."},
            ],
            "max_tokens": 256,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    answer = strip_think(content)
    print(f"  Response: {answer}")
    # 检查整个内容（包括 think 部分）或最终回答
    assert "5" in content, f"Expected '5' in response, got: {content[:200]}"
    print("  PASS")
    return True


def test_chat_long_output(base_url: str, model: str):
    """较长输出测试"""
    print("=" * 60)
    print("Test 2: Longer Generation")
    print("=" * 60)

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "List the first 5 prime numbers, one per line."},
            ],
            "max_tokens": 512,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print(f"  Response (first 200 chars): {content[:200]}...")
    for p in ["2", "3", "5", "7", "11"]:
        assert p in content, f"Expected '{p}' in response"
    print("  PASS")
    return True


def test_completion_api(base_url: str, model: str):
    """Completions API 测试"""
    print("=" * 60)
    print("Test 3: Completions API")
    print("=" * 60)

    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    print(f"  Response: {text}")
    assert "paris" in text.lower(), f"Expected 'Paris' in response, got: {text}"
    print("  PASS")
    return True


def test_concurrent_requests(base_url: str, model: str):
    """并发请求测试"""
    print("=" * 60)
    print("Test 4: Concurrent Requests (3 requests)")
    print("=" * 60)

    import concurrent.futures

    prompts = [
        "What is the largest planet in our solar system? One word answer.",
        "What is the chemical symbol for water? One word answer.",
        "How many legs does a spider have? One word answer.",
    ]
    expected = ["jupiter", "h2o", "8|eight"]

    def send_one(prompt):
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(send_one, prompts))
    elapsed = time.time() - start

    for i, (result, expect) in enumerate(zip(results, expected)):
        answer = strip_think(result)
        print(f"  Q{i+1}: {prompts[i][:50]}...")
        print(f"  A{i+1}: {answer[:100]}")
        # 搜索整个内容（包括 think 部分），支持 | 分隔的多个关键词
        keywords = expect.split("|")
        assert any(k in result.lower() for k in keywords), \
            f"Expected one of {keywords} in response"

    print(f"  3 requests completed in {elapsed:.2f}s")
    print("  PASS")
    return True


def test_streaming(base_url: str, model: str):
    """流式输出测试"""
    print("=" * 60)
    print("Test 5: Streaming Response")
    print("=" * 60)

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 128,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                chunks.append(content)

    full_text = "".join(chunks)
    print(f"  Streamed {len(chunks)} chunks: {full_text[:100]}...")
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    print("  PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test 1P1D disaggregated serving")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--model", type=str, default="qwen3-8b-fp8")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"Connecting to proxy at {base_url}...")
    try:
        requests.get(f"{base_url}/v1/models", timeout=5)
    except Exception:
        pass

    tests = [
        test_chat_basic,
        test_chat_long_output,
        test_completion_api,
        test_concurrent_requests,
        test_streaming,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn(base_url, args.model)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
