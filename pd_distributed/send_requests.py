"""
Send requests to vLLM server for Qwen3-8B-FP8-dynamic.

Usage:
    1. Start server:  bash pd_distributed/start_server.sh
    2. Send requests:  python pd_distributed/send_requests.py [--stream]
"""

import argparse
import time

from openai import OpenAI

API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL = "qwen3-8b-fp8"


def test_chat(client: OpenAI, stream: bool = False):
    """Basic chat completion request."""
    print("=" * 60)
    print("1. Chat Completion" + (" (streaming)" if stream else ""))
    print("=" * 60)

    messages = [
        {"role": "user", "content": "What is the capital of France? Answer briefly."},
    ]

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=256,
        temperature=0.0,
        stream=stream,
    )

    if stream:
        full_text = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                print(delta, end="", flush=True)
        print()
        elapsed = time.perf_counter() - start
        print(f"\n[Full response]: {full_text}")
    else:
        elapsed = time.perf_counter() - start
        result = response.choices[0].message.content
        usage = response.usage
        print(f"[Response]: {result}")
        print(f"[Usage]: prompt_tokens={usage.prompt_tokens}, "
              f"completion_tokens={usage.completion_tokens}, "
              f"total_tokens={usage.total_tokens}")

    print(f"[Latency]: {elapsed:.2f}s\n")


def test_multi_turn(client: OpenAI):
    """Multi-turn conversation."""
    print("=" * 60)
    print("2. Multi-turn Conversation")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": "What is Python?"},
    ]

    # Round 1
    resp1 = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=128, temperature=0.0,
    )
    answer1 = resp1.choices[0].message.content
    print(f"[Round 1] User: What is Python?")
    print(f"[Round 1] Assistant: {answer1}\n")

    # Round 2
    messages.append({"role": "assistant", "content": answer1})
    messages.append({"role": "user", "content": "What are its main advantages?"})

    resp2 = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=256, temperature=0.0,
    )
    answer2 = resp2.choices[0].message.content
    print(f"[Round 2] User: What are its main advantages?")
    print(f"[Round 2] Assistant: {answer2}\n")


def test_batch(client: OpenAI):
    """Batch of concurrent requests to test throughput."""
    print("=" * 60)
    print("3. Batch Requests (sequential, for throughput reference)")
    print("=" * 60)

    prompts = [
        "Explain quantum computing in one sentence.",
        "What is 25 * 37?",
        "Write a haiku about mountains.",
        "What programming language is vLLM written in?",
    ]

    start = time.perf_counter()
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        total_tokens += resp.usage.completion_tokens
        print(f"  [{i+1}] Q: {prompt}")
        print(f"      A: {text}\n")

    elapsed = time.perf_counter() - start
    print(f"[Batch summary]: {len(prompts)} requests, "
          f"{total_tokens} output tokens, {elapsed:.2f}s, "
          f"{total_tokens / elapsed:.1f} tokens/s\n")


def test_completions(client: OpenAI):
    """Text completion (non-chat) endpoint."""
    print("=" * 60)
    print("4. Text Completion (non-chat)")
    print("=" * 60)

    resp = client.completions.create(
        model=MODEL,
        prompt="The meaning of life is",
        max_tokens=64,
        temperature=0.7,
    )
    print(f"  Prompt: 'The meaning of life is'")
    print(f"  Generated: {resp.choices[0].text}\n")


def main():
    parser = argparse.ArgumentParser(description="Send requests to vLLM server")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--api-base", default=API_BASE, help="API base URL")
    parser.add_argument("--model", default=MODEL, help="Model name")
    args = parser.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=args.api_base)

    # Check server is alive
    try:
        models = client.models.list()
        print(f"Server online. Available models: "
              f"{[m.id for m in models.data]}\n")
    except Exception as e:
        print(f"Cannot connect to server at {args.api_base}: {e}")
        print("Please start the server first: bash pd_distributed/start_server.sh")
        return

    test_chat(client, stream=args.stream)
    test_multi_turn(client)
    test_batch(client)
    test_completions(client)

    print("=" * 60)
    print("All request tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
