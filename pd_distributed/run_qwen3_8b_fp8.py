"""
Qwen3-8B-FP8-dynamic single GPU inference script.
Usage:
    HIP_VISIBLE_DEVICES=0 python pd_distributed/run_qwen3_8b_fp8.py
"""

import time

from vllm import LLM, SamplingParams

MODEL_PATH = "/mnt/raid0/pretrained_model/Qwen/Qwen3-8B-FP8-dynamic"


def run_accuracy_check(llm: LLM):
    """Run a set of prompts to verify FP8 inference accuracy."""
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=256,
    )

    # ---- Test 1: Knowledge questions (factual accuracy) ----
    knowledge_prompts = [
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        [{"role": "user", "content": "What is 15 * 23? Show only the number."}],
        [{"role": "user", "content": "Who wrote 'Romeo and Juliet'? Answer in one word."}],
    ]
    expected_keywords = [
        ["Paris"],
        ["345"],
        ["Shakespeare"],
    ]

    print("=" * 70)
    print("Test 1: Factual Knowledge (greedy decoding)")
    print("=" * 70)

    outputs = llm.chat(knowledge_prompts, sampling_params, use_tqdm=False)
    passed = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        match = any(kw.lower() in text.lower() for kw in expected_keywords[i])
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        print(f"  [{status}] Q: {knowledge_prompts[i][0]['content']}")
        print(f"         A: {text}")
    print(f"  Result: {passed}/{len(knowledge_prompts)} passed\n")

    # ---- Test 2: Reasoning ----
    print("=" * 70)
    print("Test 2: Reasoning")
    print("=" * 70)
    reasoning_prompts = [
        [{"role": "user", "content": "If a train travels 120 km in 2 hours, what is its speed in km/h? Answer with just the number."}],
        [{"role": "user", "content": "Sort these numbers from smallest to largest: 5, 2, 8, 1, 9. Only output the sorted numbers separated by commas."}],
    ]
    expected_reasoning = [
        ["60"],
        ["1, 2, 5, 8, 9"],
    ]

    outputs = llm.chat(reasoning_prompts, sampling_params, use_tqdm=False)
    passed_r = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        match = any(kw in text for kw in expected_reasoning[i])
        status = "PASS" if match else "FAIL"
        if match:
            passed_r += 1
        print(f"  [{status}] Q: {reasoning_prompts[i][0]['content']}")
        print(f"         A: {text}")
    print(f"  Result: {passed_r}/{len(reasoning_prompts)} passed\n")

    # ---- Test 3: Long generation quality ----
    print("=" * 70)
    print("Test 3: Long generation coherence")
    print("=" * 70)
    long_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
    long_prompts = [
        [{"role": "user", "content": "Write a short poem about the ocean in English (4 lines)."}],
    ]
    outputs = llm.chat(long_prompts, long_params, use_tqdm=False)
    for output in outputs:
        text = output.outputs[0].text.strip()
        print(f"  Generated poem:\n{text}")
        has_content = len(text.split()) > 10
        no_garbage = not any(c * 5 in text for c in "!@#$%^&*")
        status = "PASS" if (has_content and no_garbage) else "FAIL"
        print(f"  [{status}] Content length ok: {has_content}, No garbage: {no_garbage}\n")

    # ---- Test 4: Throughput benchmark ----
    print("=" * 70)
    print("Test 4: Throughput benchmark")
    print("=" * 70)
    bench_params = SamplingParams(temperature=0.0, max_tokens=128)
    bench_prompts = [
        [{"role": "user", "content": f"Count from 1 to 20. (batch {i})"}]
        for i in range(16)
    ]
    start = time.perf_counter()
    outputs = llm.chat(bench_prompts, bench_params, use_tqdm=True)
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Batch size: {len(bench_prompts)}")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_tokens / elapsed:.1f} tokens/s\n")

    print("=" * 70)
    print("All tests completed.")
    print("=" * 70)


def main():
    print(f"Loading model: {MODEL_PATH}")
    print("Quantization: FP8 dynamic (compressed-tensors)")
    print("Tensor parallel: 1 (single GPU)")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )

    run_accuracy_check(llm)


if __name__ == "__main__":
    main()
