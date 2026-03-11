"""
GSM8K evaluation for Qwen3-8B-FP8-dynamic.

Supports two modes:
  1. Offline (default) - loads model locally, no server needed
  2. Online - sends requests to a running vLLM server

Usage:
  # Offline (single GPU, self-contained)
  HIP_VISIBLE_DEVICES=0 python pd_distributed/eval_gsm8k.py

  # Limit question count for quick test
  HIP_VISIBLE_DEVICES=0 python pd_distributed/eval_gsm8k.py --num-questions 100

  # Online (requires start_server.sh running)
  python pd_distributed/eval_gsm8k.py --online --port 8000

  # Save results
  HIP_VISIBLE_DEVICES=0 python pd_distributed/eval_gsm8k.py --save-results results.json
"""

import sys
import os

# Add project root so we can import gsm8k_eval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time

from tests.evals.gsm8k.gsm8k_eval import (
    evaluate_gsm8k,
    evaluate_gsm8k_offline,
)


MODEL_PATH = "/mnt/raid0/pretrained_model/Qwen/Qwen3-8B-FP8-dynamic"


def run_offline(args):
    """Run GSM8K evaluation in offline mode (no server needed)."""
    from vllm import LLM

    print(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )

    results = evaluate_gsm8k_offline(
        llm=llm,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    return results


def run_online(args):
    """Run GSM8K evaluation against a running vLLM server."""
    results = evaluate_gsm8k(
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        seed=args.seed,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GSM8K evaluation for Qwen3-8B-FP8-dynamic"
    )
    parser.add_argument(
        "--online", action="store_true",
        help="Use online mode (requires running server)"
    )
    parser.add_argument("--num-questions", type=int, default=1319,
                        help="Number of test questions (default: 1319 = full set)")
    parser.add_argument("--num-shots", type=int, default=5,
                        help="Number of few-shot examples")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save-results", type=str,
                        help="Save results to JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("GSM8K Evaluation - Qwen3-8B-FP8-dynamic")
    print(f"  Mode:       {'online' if args.online else 'offline'}")
    print(f"  Questions:  {args.num_questions}")
    print(f"  Few-shot:   {args.num_shots}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temp:       {args.temperature}")
    print("=" * 60)

    tic = time.perf_counter()

    if args.online:
        results = run_online(args)
    else:
        results = run_offline(args)

    total_time = time.perf_counter() - tic

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Accuracy:            {results['accuracy']:.4f} "
          f"({results['accuracy']*100:.1f}%)")
    print(f"  Invalid rate:        {results['invalid_rate']:.4f}")
    print(f"  Total output tokens: {results['total_output_tokens']}")
    print(f"  Throughput:          {results['tokens_per_second']:.1f} tokens/s")
    print(f"  Latency:             {results['latency']:.1f}s")
    print(f"  Total time:          {total_time:.1f}s")
    print("=" * 60)

    # Reference: Qwen3-8B BF16 typically gets ~79-82% on GSM8K (5-shot)
    # FP8 should be within ~1% of BF16 accuracy
    acc = results['accuracy']
    if acc >= 0.75:
        print("PASS - Accuracy is in expected range for Qwen3-8B")
    elif acc >= 0.60:
        print("WARN - Accuracy is lower than expected, possible quantization issue")
    else:
        print("FAIL - Accuracy is significantly below expected range")

    if args.save_results:
        results["model"] = MODEL_PATH
        results["mode"] = "online" if args.online else "offline"
        results["total_time"] = total_time
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
