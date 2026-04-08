#!/usr/bin/env python3
"""Benchmark DFlash vs standard generation."""

import time
import requests
import json

PORTS = {
    "standard": 11112,
    "dflash": 11111,
}

PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about coding.",
    "Explain quantum computing in simple terms.",
    "The quick brown fox",
]

def benchmark(port: int, prompt: str, max_tokens: int = 50) -> dict:
    """Run a single benchmark."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": "Qwen/Qwen3.5-27B",
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "time": elapsed,
                "tokens": data.get("usage", {}).get("completion_tokens", 0),
                "text": data["choices"][0]["text"],
            }
        else:
            return {"success": False, "error": resp.status_code, "time": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "time": time.time() - start}

def main():
    print("DFlash Benchmark")
    print("=" * 60)
    print("Make sure both servers are running:")
    print("  Standard: port 11112")
    print("  DFlash:   port 11111")
    print()

    results = {"standard": [], "dflash": []}

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] Prompt: '{prompt[:50]}...'")

        for mode, port in PORTS.items():
            result = benchmark(port, prompt)
            results[mode].append(result)

            if result["success"]:
                tokens_per_sec = result["tokens"] / result["time"] if result["time"] > 0 else 0
                print(f"  {mode:10s}: {result['time']:.2f}s ({tokens_per_sec:.1f} tok/s)")
                print(f"              Output: {result['text'][:50]}...")
            else:
                print(f"  {mode:10s}: ERROR - {result.get('error', 'unknown')}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for mode in ["standard", "dflash"]:
        successful = [r for r in results[mode] if r["success"]]
        if successful:
            avg_time = sum(r["time"] for r in successful) / len(successful)
            total_tokens = sum(r["tokens"] for r in successful)
            total_time = sum(r["time"] for r in successful)
            avg_tps = total_tokens / total_time if total_time > 0 else 0
            print(f"{mode:10s}: {avg_time:.2f}s avg, {avg_tps:.1f} tok/s")

if __name__ == "__main__":
    main()
