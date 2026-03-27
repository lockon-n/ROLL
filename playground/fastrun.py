# 并发测试脚本：向本地模型服务发送并发请求，测量吞吐和延迟
# 用法: python fastrun.py --model MODEL --port PORT --num_concurrent Z [--num_requests N] [--max_tokens T] [--print_every K]

import json
import requests
import concurrent.futures
import time
import argparse
import statistics


def call_model(prompt: str, model: str, port: int, max_tokens: int) -> dict:
    url = f"http://localhost:{port}/v1/chat/completions"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    response = requests.post(url, json=data, stream=True)
    response.raise_for_status()

    ttft = None
    content_parts: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        token_text = delta.get("content", "")
        if token_text and ttft is None:
            ttft = time.perf_counter() - start
        if token_text:
            content_parts.append(token_text)
        # 尝试从最后一个 chunk 的 usage 中拿 token 数
        usage = chunk.get("usage")

    latency = time.perf_counter() - start
    completion_tokens = usage.get("completion_tokens", 0) if usage else 0
    content = "".join(content_parts)
    return {
        "latency": latency,
        "ttft": ttft or latency,
        "completion_tokens": completion_tokens,
        "prompt": prompt,
        "response": content,
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrency benchmark for OpenAI-compatible API")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num_concurrent", type=int, required=True, help="Number of concurrent requests")
    parser.add_argument("--num_requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--print_every", type=int, default=0, help="Print prompt & response every K completed requests (0=off)")
    args = parser.parse_args()

    pseudo_prompts = [
        "Hello, who are you?",
        "Hello, how are you?",
        "Hello, what is your name?",
        "Hello, what is your favorite color?",
        "Hello, what is your favorite food?",
        "Hello, what is your favorite animal?",
        "Hello, what is your favorite book?",
        "Hello, what is your favorite movie?",
    ]
    # 循环填充到 num_requests 个
    prompts = [pseudo_prompts[i % len(pseudo_prompts)] for i in range(args.num_requests)]

    from tqdm import tqdm

    latencies: list[float] = []
    ttfts: list[float] = []
    total_tokens = 0
    errors = 0
    completed = 0

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_concurrent) as executor:
        futures = [executor.submit(call_model, p, args.model, args.port, args.max_tokens) for p in prompts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                latencies.append(result["latency"])
                ttfts.append(result["ttft"])
                total_tokens += result["completion_tokens"]
                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    tqdm.write(f"\n--- Sample #{completed} (latency: {result['latency']:.3f}s, TTFT: {result['ttft']:.3f}s) ---")
                    tqdm.write(f"  Prompt:   {result['prompt']}")
                    tqdm.write(f"  Response: {result['response'][:200]}")
            except Exception as e:
                errors += 1
                print(f"\nRequest failed: {e}")
    wall_time = time.perf_counter() - wall_start

    # 汇总统计
    print(f"\n{'='*50}")
    print(f"Concurrency:        {args.num_concurrent}")
    print(f"Total requests:     {args.num_requests}")
    print(f"Successful:         {len(latencies)}")
    print(f"Failed:             {errors}")
    print(f"Wall time:          {wall_time:.2f}s")
    print(f"Throughput:         {len(latencies) / wall_time:.2f} req/s")
    if total_tokens > 0:
        print(f"Token throughput:   {total_tokens / wall_time:.2f} tokens/s")
    if latencies:
        print(f"Latency avg:        {statistics.mean(latencies):.3f}s")
        print(f"Latency p50:        {statistics.median(latencies):.3f}s")
        sorted_lat = sorted(latencies)
        p99_idx = int(len(sorted_lat) * 0.99)
        print(f"Latency p99:        {sorted_lat[p99_idx]:.3f}s")
    if ttfts:
        print(f"TTFT avg:           {statistics.mean(ttfts):.3f}s")
        print(f"TTFT p50:           {statistics.median(ttfts):.3f}s")
        sorted_ttft = sorted(ttfts)
        p99_idx = int(len(sorted_ttft) * 0.99)
        print(f"TTFT p99:           {sorted_ttft[p99_idx]:.3f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()