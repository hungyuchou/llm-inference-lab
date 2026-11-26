import argparse, asyncio, json, os, time, statistics
import httpx

def load_prompts(path):

    with open(path, 'r') as json_file:
        json_list = list(json_file)

    prompts = []
    
    for json_str in json_list:
        if not json_str.strip():
            continue
        result = json.loads(json_str)
        prompts.append(result)

    return prompts
    

async def send_one(client, url, prompt_row, max_tokens, sem):

    async with sem:
        try:
            start = time.perf_counter()
            resp = await client.post(url, json={'prompt': prompt_row['prompt'], 'max_tokens': max_tokens})
            latency = time.perf_counter() - start

            if resp.status_code != 200:
                print(f"Error for prompt ID {prompt_row['id']}: {resp.status_code} - {resp.text}")
                return {
                    'id': prompt_row['id'],
                    'category': prompt_row['category'],
                    'status': resp.status_code,
                    'latency': latency,
                    'error': resp.text
                }
            
            print('Cold start flag for prompt ID ', prompt_row['id'], ':', resp.json().get('cold_start', 'N/A'), 'Output tokens: ', resp.json().get('output_tokens', 'N/A'))

            return {
                'id': prompt_row['id'],
                'category': prompt_row['category'],
                'status': resp.status_code,
                'latency': latency,
                **resp.json()}
        
        except httpx.HTTPError as e:
            print(f"HTTP error for prompt ID {prompt_row['id']}: {e}")
            return {
                'id': prompt_row['id'],
                'category': prompt_row['category'],
                'status': 'HTTPError',
                'latency': -1,
                'error': str(e)
            }
    
async def run(url, prompts, concurrency, max_tokens):

    MODAL_KEY = os.environ["MODAL_KEY"]
    MODAL_SECRET = os.environ["MODAL_SECRET"]

    async with httpx.AsyncClient(headers={"MODAL-KEY": MODAL_KEY, "MODAL-SECRET": MODAL_SECRET}, follow_redirects=True, timeout=300) as client:

        warmup_result = await send_one(client, url, prompts[0], max_tokens, asyncio.Semaphore(concurrency))
        print('Warmup request\'s Status: ', warmup_result['status'], 'Cold Start: ', warmup_result.get('cold_start', 'N/A'), 'Output tokens: ', warmup_result.get('output_tokens', 'N/A'))

        sem = asyncio.Semaphore(concurrency)
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[send_one(client, url, row, max_tokens, sem) for row in prompts])
        wall = time.perf_counter() - wall_start

    return results, wall

def save_results(path, meta, metrics, results):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump({"meta": meta, "metrics": metrics, "results": results}, f, indent=4)


def compute_metrics(results, wall):

    updated_results = [x for x in results if x['status'] == 200]
    
    #latency
    lat = sorted(x['latency'] for x in updated_results)

    mean_latency = sum(x['latency'] for x in updated_results) / len(updated_results) if updated_results else None
    p99_latency = statistics.quantiles(lat, n=100, method='inclusive')[98] if len(lat) >= 2 else (lat[0] if lat else None)
    p50_latency = statistics.median(lat) if lat else None
    max_latency = max(x['latency'] for x in updated_results) if updated_results else None

    request_per_second = len(updated_results) / wall if wall > 0 else None
    output_tokens_per_second = sum(x.get('output_tokens', 0) for x in updated_results) / wall if wall > 0 else None

    speeds = [x['output_tokens'] / x['generate_elapsed'] for x in updated_results if x.get('output_tokens') and x.get('generate_elapsed')]

    decode_tokens_per_second = sum(speeds) / len(speeds) if speeds else None

    TTFT = None

    return {

        'mean_latency': mean_latency,
        'p99_latency': p99_latency,
        'p50_latency': p50_latency,
        'max_latency': max_latency,
        'request_per_second': request_per_second,
        'output_tokens_per_second': output_tokens_per_second,
        'decode_tokens_per_second': decode_tokens_per_second,
        'TTFT': TTFT
    } 

def fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"

def main():

    parser = argparse.ArgumentParser(prog='llm_inference_benchmark', description='To benchmark the LLM inference service')


    parser.add_argument('--url')  
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--max-tokens', default=128, type=int)
    parser.add_argument('--prompts', default='benchmark/prompts.jsonl')
    parser.add_argument('--tag', default='baseline')

    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    results, wall = asyncio.run(run(args.url, prompts, args.concurrency, args.max_tokens))

    metrics = compute_metrics(results, wall)
    meta = {"tag": args.tag, "concurrency": args.concurrency, "max_tokens": args.max_tokens}
    path = f"benchmark/results/{args.tag}_c{args.concurrency}_{time.strftime('%Y%m%d-%H%M%S')}.json"

    save_results(path, meta, metrics, results)

    response = f"""
    
    =============================================
    Total wall time: {fmt(wall)} seconds
    Total requests: {len(results)}
    Request per second: {fmt(metrics['request_per_second'])}
    P50 / P99 latency: {fmt(metrics['p50_latency'])} / {fmt(metrics['p99_latency'])}s
    Output tokens/s: {fmt(metrics['output_tokens_per_second'])}
    Decode tokens/s: {fmt(metrics['decode_tokens_per_second'])}
    TTFT: {fmt(metrics['TTFT'])}
    =============================================
    
    Stats for each request:

    """

    for result in results:
        response += f"ID: {result['id']}, Status: {result['status']}, Latency: {fmt(result.get('latency', -1))}s, Generate elapsed: {fmt(result.get('generate_elapsed', -1))}s\n"

    print(response)

if __name__ == "__main__":
    main()