import argparse, asyncio, json, os, time
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
            
            print('Cold start flag for prompt ID ', prompt_row['id'], ':', resp.json().get('cold_start', 'N/A'))

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
        print('Warmup request\'s Status: ', warmup_result['status'], 'Cold Start: ', warmup_result.get('cold_start', 'N/A'))

        sem = asyncio.Semaphore(concurrency)
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[send_one(client, url, row, max_tokens, sem) for row in prompts])
        wall = time.perf_counter() - wall_start

    return results, wall

def main():

    parser = argparse.ArgumentParser(prog='llm_inference_benchmark', description='To benchmark the LLM inference service')


    parser.add_argument('--url')  
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--max-tokens', default=128)
    parser.add_argument('--prompts', default='benchmark/prompts.jsonl')

    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    results, wall = asyncio.run(run(args.url, prompts, int(args.concurrency), int(args.max_tokens)))
    
    request_per_second = len(results) / wall

    response = f"""
    
    =============================================
    Total wall time: {wall:.2f} seconds
    Total requests: {len(results)}
    Request per second: {request_per_second:.2f}
    =============================================
    
    Stats for each request:

    """

    for result in results:
        response += f"ID: {result['id']}, Status: {result['status']}, Latency: {result.get('latency', -1):.2f}s, Generate elapsed: {result.get('generate_elapsed', -1):.2f}s\n"

    print(response)

if __name__ == "__main__":
    main()