# llm-inference-lab
Benchmarking LLM inference (TTFT / TPOT / KV cache) from a CPU baseline to GPU serving.


## Decision Log

### M0 — Modal Serving cold start


| Situation | Time | 
|------------|-------|
| Cold start | 5.62s |
| Warm | 1.03s |

Warm requests were ~5.5x faster than the time spending on the cold start. Cold start is slow is because the app scales to zero, so the first request has to spin up a container. There is also a network time delay since the request is from local computer.

### M1.1 — Chat template ablation

The results below are using `Qwen2.5-0.5B-Instruct` model.

| Situation | Function | Results | 
|------------|-------|-----------|
| 1. No template | `model.generate` | The model keeps generating words |
| 2. Chat template with add_generation_prompt = `False` | ```tokenizer.apply_chat_template``` | The model continue to generate a response for answering with the beginning of <|im_start|> user |
| 3. Chat template with add_generation_prompt = `True` | tokenizer.apply_chat_template(raw_inputs, tokenize=True, add_generation_prompt=True, return_tensors="pt") | The normal response that the model will generate `<|im_end|>` to stop. |
| 4. Caht template with skip_special_tokens | tokenizer.batch_decode(true_add_generation_outputs, skip_special_tokens=True) | Similar response as in #3, but removing the machine tags for user-friendly reading. |

- What I learned through the process: 
1. The Instruct model is using ChatML format for SFT, where "answering after the assistant tag" is the pattern it learned during the training phase.
2. The responses might be different when calling the `model.generate` function since the config is using (word) sampling by default. When doing the experiment, using `do_sample=False` for fair comparison.

### M1.2 — TTFT vs TPOT

Configs: 
- CPU on Mac
- do_sample=False
- generate texts for few times first (to remove the cold start factor)
- max_new_tokens = 100
- Take the median out of three times


| prompt | Input tokens | TTFT (Time to first token) | TPOT (Time per output token) |
|------------|-------|------------|-------|
| Short | 11 | 0.082s | 17.3 ms |
| Long | 1,041 | 8.53s | 20.8 ms |
| Ratio | 95x | 104x | +20% |


- Before testing, my guess is the TTFT will increase by 100x, and the TPOT will increase a little bit -> The guess on TTFT is right
- Interpretation: The reason is using the CPU, where 11 tokens will also take up all capacity, hence the prefill is linearly proportional to the token numbers. 
- TPOT + 20% is larger that the initial guess as I thought the KV cache reading is just 1% more. The reason has not yet verified. Current hypothesis is because of GQA's repeat_kv. Plan: add 2k / 4k-token prompts and check whether TPOT grows linearly.
- How cold start affects: For the short prompt situation, if testing without warm-up on GCP, the TTFT is 9.9s; whereas after cold start, the TTFT is 0.87s, a 11x gap.
- Testing on GCP: The trend is similar (TTFT 205x, TPOT +166%), however magnitudes were exaggerated by the 2-core / low-memory VM, so I didn't use these numbers.


### M1.3 — Disabling KV cache

| Situation | TTFT (Time to first token) | TPOT (Time per output token) | Total time |
|------------|-------|------------|-----|
| With cache | 0.094s | 21.5 ms | 4.4s |
| Without cache | 0.104s | 995 ms | 3.3 mins |


- TTFT is similar as the first token is only affected by the prefill stage
- When there is no chace, the model has to calcuate (11+t) tokens all over again at step t, hence is 46x slower as the result. 

### Measurement caveats
1. The number testing with Mac's CPU cannot be extrapolated to GPU's cases, as CPU's capacity is 120 GFLOPS where L4 is 121 TFLOPS (1,000x), and the bandwidth is 60 vs 300 GB/s (5x). so prefill (compute-bound) and decode (memory-bound) would speed up by very different factors.
2. `TextIteratorStreamer` will output texts after the word finishes. It will affect short prompt by 20% since TTFT will take 1 more 1 TPOT.
3. Every function calling (model.generate) is running Only 3 runs per setting, on a single machine to take the median.
4. The +20% TPOT is unexplained (see M1.2).