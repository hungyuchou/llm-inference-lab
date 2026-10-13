# llm-inference-lab
Benchmarking LLM inference (TTFT / TPOT / KV cache) from a CPU baseline to GPU serving.


## Decision Log

### M0 — Modal Serving cold start


| Situation | Time | 
|------------|-------|
| Cold start | 5.62s |
| Warm | 1.03s |

Warm requests were ~5.5x faster. Cold start is slow because the app scales to zero, so the first request has to spin up a container. There is also a network time delay since the request is from local computer.

### M1.1 — Chat template ablation

The results below are using `Qwen2.5-0.5B-Instruct` model.

| Situation | Function | Results | 
|------------|-------|-----------|
| 1. No template | `model.generate` | The model keeps generating words |
| 2. Chat template with add_generation_prompt = `False` | ```tokenizer.apply_chat_template``` | The model continue to generate a response for answering with the beginning of `<\|im_start\|>` user |
| 3. Chat template with add_generation_prompt = `True` | add_generation_prompt=True | The normal response that the model will generate `<\|im_end\|>` to stop. |
| 4. Chat template with skip_special_tokens | skip_special_tokens=True | Similar response as in #3, but removing the machine tags for user-friendly reading. |

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
- TPOT + 20% is larger that the initial guess as I thought the KV cache reading is just 1% more. The reason has not been verified yet. Current hypothesis is because of GQA's repeat_kv. Plan: add 2k / 4k-token prompts and check whether TPOT grows linearly.
- How cold start affects: For the short prompt situation, if testing without warm-up on GCP, the TTFT is 9.9s; whereas with warm-up, the TTFT is 0.87s, a 11x gap.
- Testing on GCP: The trend is similar (TTFT 205x, TPOT +166%), however magnitudes were exaggerated by the 2-core / low-memory VM, so I didn't use these numbers.


### M1.3 — Disabling KV cache

Setting: Short prompt (11 tokens), 200 new tokens , greedy.

| Situation | TTFT (Time to first token) | TPOT (Time per output token) | Total time |
|------------|-------|------------|-----|
| With cache | 0.094s | 21.5 ms | 4.4s |
| Without cache | 0.104s | 995 ms | 3.3 mins |


- TTFT is similar as the first token is only affected by the prefill stage
- When there is no cache, the model has to calculate (11+t) tokens all over again at step t, hence is 46x slower as the result. 

### M2.1 — GPU sizing (why choosing L4)

Model: `Qwen2.5-7B-Instruct`, bf16. Numbers below are estimates by hand, to be verified in M2.4 / M3.

Model config from the Hugging Face

```

{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}

```

**Memory budget on L4 (24 GB)**

| Item | Size | How I got it |
|------------|-------|------------|
| Weights | ~15.2GB | Number of Parameters (7.61B) × 2 bytes |
| CUDA context / activations (reserved) | 1GB | Served as a buffer for activation |
| Left for KV cache | 7.8 GB | 24 - 15.2 - 1 |

**KV cache per token**

```
Formula = 2 * num of hidden layer * num_key_value_heads * head_dim * bytes
        = 2 * 28 * 4 * 128 (3584/28) * 2
```

- The reason for having the 2 in the beginning is to consider K & V at the same time.
- Using `num_key_value_heads` because the model has GQA, where 28 query heads share 4 K/Vs.
- `head_dim` is calculated by hidden_size / attention head

**Capacity**

| | Value |
|------------|-------|
| Max tokens in KV cache | ~136k (Left for KV cache / KV cache per token = 7.8GB / 57,344 bytes)|
| Concurrent 2k-token requests | ~66 (Max tokens / 2048) |

- This is the theoretical upper limit and doesn't apply to the baseline, which serves one request at a time. We will compare multiple requests using vLLM.

**Why L4, not T4 or A100**

| GPU | Memory | Bandwidth | Verdict |
|------------|-------|------------|-----|
| T4 | 16 GB | 320GB/s | Cannot use all 16GB (model weight is already 15.2GB) + doesn't support bf16 |
| L4 | 24 GB | 300 GB/s | Cheapest among all options fit the model |
| A100 | 40 / 80 GB | 1.5-2TB/s | Faster, but more expensive as well |

**Prediction**

- The bottleneck of decode stage is on the memory bandwidth. For L4, the lower limit of TPOT is around weight size / memory bandwidth = 15.2GB / 300GB/s ~ 51ms per token. In reality it will be much slower due to the cost from Hugging Face and activation.
- We will check the actual number once we have results from the M3.

### Measurement caveats
1. The number testing with Mac's CPU cannot be extrapolated to GPU's cases, as CPU's capacity is 120 GFLOPS where L4 is 121 TFLOPS (1,000x), and the bandwidth is 60 vs 300 GB/s (5x), so prefill (compute-bound) and decode (memory-bound) would speed up by very different factors.
2. `TextIteratorStreamer` will output texts after the word finishes. It will affect short prompt by 20% since TTFT will take about 1 extra TPOT.
3. Every function calling (model.generate) is only 3 runs per setting (median reported), on a single machine.
4. The +20% TPOT is unexplained (see M1.2).