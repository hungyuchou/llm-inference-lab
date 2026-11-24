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
2. The responses might be different when calling the `model.generate` function since the config is using (word) sampling by default. When doing the experiment, using `do_sample=False (greedy)` for fair comparison.

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
        = 57,344 bytes
        = 56 KB
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

- The bottleneck of decode stage is on the memory bandwidth. For L4, the lower limit of TPOT is around weight size / memory bandwidth = 15.2GB / 300GB/s ~ 51ms per token. In reality it will be much slower (by 15% after testing in M2.3, which is 58.7ms) due to the cost from Hugging Face and activation. 
- We will check the actual number once we have results from the M3.

### Measurement caveats
1. The number testing with Mac's CPU cannot be extrapolated to GPU's cases, as CPU's capacity is 120 GFLOPS where L4 is 121 TFLOPS (1,000x), and the bandwidth is 60 vs 300 GB/s (5x), so prefill (compute-bound) and decode (memory-bound) would speed up by very different factors.
2. `TextIteratorStreamer` will output texts after the word finishes. It will affect short prompt by 20% since TTFT will take about 1 extra TPOT.
3. Every function calling (model.generate) is only 3 runs per setting (median reported), on a single machine.
4. The +20% TPOT is unexplained (see M1.2).

### M2.2 — Model weights on Modal Volume

Created an image that only installs `huggingface_hub` (no torch, since this step only downloads files). We also created a volume called `model-weights` under the container's path `/models`. `snapshot_download` downloads the whole repo to `/models/Qwen2.5-7B-Instruct/`. Since this path is under the volume’s mount point (/models), the files are written into the volume rather than the container’s temporary disk. After `vol.commit()`, the content is written to the volume.

Below are the stats of running `download_weights.py`:

| Run | Time | What happened |
|-----|------|---------------|
| 1st | 75.8s | Download 14 files where the weight is 14.2 GiB (15.2 GB) with ~200MB/s downloading speed. |
| 2nd | 5.9s  | Got faster as the files already exist. |

Cost: $0.01 (CPU only). Timing covers `snapshot_download` only (not image build / container start).

**Why not download from Hugging Face on every cold start?**
It would add ~76s to every cold start, on top of loading the weights onto the GPU. Also, if Hugging Face is down, our service would be affected.

**Why put weights in a Volume instead of baking them into the image?**
It's okay to put weights in the image. However, putting into the volume provides a flexibility of the possibility of changing the model. We would need to re-create the image every time when we need to change the model.


**Why is the 2nd run 13x faster?**
Since the `snapshot_download` finds the required repo & files, it won't proceed to download the files from Hugging Face. Hence, the script runs faster.


### M2.3 — Baseline `/generate` endpoint on L4

Setup: Create an /generate endpoint on L4 GPU for baseline benchmarking. The model weights are loaded from the Modal volume which is downloaded from previous `download_weights.py`.
Baseline on purpose: The max_containers parameter is set to 1 with no optimization for the baseline comparison.

**Results** 
Test config: 
- prompt: "Explain KV cache in 3 sentences."
- max_tokens=100

| Metric | Value |
|--------|-------|
| Model loading (volume → GPU) | 12 s |
| Input / output tokens | 38 / 72 |
| Generate time (server) | 4.22 s |
| Throughput | 17.04 tokens/s |
| TPOT |  58.7 ms |

**Latency breakdown (warm request)**

| Layer | Time | What's added at this layer |
|-------|------|----------------------------|
| `model.generate` | 4.22s | GPU compute: prefill (38 tokens) + 71 decode steps |
| Modal execution | 4.4s | + tokenization & decode |
| Modal duration | 4.51s | + internal transfer within Modal |
| Client (`curl` total) | 5.36s | + Singapore ↔ us-east round trip, TLS handshake, Modal web proxy |

**TPOT vs M2.1 prediction**

The predicted TPOT in M2.1 is 51ms, whereas the actual TPOT in the real measurement is 58.7ms. Based on this number, it uses 86% of L4 bandwidth. (15.2 GB of Model weight / 0.0587s = 259GB/s, which is roughly 86% of L4 bandwidth of 300 GB/s.)

**Why record server-side and client-side time separately?**
Client time is ~19% higher than Modal duration (5.36s vs 4.51s). If only measuring the client time, user would overestimate the model's latency. However, optimizations for the model only change the server part. So the server time, Modal duration, is the one to compare across the milestone.

**Observations**
- The first 3 requests (from earlier timed-out attempts) were slower (8–9.6 tokens/s), likely warmup.
- The output stopped at 72 tokens though the upper limit is 100. The model stops early since output the EOS.
- The 3 responses were identical since `do_sample` is set to `False` for the reproducible result.
- The response from the model for the KV cache is wrong though, likely because the prompt is ambiguous (KV cache is also a database term)

Cost: 0.03 USD on L4 GPU

**Caveats**
- There is a queuing time for 7 mins for getting GPU from Modal.
- Test 3 requests (excluding the warm-ups)
- Queue time is not billed.


### M2.4 — Cold start breakdown

Setup: Switched from `modal serve` to `modal deploy` with proxy auth. Pinned versions (torch 2.15.0, transformers 5.23.0, accelerate 1.15.0, fastapi 0.163.1).

How I measured: Recorded `time.time()` on the client before sending and at the first line of `@modal.enter` on the server; stage ① = server time − client time. Import and weight loading are timed with `perf_counter`.

| Stage | Time | Share |
|-------|------|-------|
| ① Queue for GPU + container start | 575.7 s | 96.6% |
| ② Import torch + transformers | 4.7 s | 0.8% |
| ③ Load weights (volume → GPU) | 9.4 s | 1.6% |
| ④ First request* (incl. generate 5.05 s) | 6.1 s | 1.0% |
| **Total (client)** | **595.9 s** | |

\* Not part of cold start — a warm request takes ~5.2 s too.

**Cold-start penalty = cold total − warm total ≈ 595.9 − 5.2 ≈ 590.7 s**

**Findings**
- Queuing takes 96.6% of the cold start. Excluding the queue: import (4.7 s) + weights (9.4 s) + extra time of the first generate (~0.8 s) ≈ 15 s, plus container start (2–12 s, estimated by polling) → roughly 17–27 s.
- Queue time varies a lot: ~7 min in M2.3 vs ~9.6 min in this run. GPU availability on Modal is the largest and least controllable factor.
- The first generate is ~0.8 s slower than warm ones (5.05 s vs 4.23 s; 14.3 vs 17.0 tok/s). Hypothesis: one-time GPU initialization on the first run (CUDA context setup, kernel loading) — not verified.
- Warm requests match M2.3 (4.23 s, 17.0 tok/s vs 4.22 s, 17.04 tok/s), so the baseline is stable and reproducible after pinning versions and switching to `modal deploy`.

**Trade-off: how to reduce cold start**
- `min_containers=1` keeps one container always running, so no queue and no cold start — but the L4 is billed 24/7 (~$0.80/h → ~$575/month).
- `scaledown_window`: longer → fewer cold starts but more idle cost; shorter → cheaper but more cold starts. Currently 180 s, and it works as expected (the container scaled to zero ~3 min after the last request).
- Only ② + ③ (~14 s) can be optimized in code; the queue (①) can only be traded for money (`min_containers`) or a more available GPU type.

**Caveats**
- There is no time measurement for loading the tokenizer, which is included in ④ (request time).
- Clock skew between Mac and Modal.