# llm-inference-lab
An agent service to test the benchmark with quantified numbers.


## Decision Log

### M0 — Modal Serving cold start


| Situation | Time | 
|------------|-------|
| Cold start | 5.62s |
| Afterwards | 1.03s |

The following serving is 5.5x less than the time spending on the cold start. The reason is beacuse when the endpoint starts, the container needs to start as well to scale from zero. 

### M1.1 — Chat template ablation

The results below are using `Qwen2.5-0.5B-Instruct` model.

（表格：三組 → 結果）
- 學到什麼：
- 踩坑：


| Situation | Function | Results | 
|------------|-------|-----------|
| 1. No template | `model.generate` | The model keeps generating words |
| 2. Chat template with add_generation_prompt = `False` | ```python tokenized_chat = tokenizer.apply_chat_template(raw_inputs, tokenize=True, return_tensors="pt")
false_add_generation_outputs = model.generate(**tokenized_chat, max_new_tokens=128)``` | The model continue to generate a response for answering with the beginning of ``|im_start`|> user` |
| 3. Chat template with add_generation_prompt = `True` | ``` python tokenized_chat = tokenizer.apply_chat_template(raw_inputs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
true_add_generation_outputs = model.generate(**tokenized_chat, max_new_tokens=128)``` | The normal response that the model will generate `<|im_end|>` to stop. |
| 4. Caht template with skip_special_tokens |```python tokenizer.batch_decode(true_add_generation_outputs, skip_special_tokens=True)```| Similiar response as in #3, but removing the machine tags for user-friendly reading. |

- What I learned through the process: 
1. The Instruct model is uinsg ChatML format for SFT, where "answering after the assistant tag" is the pattern it learned during the training phase.
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
- Interpretation: The reason is using the CPU, where 11 tokens will also take up all capacity, hence the prefill is linearly propotional to the token numbers. 
- TPOT + 20% is larger that the intial guess as I thought the KV chache reading is just 1% more. `The reason has not yet accessed`. Current hypothesis is because of GQA's repeat_kv.
- How cold start affects: For the short promopt situation, if testing with the cold start, the TTFT is 9.9s; whereas after cold start, the TTFT is 0.87s, a 11x gap.
- Testing on GCP: The trend is similar (TTFT 205x, TPOT +166%), however due to the spec with only 2 cores and limited memory, the scale is much larger.


### M1.3 — Disabling KV cache
（表格：有/沒 cache）
- 解讀：

### Measurement caveats
1.
2.
3.