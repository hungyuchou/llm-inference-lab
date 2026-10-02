from transformers import AutoTokenizer, AutoModelForCausalLM, AsyncTextIteratorStreamer
from threading import Thread
from time import perf_counter
import asyncio


checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

raw_inputs = [

    "How many helicopters can a human eat in one sitting?",

 ]

inputs = tokenizer(raw_inputs, padding=True, padding_side="left", truncation=True, return_tensors="pt")

async def main():
    # Important: AsyncTextIteratorStreamer must be initialized inside a coroutine!
    streamer = AsyncTextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    start = perf_counter()
    thread.start()
    generated_text = ""

    has_first_token = False
    word_count = 0

    async for new_text in streamer:

        if new_text and not has_first_token:
            has_first_token = True
            first_token_time = perf_counter()

        generated_text += new_text

        word_count += 1

    print(generated_text)

    end = perf_counter()
    
    print(f"TTFT: {first_token_time - start}")
    print(f"Total words generated: {word_count}")
    print(f"TPOT: {(end - start)/word_count} per token")

asyncio.run(main())