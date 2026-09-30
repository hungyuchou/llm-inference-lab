from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

generated_ids = model.generate(**inputs)
# print(generated_ids.keys())
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
