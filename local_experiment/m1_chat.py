from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
# ]

# inputs = tokenizer(raw_inputs, padding=True, padding_side="left", truncation=True, return_tensors="pt")
# outputs = model(**inputs)

# generated_ids = model.generate(**inputs, max_new_tokens=100)
# print(generated_ids.keys())
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))


#chat template

raw_inputs = [

    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},

 ]
tokenized_chat = tokenizer.apply_chat_template(raw_inputs, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(**tokenized_chat, max_new_tokens=128)
print(tokenizer.batch_decode(outputs))
