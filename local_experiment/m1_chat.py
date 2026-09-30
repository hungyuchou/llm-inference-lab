from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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


# without chat template

print("--- Without Chat template ---\n")

raw_inputs = [

    "How many helicopters can a human eat in one sitting?",

 ]


inputs = tokenizer(raw_inputs, padding=True, padding_side="left", truncation=True, return_tensors="pt")

generated_ids = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated_ids, skip_special_tokens=True), '\n')

#chat template

raw_inputs = [

    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},

 ]



# generated_ids = model.generate(**inputs, max_new_tokens=100)
# print(generated_ids.keys())
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

print("--- Chat template ---\n")

#add_generation_prompt=False
tokenized_chat = tokenizer.apply_chat_template(raw_inputs, tokenize=True, return_tensors="pt")
false_add_generation_outputs = model.generate(**tokenized_chat, max_new_tokens=128)

print(tokenizer.batch_decode(false_add_generation_outputs)[0], '\n')

#add_generation_prompt=True
tokenized_chat = tokenizer.apply_chat_template(raw_inputs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
true_add_generation_outputs = model.generate(**tokenized_chat, max_new_tokens=128)

print(tokenizer.batch_decode(true_add_generation_outputs)[0], '\n')


#skip_special_tokens

print(tokenizer.batch_decode(true_add_generation_outputs, skip_special_tokens=True)[0], '\n')
