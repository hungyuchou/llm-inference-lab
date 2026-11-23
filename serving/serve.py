import time
import modal

image = modal.Image.debian_slim()\
             .pip_install("torch", "transformers" ,"accelerate", "fastapi[standard]")

app = modal.App(image=image, name="llm-serving-baseline")

vol = modal.Volume.from_name("model-weights")

MODEL = "Qwen2.5-7B-Instruct"
MODEL_DIR = "/models"                         


@app.cls(gpu="L4", volumes={MODEL_DIR: vol}, scaledown_window=180, max_containers=1)
class Model:

    @modal.enter()
    def loading_model(self):

        start = time.perf_counter()
        import torch
        import_torch = time.perf_counter()
        self.torch_elapsed = import_torch - start
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import_transformer = time.perf_counter()
        self.transformer_elapsed = import_transformer - import_torch
        # checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

        loading_start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(f"{MODEL_DIR}/{MODEL}", dtype=torch.bfloat16, device_map="cuda")
        loading_elapsed = time.perf_counter() - loading_start
        self.loading_elapsed = loading_elapsed
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/{MODEL}")
        
        print(f"Loading in {loading_elapsed:.1f}s")

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate(self, req: dict):

        prompt = req['prompt']
        inputs = [{'role':'user', 'content': prompt}]

        tokenized_chat = self.tokenizer.apply_chat_template(inputs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        token_length = tokenized_chat['input_ids'].shape[1]

        generate_start = time.perf_counter()
        generation_outputs = self.model.generate(**tokenized_chat, max_new_tokens= req.get('max_tokens', 200), do_sample=False)
        generate_elapsed = time.perf_counter() - generate_start

        generated_tokens = len(generation_outputs[0][token_length:])

        print(f"Generate in {generate_elapsed:.1f}s. Avg {generated_tokens/generate_elapsed} token/s")
        
        

        return {'prompt': prompt, 
                'response': self.tokenizer.decode(generation_outputs[0][token_length:], skip_special_tokens=True),
                'avg_token_per_second': generated_tokens/generate_elapsed,
                'input_tokens': token_length,
                'output_tokens': generated_tokens,
                'generate_elapsed': generate_elapsed,
                }
