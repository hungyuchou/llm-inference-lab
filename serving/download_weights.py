import time
import modal

image = modal.Image.debian_slim()\
             .pip_install("fastapi[standard]", "huggingface_hub", "torch", "transformers")

app = modal.App(image=image, name="download-weights")

vol = modal.Volume.from_name("download-weights-volume", create_if_missing=True)

MODEL = "Qwen2.5-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = "/models"                         

@app.function(volumes={MODEL_DIR: vol}, timeout=600)
def download():
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR + '/' + MODEL)
    elapsed = time.perf_counter() - start

    vol.commit()
    print(f"Downloaded in {elapsed:.1f}s")


@app.local_entrypoint()
def main():
    download.remote()