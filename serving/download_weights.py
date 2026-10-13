import time
import modal

image = modal.Image.debian_slim()\
             .pip_install("huggingface_hub")

app = modal.App(image=image, name="download-weights")

vol = modal.Volume.from_name("model-weights", create_if_missing=True)

MODEL = "Qwen2.5-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = "/models"                         

@app.function(volumes={MODEL_DIR: vol}, timeout=1800)
def download():
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    snapshot_download(repo_id=MODEL_ID, local_dir=f"{MODEL_DIR}/{MODEL}/weights/")
    elapsed = time.perf_counter() - start

    vol.commit()
    print(f"Downloaded in {elapsed:.1f}s")


@app.local_entrypoint()
def main():
    download.remote()