import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App(image=image, name="hello-world")

@app.function()
@modal.fastapi_endpoint()
def hello():
    return "Hello world!"