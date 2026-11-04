"""OpenAI-compatible vLLM server on Modal, used as the interpreter/annotator LLM for HypotheSAEs.

Deploy:   modal deploy vllm_server.py
Stop:     modal app stop hypothesaes-vllm

The server is protected by the API key in the Modal secret `hypothesaes-vllm-key`
(the same key is in ../.env.local as OPENAI_KEY_SAE, which HypotheSAEs sends automatically).
The container shuts down after SCALEDOWN_MINUTES without requests, so idle time costs nothing.
"""
import subprocess

import modal

MODEL = "Qwen/Qwen3-8B"
GPU = ["L4", "A10G"]  # both 24 GB (Qwen3-8B in bf16 needs ~16 GB); A10G is the fallback when L4s are scarce
MAX_MODEL_LEN = 8192
SCALEDOWN_MINUTES = 5
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers 5.x removed a tokenizer attribute that vLLM 0.11 still calls
    .uv_pip_install("vllm==0.11.0", "transformers<5", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("hypothesaes-vllm")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MINUTES * 60,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("hypothesaes-vllm-key")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        # keep Qwen3's <think> reasoning out of the returned answer text
        "--reasoning-parser", "qwen3",
        "--api-key", "$VLLM_API_KEY",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
