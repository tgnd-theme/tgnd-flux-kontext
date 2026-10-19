"""
RunPod Serverless Handler — TGND foto-bewerking (19 okt 2026).

Vervangt Flux Kontext dev (gated op de Hub, "cannot find the requested files") door open modellen zonder
contentfilter en met een commerciële licentie (Apache 2.0):
  - qwen  (standaard): Qwen/Qwen-Image-Edit-2509 + Lightning-LoRA (8 stappen, true_cfg 1.0). Beste gezichts- en
          lichaamsbehoud; ~57 GB gewichten, dus 80 GB-GPU en een network volume als cache.
  - klein: black-forest-labs/FLUX.2-klein-4B (4 stappen, guidance 1.0). Snel en klein; B-test naast Qwen.

Zelfde contract als de oude Kontext-worker, zodat WordPress (TGND_Studio_RunPod_Client::create_kontext_prediction)
niets hoeft te veranderen:
  input:  image (url of base64), prompt, seed (optioneel), model ("qwen" | "klein", optioneel),
          num_inference_steps / guidance_scale (optioneel; standaard per model)
  output: image (base64 PNG), mode, seed, inference_time, resolution
Health check: {"health_check": true}.
"""

import os
import io
import sys
import base64
import time
import traceback

try:
    import runpod
    print(f"[EDIT] runpod {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[EDIT] FATAL: cannot import runpod: {e}", flush=True)
    sys.exit(1)

import torch
from PIL import Image

print(f"[EDIT] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

VOLUME = "/runpod-volume"
if os.path.isdir(VOLUME):
    # Gewichten één keer downloaden, daarna uit het volume (koude start in seconden in plaats van minuten).
    os.environ.setdefault("HF_HOME", os.path.join(VOLUME, "hf"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    print(f"[EDIT] HF cache on volume: {os.environ['HF_HOME']}", flush=True)

QWEN_ID = "Qwen/Qwen-Image-Edit-2509"
QWEN_LORA_REPO = "lightx2v/Qwen-Image-Lightning"
QWEN_LORA_FILE = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
KLEIN_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_MODEL = os.environ.get("TGND_EDIT_MODEL", "qwen")
MAX_SIDE = 1536

_pipes = {}


def load_model(name):
    """Pipeline per model, één keer per worker."""
    if name in _pipes:
        return _pipes[name]
    t0 = time.time()
    token = os.environ.get("HF_TOKEN") or None
    if name == "klein":
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(KLEIN_ID, torch_dtype=torch.bfloat16, token=token)
        pipe.to("cuda")
    else:
        from diffusers import QwenImageEditPlusPipeline
        pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN_ID, torch_dtype=torch.bfloat16, token=token)
        pipe.load_lora_weights(QWEN_LORA_REPO, weight_name=QWEN_LORA_FILE, adapter_name="lightning")
        pipe.fuse_lora(adapter_names=["lightning"])
        pipe.unload_lora_weights()
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for attr in ("safety_checker", "feature_extractor"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)
    _pipes[name] = pipe
    print(f"[EDIT] {name} loaded in {time.time() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    return pipe


def fetch_image(src):
    if src.startswith("http"):
        import requests
        r = requests.get(src, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(src))).convert("RGB")


def fit(img):
    """Langste zijde max 1536, veelvoud van 16 (beide modellen willen dat)."""
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        w, h = int(w * s), int(h * s)
    w, h = (w // 16) * 16, (h // 16) * 16
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    return img


def handler(job):
    try:
        inp = job.get("input", {}) or {}
        model = inp.get("model") or DEFAULT_MODEL
        model = "klein" if model == "klein" else "qwen"
        if inp.get("health_check"):
            load_model(model)
            return {"status": "ok", "mode": model, "message": "Model loaded and ready"}

        src = inp.get("image", "")
        prompt = (inp.get("prompt") or "").strip()
        if not src:
            return {"error": "No input image provided"}
        if not prompt:
            return {"error": "No prompt provided"}

        seed = inp.get("seed")
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        gen = torch.Generator("cuda").manual_seed(int(seed))

        img = fit(fetch_image(src))
        w, h = img.size
        pipe = load_model(model)
        print(f"[EDIT] {model}: {w}x{h}, seed {seed}, prompt: {prompt[:100]}", flush=True)

        t0 = time.time()
        with torch.inference_mode():
            if model == "klein":
                steps = int(inp.get("num_inference_steps") or 4)
                out = pipe(prompt=prompt, image=img, width=w, height=h, guidance_scale=float(inp.get("guidance_scale") or 1.0),
                           num_inference_steps=steps, generator=gen)
            else:
                steps = int(inp.get("num_inference_steps") or 8)
                out = pipe(image=[img], prompt=prompt, negative_prompt=" ", true_cfg_scale=1.0, guidance_scale=1.0,
                           num_inference_steps=steps, width=w, height=h, num_images_per_prompt=1, generator=gen)
        dt = time.time() - t0
        result = out.images[0]
        print(f"[EDIT] done in {dt:.1f}s", flush=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG", optimize=True)
        return {
            "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "mode": model,
            "seed": int(seed),
            "inference_time": round(dt, 2),
            "resolution": f"{result.size[0]}x{result.size[1]}",
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
