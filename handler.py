"""
RunPod Serverless Handler — Flux Kontext Dev image editing.

Takes a real photo + text instruction, outputs edited photo.
No safety checker. No NSFW filter. Full control.

Input:
  - image: URL or base64 of input image
  - prompt: edit instruction (e.g. "Change setting to bedroom, she is wearing only a black thong")
  - guidance_scale: float (default 2.5)
  - num_inference_steps: int (default 28)
  - seed: int (optional, random if not set)
  - width: int (optional, auto from input)
  - height: int (optional, auto from input)

Output:
  - image: base64 PNG
  - mode: "kontext"
  - seed: int
  - inference_time: float
"""

import os
import io
import sys
import base64
import time
import traceback

try:
    import runpod
    print(f"[KONTEXT] runpod {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[KONTEXT] FATAL: cannot import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"[KONTEXT] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[KONTEXT] FATAL: cannot import torch: {e}", flush=True)
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image
    print(f"[KONTEXT] numpy {np.__version__}, PIL OK", flush=True)
except Exception as e:
    print(f"[KONTEXT] FATAL: cannot import numpy/PIL: {e}", flush=True)
    sys.exit(1)

pipe = None


def load_model():
    """Load Flux Kontext Dev pipeline once on cold start."""
    global pipe

    if pipe is not None:
        return

    from diffusers import FluxKontextPipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"

    # Use network volume cache if available
    volume_cache = "/runpod-volume/models/kontext-dev"
    local_cache = None
    if os.path.exists("/runpod-volume"):
        if os.path.exists(os.path.join(volume_cache, "model_index.json")):
            local_cache = volume_cache
            print(f"[KONTEXT] Loading from volume cache: {volume_cache}", flush=True)
        else:
            print(f"[KONTEXT] Volume cache miss, downloading + caching...", flush=True)

    t0 = time.time()
    pipe = FluxKontextPipeline.from_pretrained(
        local_cache or model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token if hf_token else None,
    )

    # Cache to volume for next cold start
    if not local_cache and os.path.exists("/runpod-volume"):
        os.makedirs(volume_cache, exist_ok=True)
        pipe.save_pretrained(volume_cache)
        print(f"[KONTEXT] Cached model to {volume_cache}", flush=True)

    pipe.to("cuda")

    # Disable safety checker if it exists
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "feature_extractor"):
        pipe.feature_extractor = None

    dt = time.time() - t0
    print(f"[KONTEXT] Model loaded in {dt:.1f}s", flush=True)
    print(f"[KONTEXT] VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated", flush=True)


def fetch_image(url_or_b64):
    """Fetch image from URL or decode base64."""
    if url_or_b64.startswith("http"):
        import requests
        r = requests.get(url_or_b64, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img_bytes = base64.b64decode(url_or_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def handler(job):
    """RunPod serverless handler."""
    try:
        load_model()

        inp = job.get("input", {})

        # Health check for warmup pings
        if inp.get("health_check"):
            load_model()
            return {"status": "ok", "mode": "kontext", "message": "Model loaded and ready"}

        image_src = inp.get("image", "")
        prompt = inp.get("prompt", "")
        guidance_scale = float(inp.get("guidance_scale", 2.5))
        num_steps = int(inp.get("num_inference_steps", 28))
        seed = inp.get("seed", None)
        target_width = inp.get("width", None)
        target_height = inp.get("height", None)

        if not image_src:
            return {"error": "No input image provided"}
        if not prompt:
            return {"error": "No prompt provided"}

        # Fetch input image
        print(f"[KONTEXT] Fetching input image...", flush=True)
        input_image = fetch_image(image_src)
        orig_w, orig_h = input_image.size
        print(f"[KONTEXT] Input: {orig_w}x{orig_h}", flush=True)

        # Resize if target dimensions specified, otherwise use input size
        # Round to nearest 16 for FLUX compatibility
        if target_width and target_height:
            w = (int(target_width) // 16) * 16
            h = (int(target_height) // 16) * 16
        else:
            # Keep original aspect ratio, max 1536 on longest side
            max_side = 1536
            if max(orig_w, orig_h) > max_side:
                scale = max_side / max(orig_w, orig_h)
                w = (int(orig_w * scale) // 16) * 16
                h = (int(orig_h * scale) // 16) * 16
            else:
                w = (orig_w // 16) * 16
                h = (orig_h // 16) * 16

        if input_image.size != (w, h):
            input_image = input_image.resize((w, h), Image.LANCZOS)
            print(f"[KONTEXT] Resized to {w}x{h}", flush=True)

        # Set seed
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(int(seed))

        # Generate
        print(f"[KONTEXT] Generating: steps={num_steps}, guidance={guidance_scale}, seed={seed}", flush=True)
        print(f"[KONTEXT] Prompt: {prompt[:100]}...", flush=True)

        t0 = time.time()
        result = pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            width=w,
            height=h,
        )
        inference_time = time.time() - t0
        print(f"[KONTEXT] Done in {inference_time:.1f}s", flush=True)

        output_image = result.images[0]

        # Encode to base64 PNG
        buf = io.BytesIO()
        output_image.save(buf, format="PNG", optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image": img_b64,
            "mode": "kontext",
            "seed": seed,
            "inference_time": round(inference_time, 2),
            "resolution": f"{w}x{h}",
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
