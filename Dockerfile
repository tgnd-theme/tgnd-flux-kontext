FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade torch for CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
# diffusers from git main for FluxKontextPipeline support
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/huggingface/diffusers.git \
    'transformers>=4.50.0,<4.52.0' \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub \
    requests \
    Pillow

# Verify FluxKontextPipeline is available
RUN python3 -c "from diffusers import FluxKontextPipeline; print('FluxKontextPipeline OK')"

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
