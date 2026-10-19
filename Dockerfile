FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Torch voor CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# diffusers van git main: QwenImageEditPlusPipeline en Flux2KleinPipeline; transformers nieuw genoeg voor Qwen2.5-VL en Qwen3.
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/huggingface/diffusers.git \
    'transformers>=4.56.0,<5' \
    accelerate \
    peft \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub \
    requests \
    Pillow

RUN python3 -c "from diffusers import QwenImageEditPlusPipeline, Flux2KleinPipeline; print('pipelines OK')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
