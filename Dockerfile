# Compatible with both Hugging Face Spaces (Docker GPU) and self-hosted CUDA VPS
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    YUE_QUANT=Q8_0 \
    GRADIO_PORT=7860

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    ca-certificates cmake ninja-build git curl libcurl4-openssl-dev ffmpeg \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build yue2.cpp (yue-server)
RUN git clone --recurse-submodules https://github.com/timoncool/yue2.cpp.git /opt/yue2.cpp \
    && cmake -S /opt/yue2.cpp -B /opt/yue2.cpp/build -G Ninja \
       -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
       -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    && cmake --build /opt/yue2.cpp/build --config Release -j "$(nproc)" \
    && mkdir -p /app/yue2-workspace/bin \
    && cp /opt/yue2.cpp/build/yue-server /app/yue2-workspace/bin/yue-server \
    && find /opt/yue2.cpp/build -name '*.so*' -exec cp -a {} /app/yue2-workspace/bin/ \; \
    && rm -rf /opt/yue2.cpp

# Build HOT-Step-CPP (ace-train for LoRA training)
RUN git clone --recurse-submodules https://github.com/timoncool/HOT-Step-CPP.git /opt/HOT-Step-CPP \
    && cmake -S /opt/HOT-Step-CPP/engine -B /opt/HOT-Step-CPP/engine/build -G Ninja \
       -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
       -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    && cmake --build /opt/HOT-Step-CPP/engine/build --target ace-train --config Release -j "$(nproc)" \
    && cp /opt/HOT-Step-CPP/engine/build/ace-train /app/yue2-workspace/bin/ace-train \
    && find /opt/HOT-Step-CPP/engine/build -name '*.so*' -exec cp -a {} /app/yue2-workspace/bin/ \; \
    && rm -rf /opt/HOT-Step-CPP

COPY dashboard/requirements.txt /app/dashboard/requirements.txt
RUN pip3 install --no-cache-dir -r /app/dashboard/requirements.txt

COPY dashboard /app/dashboard
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh

ENV YUE_WORKSPACE=/app/yue2-workspace \
    ACE_TRAIN_BIN=/app/yue2-workspace/bin/ace-train

EXPOSE 7860 8087

CMD ["/app/scripts/start.sh"]
