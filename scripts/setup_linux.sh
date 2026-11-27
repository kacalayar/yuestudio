#!/usr/bin/env bash
set -euo pipefail

# Universal setup script for Google Colab, Kaggle, Hugging Face Spaces, and Linux CUDA VPS.
WORKSPACE="${YUE_WORKSPACE:-$(pwd)/yue2-workspace}"
BIN_DIR="$WORKSPACE/bin"
MODELS_DIR="$WORKSPACE/models"
QUANT="${YUE_QUANT:-Q8_0}"
BUILD_TRAINER="${BUILD_LORA_TRAINER:-1}"
CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES:-75;80;86;89}"

mkdir -p "$BIN_DIR" "$MODELS_DIR" "$WORKSPACE/adapters" "$WORKSPACE/library" "$WORKSPACE/datasets" "$WORKSPACE/trained_loras"

echo "[1/4] Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build libcurl4-openssl-dev ffmpeg git python3-pip
fi

echo "[2/4] Building timoncool/yue2.cpp (yue-server CUDA)..."
if [ ! -x "$BIN_DIR/yue-server" ]; then
    SRC_YUE="${YUE_CPP_DIR:-/opt/yue2.cpp}"
    if [ ! -d "$SRC_YUE" ]; then
        git clone --recurse-submodules https://github.com/timoncool/yue2.cpp.git "$SRC_YUE"
    fi
    cmake -S "$SRC_YUE" -B "$SRC_YUE/build" -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
    cmake --build "$SRC_YUE/build" --config Release -j "$(nproc)"
    cp "$SRC_YUE/build/yue-server" "$BIN_DIR/yue-server"
    chmod +x "$BIN_DIR/yue-server"
fi

if [ "$BUILD_TRAINER" = "1" ]; then
    echo "[3/4] Building timoncool/HOT-Step-CPP (ace-train CUDA)..."
    if [ ! -x "$BIN_DIR/ace-train" ]; then
        SRC_TRAIN="${HOTSTEP_DIR:-/opt/HOT-Step-CPP}"
        if [ ! -d "$SRC_TRAIN" ]; then
            git clone --recurse-submodules https://github.com/timoncool/HOT-Step-CPP.git "$SRC_TRAIN"
        fi
        cmake -S "$SRC_TRAIN/engine" -B "$SRC_TRAIN/engine/build" -G Ninja \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
        cmake --build "$SRC_TRAIN/engine/build" --target ace-train --config Release -j "$(nproc)"
        cp "$SRC_TRAIN/engine/build/ace-train" "$BIN_DIR/ace-train"
        chmod +x "$BIN_DIR/ace-train"
    fi
fi

echo "[4/4] Installing Python dependencies & downloading YuE2 GGUF ($QUANT)..."
pip install --no-cache-dir -q -r "$(dirname "$0")/../dashboard/requirements.txt"

python3 - <<PY
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path("$MODELS_DIR")
for fname in ["YuE2-3B-$QUANT.gguf", "YuE2-Vae-F32.gguf"]:
    target = models_dir / fname
    if not target.exists():
        print(f"Downloading {fname}...")
        hf_hub_download(repo_id="Serveurperso/YuE2-GGUF", filename=fname, local_dir=str(models_dir))
    else:
        print(f"[OK] {fname}")
PY

echo "✅ Setup complete! Run ./scripts/start.sh to start yue-server + Gradio Dashboard."
