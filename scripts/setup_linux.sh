#!/usr/bin/env bash
set -euo pipefail

# Universal setup script for Google Colab, Kaggle, Hugging Face Spaces, and Linux CUDA VPS.
WORKSPACE="${YUE_WORKSPACE:-$(pwd)/yue2-workspace}"
BIN_DIR="$WORKSPACE/bin"
MODELS_DIR="$WORKSPACE/models"
QUANT="${YUE_QUANT:-q8_0}"
DETECTED_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' || true)"
CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES:-${DETECTED_ARCH:-75;80;86;89}}"

mkdir -p "$BIN_DIR" "$MODELS_DIR" "$WORKSPACE/adapters" "$WORKSPACE/library" "$WORKSPACE/datasets" "$WORKSPACE/trained_loras"

echo "[1/3] Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build libcurl4-openssl-dev ffmpeg git python3-pip
fi

echo "[2/3] Building timoncool/HOT-Step-CPP (ace-server + ace-train CUDA)..."
if [ ! -x "$BIN_DIR/ace-server" ] || [ ! -x "$BIN_DIR/ace-train" ]; then
    SRC_HOT="${HOTSTEP_DIR:-/opt/HOT-Step-CPP}"
    if [ ! -d "$SRC_HOT" ]; then
        git clone --recurse-submodules https://github.com/timoncool/HOT-Step-CPP.git "$SRC_HOT"
    fi
    cmake -S "$SRC_HOT/engine" -B "$SRC_HOT/engine/build" -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
    cmake --build "$SRC_HOT/engine/build" --target ace-server ace-train --config Release -j "$(nproc)"
    cp "$SRC_HOT/engine/build/ace-server" "$BIN_DIR/ace-server"
    cp "$SRC_HOT/engine/build/ace-train" "$BIN_DIR/ace-train"
    find "$SRC_HOT/engine/build" -name '*.so*' -exec cp -a {} "$BIN_DIR/" \;
    chmod +x "$BIN_DIR/ace-server" "$BIN_DIR/ace-train"
fi

echo "[3/3] Installing Python dependencies & downloading YuE2 GGUF ($QUANT)..."
pip install --no-cache-dir -q -r "$(dirname "$0")/../dashboard/requirements.txt"

python3 - <<PY
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path("$MODELS_DIR")
for fname in ["yue2-lm-$QUANT.gguf", "yue2-vae-standard-f32.gguf", "yue2-tok-f16.gguf"]:
    target = models_dir / fname
    if not target.exists():
        print(f"Downloading {fname}...")
        hf_hub_download(repo_id="scragnog/YuE2-GGUF", filename=fname, local_dir=str(models_dir))
    else:
        print(f"[OK] {fname}")
PY

echo "✅ Setup complete! Run ./scripts/start.sh to start ace-server + Gradio Dashboard."
