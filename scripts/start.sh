#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export YUE_WORKSPACE="${YUE_WORKSPACE:-$ROOT_DIR/yue2-workspace}"
QUANT="${YUE_QUANT:-q8_0}"

BIN_DIR="${YUE_BIN_DIR:-$YUE_WORKSPACE/bin}"
if [ ! -x "$BIN_DIR/ace-server" ] && [ -x "/app/yue2-workspace/bin/ace-server" ]; then
    BIN_DIR="/app/yue2-workspace/bin"
fi
MODELS_DIR="$YUE_WORKSPACE/models"
ADAPTERS_DIR="$YUE_WORKSPACE/adapters"
mkdir -p "$MODELS_DIR" "$ADAPTERS_DIR" "$YUE_WORKSPACE/library" "$YUE_WORKSPACE/datasets" "$YUE_WORKSPACE/trained_loras"

# Auto-download models on first boot (useful for Docker / HF Spaces)
python3 - <<PY
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path("$MODELS_DIR")
for fname in ["yue2-lm-$QUANT.gguf", "yue2-vae-standard-f32.gguf", "yue2-tok-f16.gguf"]:
    if not (models_dir / fname).exists():
        print(f"Downloading {fname} from scragnog/YuE2-GGUF...")
        hf_hub_download(repo_id="scragnog/YuE2-GGUF", filename=fname, local_dir=str(models_dir))
PY

export ACE_TRAIN_BIN="${ACE_TRAIN_BIN:-$BIN_DIR/ace-train}"
export YUE_SERVER_BIN="$BIN_DIR/ace-server"
export YUE_MODELS_DIR="$MODELS_DIR"
export YUE_ADAPTERS_DIR="$ADAPTERS_DIR"
export YUE_SERVER_URL="${YUE_SERVER_URL:-http://127.0.0.1:8087}"
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

echo "Starting ace-server on port 8087..."
"$BIN_DIR/ace-server" \
    --models "$MODELS_DIR" \
    --adapters "$ADAPTERS_DIR" \
    --host 0.0.0.0 \
    --port 8087 \
    > "$YUE_WORKSPACE/yue-server.log" 2>&1 &

SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 45); do
    if python3 -c "import requests; assert requests.get('http://127.0.0.1:8087/health', timeout=2).status_code == 200" 2>/dev/null; then
        echo "ace-server is ONLINE."
        break
    fi
    sleep 2
done

cd "$ROOT_DIR"
exec python3 -m dashboard.app
