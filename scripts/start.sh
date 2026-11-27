#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export YUE_WORKSPACE="${YUE_WORKSPACE:-$ROOT_DIR/yue2-workspace}"
QUANT="${YUE_QUANT:-Q8_0}"

BIN_DIR="${YUE_BIN_DIR:-$YUE_WORKSPACE/bin}"
if [ ! -x "$BIN_DIR/yue-server" ] && [ -x "/app/yue2-workspace/bin/yue-server" ]; then
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
for fname in ["YuE2-3B-$QUANT.gguf", "YuE2-Vae-F32.gguf"]:
    if not (models_dir / fname).exists():
        print(f"Downloading {fname} from Serveurperso/YuE2-GGUF...")
        hf_hub_download(repo_id="Serveurperso/YuE2-GGUF", filename=fname, local_dir=str(models_dir))
PY

export ACE_TRAIN_BIN="${ACE_TRAIN_BIN:-$BIN_DIR/ace-train}"
export YUE_SERVER_URL="${YUE_SERVER_URL:-http://127.0.0.1:8087}"
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

echo "Starting yue-server on port 8087..."
"$BIN_DIR/yue-server" \
    --host 0.0.0.0 \
    --port 8087 \
    --model "$MODELS_DIR/YuE2-3B-$QUANT.gguf" \
    --vae "$MODELS_DIR/YuE2-Vae-F32.gguf" \
    > "$YUE_WORKSPACE/yue-server.log" 2>&1 &

SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 45); do
    if python3 -c "import requests; assert requests.get('http://127.0.0.1:8087/health', timeout=2).status_code == 200" 2>/dev/null; then
        echo "yue-server is ONLINE."
        break
    fi
    sleep 2
done

cd "$ROOT_DIR"
exec python3 -m dashboard.app
