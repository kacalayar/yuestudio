"""LoRA / LoKr dataset preparation and training wrapper for HOT-Step-CPP (ace-train)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .env import default_workspace_root

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


@dataclass
class LoraTrainConfig:
    """Configuration for YuE2 Joint LoRA/LoKr training."""

    dataset_dir: str
    output_dir: str
    adapter_name: str
    trigger_word: str = "custom_style"
    adapter_type: str = "lokr"  # lokr | lora
    rank: int = 128
    alpha: int = 128
    learning_rate: float = 1e-4
    max_updates: int = 100
    save_every: int = 10
    preset: str = "Fast"  # Fast | Balanced | Thorough
    base_model: str = ""
    vae_model: str = ""
    tokenizer_model: str = ""
    trainer_bin: str = ""


def scan_audio_dataset(dataset_dir: str | Path) -> list[Path]:
    """Return sorted audio files in dataset_dir."""
    root = Path(dataset_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def prepare_dataset_manifest(
    dataset_dir: str | Path,
    trigger_word: str,
    default_genre: str = "pop, vocal",
    language: str = "en",
) -> tuple[Path, int]:
    """Create/update dataset.json and missing .yue2.txt sidecars for YuE2 training.

    Returns (manifest_path, track_count).
    """
    root = Path(dataset_dir)
    tracks = scan_audio_dataset(root)
    if not tracks:
        raise ValueError(f"Tidak ada file audio (.mp3/.wav/.flac) di {root}")

    entries: list[dict[str, object]] = []
    for audio_path in tracks:
        stem = audio_path.with_suffix("")
        lyrics_file = stem.with_suffix(".txt")
        yue2_caption_file = Path(f"{stem}.yue2.txt")

        lyrics_text = lyrics_file.read_text(encoding="utf-8").strip() if lyrics_file.exists() else "[instrumental]"
        if yue2_caption_file.exists():
            caption_text = yue2_caption_file.read_text(encoding="utf-8").strip()
        else:
            caption_text = f"{language}, {default_genre}, {trigger_word}".strip(", ")
            yue2_caption_file.write_text(caption_text + "\n", encoding="utf-8")

        entries.append(
            {
                "audio": str(audio_path.relative_to(root)).replace("\\", "/"),
                "caption": caption_text,
                "lyrics": lyrics_text,
                "trigger": trigger_word,
                "language": language,
            }
        )

    manifest_path = root / "dataset.json"
    manifest_path.write_text(json.dumps({"tracks": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, len(entries)


def _resolve_trainer_bin(cfg: LoraTrainConfig) -> str:
    default_bin = str(default_workspace_root() / "bin" / "ace-train")
    return cfg.trainer_bin or os.environ.get("ACE_TRAIN_BIN", default_bin)


def _trainer_models_dir(cfg: LoraTrainConfig) -> Path:
    if cfg.vae_model:
        return Path(cfg.vae_model).parent / "trainer"
    return default_workspace_root() / "models" / "trainer"


def _ensure_trainer_models(cfg: LoraTrainConfig) -> Iterator[str]:
    """Download scragnog/YuE2-GGUF files required by HOT-Step-CPP ace-train if missing."""
    from huggingface_hub import hf_hub_download

    t_dir = _trainer_models_dir(cfg)
    t_dir.mkdir(parents=True, exist_ok=True)

    # Clean up legacy symlinks from Serveurperso models that cause GGUF metadata mismatch
    parent_models = t_dir.parent
    for old_link in parent_models.glob("yue2-*.gguf"):
        if old_link.is_symlink():
            old_link.unlink(missing_ok=True)

    required = ["yue2-vae-standard-f32.gguf", "yue2-lm-q8_0.gguf", "yue2-tok-f16.gguf"]
    for fname in required:
        dest = t_dir / fname
        if not dest.exists():
            yield f"⬇️ Mengunduh {fname} dari scragnog/YuE2-GGUF (khusus ace-train)...\n"
            hf_hub_download(repo_id="scragnog/YuE2-GGUF", filename=fname, local_dir=str(t_dir))
            yield f"✅ {fname} siap.\n"


def build_preprocess_command(cfg: LoraTrainConfig, cache_dir: str | Path) -> list[str]:
    """Build the ace-train yue2-preprocess CLI command."""
    t_dir = _trainer_models_dir(cfg)
    return [
        _resolve_trainer_bin(cfg),
        "yue2-preprocess",
        "--audio", str(cfg.dataset_dir),
        "--out", str(cache_dir),
        "--caption-mode", "yue2",
        "--ffmpeg", "ffmpeg",
        "--models", str(t_dir),
        "--vae", str(t_dir / "yue2-vae-standard-f32.gguf"),
    ]


def build_train_command(cfg: LoraTrainConfig, manifest_path: str | Path | None = None) -> list[str]:
    """Build the ace-train yue2-nar-train CLI command for YuE2 LoRA training."""
    binary = _resolve_trainer_bin(cfg)
    t_dir = _trainer_models_dir(cfg)
    out_path = Path(cfg.output_dir) / cfg.adapter_name
    manifest = str(manifest_path or (out_path / "cache" / "yue2_preprocess.json"))
    return [
        binary,
        "yue2-nar-train",
        "--manifest", manifest,
        "--out", str(out_path),
        "--trigger", cfg.trigger_word,
        "--name", cfg.adapter_name,
        "--rank", str(cfg.rank),
        "--alpha", str(cfg.alpha),
        "--lr", str(cfg.learning_rate),
        "--steps", str(cfg.max_updates),
        "--save-every", str(cfg.save_every),
        "--kv-cache", "2",
        "--models", str(t_dir),
        "--lm", str(t_dir / "yue2-lm-q8_0.gguf"),
    ]


def _run_and_stream(cmd: list[str], env: dict[str, str]) -> Iterator[tuple[str, int | None]]:
    yield f"[CMD] {' '.join(cmd)}\n", None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line, None
    yield "", proc.wait()


def _pause_yue_server() -> bool:
    """Stop yue-server temporarily to free GPU VRAM for training."""
    try:
        res = subprocess.run(["pgrep", "-f", "yue-server"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            subprocess.run(["pkill", "-f", "yue-server"], check=False)
            return True
    except Exception:
        pass
    return False


def _restart_yue_server(bin_dir: Path) -> bool:
    """Restart yue-server in the background if model env vars are available."""
    model_file = os.environ.get("YUE_MODEL_FILE", "")
    vae_file = os.environ.get("YUE_VAE_FILE", "")
    server_bin = os.environ.get("YUE_SERVER_BIN", str(bin_dir / "yue-server"))
    if not (model_file and vae_file and Path(server_bin).exists()):
        return False
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else str(bin_dir)
    log_path = default_workspace_root() / "yue-server.log"
    log_f = open(log_path, "a")
    subprocess.Popen(
        [server_bin, "--host", "0.0.0.0", "--port", "8087", "--model", model_file, "--vae", vae_file],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return True


def stream_training(cfg: LoraTrainConfig) -> Iterator[str]:
    """Run yue2-preprocess (+ optional yue2-tokenize) + yue2-nar-train and yield live logs."""
    binary = Path(_resolve_trainer_bin(cfg))
    if not binary.exists():
        yield (
            f"[ERROR] Trainer binary tidak ditemukan di: {binary}\n"
            "Pastikan Sel 3 (Build HOT-Step-CPP / ace-train) sudah dijalankan.\n"
        )
        return

    out_dir = Path(cfg.output_dir) / cfg.adapter_name
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    bin_dir = str(binary.parent)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else bin_dir

    paused_server = _pause_yue_server()
    if paused_server:
        yield "⏸️ Menghentikan sementara yue-server untuk membebaskan VRAM GPU...\n"
        import time
        time.sleep(2)

    try:
        # Stage 0: Ensure scragnog/YuE2-GGUF trainer models exist
        for msg in _ensure_trainer_models(cfg):
            yield msg

        # Stage 1: yue2-preprocess
        yield "=== [1/2] Menjalankan yue2-preprocess (VAE latents & manifest) ===\n"
        prep_cmd = build_preprocess_command(cfg, cache_dir)
        rc = 0
        for line, code in _run_and_stream(prep_cmd, env):
            if line:
                yield line
            if code is not None:
                rc = code
        if rc != 0:
            yield f"\n[EXIT] yue2-preprocess gagal dengan kode: {rc}\n"
            return

        manifest_path = cache_dir / "yue2_preprocess.json"

        # Stage 1b: yue2-tokenize
        tok_file = _trainer_models_dir(cfg) / "yue2-tok-f16.gguf"
        if tok_file.exists():
            yield "\n=== [1b] Menjalankan yue2-tokenize ===\n"
            tok_cmd = [
                str(binary),
                "yue2-tokenize",
                "--manifest", str(manifest_path),
                "--tok", str(tok_file),
                "--ffmpeg", "ffmpeg",
            ]
            for line, code in _run_and_stream(tok_cmd, env):
                if line:
                    yield line
                if code is not None and code != 0:
                    yield f"\n[WARN] yue2-tokenize exit {code}, lanjut training text-only...\n"

        # Stage 2: yue2-nar-train
        yield "\n=== [2/2] Menjalankan yue2-nar-train (YuE2 LoRA training) ===\n"
        train_cmd = build_train_command(cfg, manifest_path)
        for line, code in _run_and_stream(train_cmd, env):
            if line:
                yield line
            if code is not None:
                rc = code
        yield f"\n[EXIT] Training selesai dengan kode: {rc}\n"
        if rc == 0:
            yield f"✅ Adapter tersimpan di: {out_dir / (cfg.adapter_name + '.safetensors')}\n"
    finally:
        if paused_server:
            if _restart_yue_server(binary.parent):
                yield "\n▶️ yue-server dijalankan kembali di background.\n"
            else:
                yield "\n⚠️ Training selesai. Jalankan ulang Sel 5 / yue-server untuk generate lagu.\n"


def register_trained_adapter(source_file: str | Path, adapters_dir: str | Path, alias: str | None = None) -> Path:
    """Copy a trained checkpoint into yue-server's ./adapters directory."""
    src = Path(source_file)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {src}")
    dest_dir = Path(adapters_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{alias}{src.suffix}" if alias else src.name
    dest = dest_dir / target_name
    shutil.copy2(src, dest)
    return dest
