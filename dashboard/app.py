"""YuE2-Colab-Studio: Bilingual (ID/EN) Gradio Dashboard inspired by YuE2-Studio."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download

from .env import default_gradio_share, default_workspace_root, detect_platform
from .trainer import (
    LoraTrainConfig,
    prepare_dataset_manifest,
    register_trained_adapter,
    scan_audio_dataset,
    stream_training,
)
from .yue_client import YueServerClient, build_synth_payload, list_adapters

PLATFORM = detect_platform()
BASE_DIR = default_workspace_root(PLATFORM)
MODELS_DIR = Path(os.environ.get("YUE_MODELS_DIR", str(BASE_DIR / "models")))
ADAPTERS_DIR = Path(os.environ.get("YUE_ADAPTERS_DIR", str(BASE_DIR / "adapters")))
LIBRARY_DIR = Path(os.environ.get("YUE_LIBRARY_DIR", str(BASE_DIR / "library")))
DATASETS_DIR = Path(os.environ.get("YUE_DATASETS_DIR", str(BASE_DIR / "datasets")))
LORA_OUT_DIR = Path(os.environ.get("YUE_LORA_OUT_DIR", str(BASE_DIR / "trained_loras")))
DRIVE_ROOT = Path(os.environ.get("YUE_DRIVE_ROOT", "/content/drive/MyDrive/YuE2-Studio"))

for _d in (MODELS_DIR, ADAPTERS_DIR, LIBRARY_DIR, DATASETS_DIR, LORA_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

client = YueServerClient()


def get_server_banner() -> str:
    online = client.is_online()
    status = "🟢 ONLINE" if online else "🔴 OFFLINE (Jalankan yue-server terlebih dahulu / Start yue-server first)"
    adapters = list_adapters(ADAPTERS_DIR)
    return (
        f"**Platform:** `{PLATFORM.upper()}` | **Workspace:** `{BASE_DIR}` | "
        f"**Engine:** {status} | **Adapters:** {len(adapters)}"
    )


def get_gpu_status() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        return f"GPU: `{out}`"
    except Exception:
        return "GPU: `nvidia-smi tidak tersedia / unavailable (mode lokal)`"


def refresh_adapter_choices() -> tuple[Any, Any, Any]:
    choices = ["(None)"] + list_adapters(ADAPTERS_DIR)
    return (
        gr.update(choices=choices, value="(None)"),
        gr.update(choices=choices, value="(None)"),
        gr.update(choices=choices, value="(None)"),
    )


def append_section_tag(lyrics: str, tag: str) -> str:
    lyrics = lyrics or ""
    prefix = "\n\n" if lyrics.strip() else ""
    return f"{lyrics.rstrip()}{prefix}{tag}\n"


def save_library_entry(title: str, audio_path: Path, replay_request: dict[str, Any]) -> None:
    meta_path = audio_path.with_suffix(".json")
    entry = {
        "title": title,
        "audio_file": audio_path.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "replay_request": replay_request,
    }
    meta_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")


def list_library_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for meta in sorted(LIBRARY_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            req = data.get("replay_request", {})
            rows.append([
                meta.stem,
                str(data.get("title", meta.stem)),
                str(req.get("style", ""))[:60],
                str(data.get("created_at", "")),
                str(LIBRARY_DIR / data.get("audio_file", "")),
            ])
        except Exception:
            continue
    return rows


def generate_song(
    title: str,
    style: str,
    lyrics: str,
    cot_mode: str,
    abc_score: str,
    semantic_tokens_in: str,
    lora_1: str,
    lora_1_w: float,
    lora_2: str,
    lora_2_w: float,
    lora_3: str,
    lora_3_w: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    output_format: str,
    mp3_bitrate: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    abc_temperature: float,
) -> tuple[str | None, str, str, str, list[list[str]]]:
    adapters: list[dict[str, Any]] = []
    for name, weight in ((lora_1, lora_1_w), (lora_2, lora_2_w), (lora_3, lora_3_w)):
        if name and name != "(None)":
            adapters.append({"path": str(ADAPTERS_DIR / name), "weight": float(weight)})

    payload = build_synth_payload(
        style=style,
        lyrics=lyrics,
        abc=abc_score,
        cot=cot_mode,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        output_format=output_format,
        mp3_bitrate=mp3_bitrate,
        semantic_tokens=semantic_tokens_in,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        abc_temperature=abc_temperature,
        adapters=adapters or None,
    )

    result = client.generate_and_wait(payload)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in (title or "track").strip()) or "track"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    audio_path = LIBRARY_DIR / f"{stamp}_{safe_title}.{result.audio_ext}"
    audio_path.write_bytes(result.audio_bytes)
    save_library_entry(title or safe_title, audio_path, result.replay_request)

    status_msg = f"✅ Selesai / Done! Job `{result.job_id}` | Seed: `{result.seed}` | File: `{audio_path.name}`"
    return (
        str(audio_path),
        result.abc_score,
        result.semantic_tokens,
        status_msg,
        list_library_rows(),
    )


def load_track_for_replay(track_id: str) -> tuple[str, str, str, str, str | None, str]:
    meta_file = LIBRARY_DIR / f"{track_id}.json"
    if not meta_file.exists():
        return "", "", "", "", None, f"⚠️ Track `{track_id}` tidak ditemukan / not found."
    data = json.loads(meta_file.read_text(encoding="utf-8"))
    req = data.get("replay_request", {})
    audio_file = LIBRARY_DIR / data.get("audio_file", "")
    return (
        str(data.get("title", track_id)),
        str(req.get("style", "")),
        str(req.get("lyrics", "")),
        str(req.get("abc", "") or req.get("score", "")),
        str(audio_file) if audio_file.exists() else None,
        str(req.get("semantic_tokens", "")),
    )


def handle_upload_to_dataset(files: list[Any] | None, dataset_name: str) -> str:
    if not files:
        return "⚠️ Pilih minimal 1 file audio (.mp3/.wav/.flac)."
    folder = DATASETS_DIR / (dataset_name.strip() or "my_dataset")
    folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in files:
        src = Path(f.name if hasattr(f, "name") else str(f))
        shutil.copy2(src, folder / src.name)
        copied += 1
    total = len(scan_audio_dataset(folder))
    return f"✅ {copied} file di-upload ke `{folder}`. Total audio: {total}."


def handle_prepare_dataset(dataset_path: str, trigger_word: str, genre: str, language: str) -> str:
    manifest, count = prepare_dataset_manifest(
        dataset_path,
        trigger_word=trigger_word.strip() or "custom_style",
        default_genre=genre.strip() or "pop, vocal",
        language=language.strip() or "en",
    )
    return f"✅ Dataset siap! {count} track diproses -> `{manifest}`"


def list_trained_checkpoints() -> list[str]:
    if not LORA_OUT_DIR.exists():
        return []
    return [
        str(p)
        for p in sorted(LORA_OUT_DIR.rglob("*"), reverse=True)
        if p.is_file() and p.suffix.lower() in {".safetensors", ".gguf"}
    ]


def handle_start_training(
    dataset_path: str,
    adapter_name: str,
    trigger_word: str,
    adapter_type: str,
    rank: int,
    alpha: int,
    lr: float,
    steps: int,
    save_every: int,
    preset: str,
):
    name = (adapter_name or "").strip() or "my_yue2_lora"
    cfg = LoraTrainConfig(
        dataset_dir=(dataset_path or "").strip(),
        output_dir=str(LORA_OUT_DIR),
        adapter_name=name,
        trigger_word=(trigger_word or "").strip() or "custom_style",
        adapter_type=adapter_type,
        rank=int(rank),
        alpha=int(alpha),
        learning_rate=float(lr),
        max_updates=int(steps),
        save_every=int(save_every),
        preset=preset,
        base_model=str(MODELS_DIR / "YuE2-3B-Q8_0.gguf"),
        vae_model=str(MODELS_DIR / "YuE2-Vae-F32.gguf"),
        tokenizer_model=str(MODELS_DIR / "yue2-mothersuperior-realaudio-tokenizer-v4-Q8_0.gguf"),
    )
    buf = ""
    for chunk in stream_training(cfg):
        buf += chunk
        yield buf

    final_ckpt = LORA_OUT_DIR / name / f"{name}.safetensors"
    if final_ckpt.is_file():
        dest = register_trained_adapter(final_ckpt, ADAPTERS_DIR, alias=name)
        buf += f"\n🎉 Otomatis didaftarkan ke LoRA Mixer: `{dest}` (klik 🔄 Refresh Adapters di tab Create)\n"
        yield buf


def handle_register_lora(checkpoint_path: str, alias: str) -> str:
    try:
        dest = register_trained_adapter((checkpoint_path or "").strip(), ADAPTERS_DIR, alias=(alias or "").strip() or None)
        return f"✅ LoRA terdaftar di `{dest}`! Klik 🔄 Refresh Adapters di tab Create untuk menggunakannya."
    except Exception as exc:
        return f"⚠️ Gagal mendaftarkan LoRA: {exc}"


def handle_download_hf_lora(repo_id: str, filename: str) -> str:
    path = hf_hub_download(repo_id=repo_id.strip(), filename=filename.strip(), local_dir=str(ADAPTERS_DIR))
    return f"✅ LoRA berhasil di-download ke `{path}`"


def handle_sync_drive() -> str:
    if not Path("/content/drive/MyDrive").exists():
        return "⚠️ Google Drive belum di-mount. Jalankan Sel 1 di Google Colab dengan opsi Mount Drive."
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    synced: list[str] = []
    for name, src in (("adapters", ADAPTERS_DIR), ("library", LIBRARY_DIR), ("trained_loras", LORA_OUT_DIR)):
        dst = DRIVE_ROOT / name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        synced.append(name)
    return f"✅ Tersinkronisasi ke `{DRIVE_ROOT}`: {', '.join(synced)}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YuE2-Colab-Studio", theme=gr.themes.Soft(primary_hue="pink", neutral_hue="zinc")) as demo:
        gr.Markdown("# 🎶 YuE2-Colab-Studio\n*Dashboard Generasi Lagu & LoRA Training Studio untuk Google Colab (terinspirasi dari YuE2-Studio)*")
        banner = gr.Markdown(get_server_banner())

        with gr.Tabs():
            # ---------------- TAB 1: CREATE ----------------
            with gr.Tab("🎵 Create / Buat Lagu"):
                with gr.Row():
                    with gr.Column(scale=3):
                        title_in = gr.Textbox(label="Track Title / Judul Lagu", value="My First YuE2 Track")
                        style_in = gr.Textbox(
                            label="Style Prompt (Genre, Vocal, Instruments, Mood, BPM)",
                            placeholder="indie pop, warm female vocal, acoustic guitar, dreamy synth, 110 bpm",
                            lines=2,
                        )
                        with gr.Row():
                            btn_verse = gr.Button("+ [verse]", size="sm")
                            btn_chorus = gr.Button("+ [chorus]", size="sm")
                            btn_bridge = gr.Button("+ [bridge]", size="sm")
                            btn_outro = gr.Button("+ [outro]", size="sm")
                        lyrics_in = gr.Textbox(
                            label="Lyrics / Lirik (Gunakan tag bagian seperti [verse], [chorus])",
                            value="[verse]\nWalking down the neon street tonight\nCity lights are shining bright\n\n[chorus]\nWe are singing to the stars above\nSpinning in a melody of love",
                            lines=9,
                        )
                        with gr.Accordion("🎼 ABC Score & Replay Tokens (Opsional / Optional)", open=False):
                            cot_mode = gr.Radio(["auto", "melody", "off"], value="auto", label="Score Mode (CoT)")
                            abc_in = gr.Textbox(label="ABC Score (diisi otomatis atau edit manual)", lines=5)
                            semantic_in = gr.Textbox(label="Semantic Tokens (untuk Exact Replay / Re-render cepat)", lines=2)

                    with gr.Column(scale=2):
                        gr.Markdown("### 🧬 LoRA / LoKr Mixer")
                        btn_refresh_lora = gr.Button("🔄 Refresh Adapters", size="sm")
                        adapter_choices = ["(None)"] + list_adapters(ADAPTERS_DIR)
                        with gr.Row():
                            lora_1 = gr.Dropdown(adapter_choices, value="(None)", label="LoRA 1", scale=2)
                            lora_1_w = gr.Slider(-10.0, 10.0, value=1.0, step=0.1, label="Weight 1", scale=1)
                        with gr.Row():
                            lora_2 = gr.Dropdown(adapter_choices, value="(None)", label="LoRA 2", scale=2)
                            lora_2_w = gr.Slider(-10.0, 10.0, value=1.0, step=0.1, label="Weight 2", scale=1)
                        with gr.Row():
                            lora_3 = gr.Dropdown(adapter_choices, value="(None)", label="LoRA 3", scale=2)
                            lora_3_w = gr.Slider(-10.0, 10.0, value=1.0, step=0.1, label="Weight 3", scale=1)

                        with gr.Accordion("⚙️ Engine Settings", open=False):
                            steps_in = gr.Slider(4, 100, value=32, step=1, label="Flow-Matching Steps")
                            cfg_in = gr.Slider(-1.0, 10.0, value=-1.0, step=0.5, label="CFG Scale (-1 = auto)")
                            seed_in = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                            fmt_in = gr.Dropdown(["mp3", "wav16", "wav24", "wav32"], value="mp3", label="Output Format")
                            bitrate_in = gr.Dropdown([128, 192, 256, 320], value=192, label="MP3 Bitrate (kbps)")
                            temp_in = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Semantic Temperature")
                            top_p_in = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P")
                            top_k_in = gr.Slider(1, 200, value=100, step=1, label="Top-K")
                            rep_in = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition Penalty")
                            abc_temp_in = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="ABC Score Temperature")

                        btn_generate = gr.Button("🚀 Generate Song / Buat Lagu", variant="primary", size="lg")
                        gen_status = gr.Markdown("")
                        audio_out = gr.Audio(label="Generated Track / Hasil Lagu", type="filepath")

                btn_verse.click(lambda l: append_section_tag(l, "[verse]"), inputs=lyrics_in, outputs=lyrics_in)
                btn_chorus.click(lambda l: append_section_tag(l, "[chorus]"), inputs=lyrics_in, outputs=lyrics_in)
                btn_bridge.click(lambda l: append_section_tag(l, "[bridge]"), inputs=lyrics_in, outputs=lyrics_in)
                btn_outro.click(lambda l: append_section_tag(l, "[outro]"), inputs=lyrics_in, outputs=lyrics_in)
                btn_refresh_lora.click(refresh_adapter_choices, outputs=[lora_1, lora_2, lora_3])

            # ---------------- TAB 2: LIBRARY & REPLAY ----------------
            with gr.Tab("📚 Library & Replay"):
                gr.Markdown("Daftar lagu yang telah dibuat di sesi ini. Masukkan **Track ID** untuk memuat ulang lirik, score, dan `semantic_tokens` (Exact Replay).")
                btn_refresh_lib = gr.Button("🔄 Refresh Library", size="sm")
                lib_table = gr.Dataframe(
                    headers=["Track ID", "Title", "Style", "Created At", "Audio Path"],
                    value=list_library_rows(),
                    interactive=False,
                )
                with gr.Row():
                    replay_id_in = gr.Textbox(label="Track ID untuk Replay", placeholder="20261127_183000_My_First_YuE2_Track")
                    btn_load_replay = gr.Button("♻️ Load into Create Tab (Exact Replay)")
                replay_audio_preview = gr.Audio(label="Preview Track", type="filepath")

                btn_refresh_lib.click(list_library_rows, outputs=lib_table)
                btn_load_replay.click(
                    load_track_for_replay,
                    inputs=replay_id_in,
                    outputs=[title_in, style_in, lyrics_in, abc_in, replay_audio_preview, semantic_in],
                )

            # ---------------- TAB 3: LORA TRAINING STUDIO ----------------
            with gr.Tab("🧬 LoRA Training Studio"):
                gr.Markdown(
                    "### Latih Model LoRA / LoKr Sendiri (HOT-Step-CPP YuE2 Joint Training)\n"
                    "1. Upload kumpulan lagu (`.mp3`/`.wav`/`.flac`) atau arahkan ke folder dataset di Google Drive.\n"
                    "2. Klik **Prepare Dataset** untuk membuat `dataset.json` & caption `.yue2.txt`.\n"
                    "3. Atur resep training (AR + NAR Joint Training) lalu klik **Start Training**."
                )
                with gr.Row():
                    with gr.Column():
                        ds_name = gr.Textbox(label="Dataset Name / Nama Dataset", value="my_artist_dataset")
                        ds_files = gr.File(label="Upload Audio & Lyrics (.mp3/.wav/.flac + .txt opsional)", file_count="multiple")
                        btn_upload_ds = gr.Button("📤 Upload ke Folder Dataset")
                        ds_path_in = gr.Textbox(label="Dataset Folder Path", value=str(DATASETS_DIR / "my_artist_dataset"))
                        trigger_in = gr.Textbox(label="Trigger Word (Kata Kunci Gaya/Artis)", value="my_artist_style")
                        ds_genre_in = gr.Textbox(label="Default Genre / Tags", value="indie pop, expressive vocal")
                        ds_lang_in = gr.Dropdown(["en", "id", "ja", "zh", "ko", "ru"], value="en", label="Language / Bahasa")
                        btn_prep_ds = gr.Button("📋 Prepare Dataset (dataset.json)", variant="secondary")
                        ds_status = gr.Markdown("")

                    with gr.Column():
                        train_name = gr.Textbox(label="Output LoRA Name", value="my_artist_lokr")
                        with gr.Row():
                            train_preset = gr.Dropdown(["Fast", "Balanced", "Thorough"], value="Fast", label="Preset")
                            train_type = gr.Dropdown(["lokr", "lora"], value="lokr", label="Adapter Type")
                        with gr.Row():
                            train_rank = gr.Slider(16, 256, value=32, step=16, label="Rank (Dim)")
                            train_alpha = gr.Slider(16, 256, value=32, step=16, label="Alpha")
                        with gr.Row():
                            train_lr = gr.Number(value=1e-4, label="Learning Rate")
                            train_steps = gr.Number(value=100, precision=0, label="Max Updates / Steps")
                            train_save_every = gr.Number(value=10, precision=0, label="Save Every")
                        btn_start_train = gr.Button("🔥 Start LoRA Training", variant="primary")
                        train_logs = gr.Textbox(label="Live Training Logs", lines=12, max_lines=20)

                with gr.Row():
                    ckpt_path_in = gr.Dropdown(
                        choices=list_trained_checkpoints(),
                        allow_custom_value=True,
                        label="Pilih / Isi Path Checkpoint (.safetensors / .gguf)",
                        scale=3,
                    )
                    btn_scan_ckpts = gr.Button("🔄 Scan Checkpoints", scale=1)
                    ckpt_alias_in = gr.Textbox(label="Nama Alias di Tab Create", value="my_artist_lokr", scale=2)
                    btn_reg_lora = gr.Button("➕ Daftarkan ke Create Tab", scale=1)
                reg_status = gr.Markdown("")

                btn_scan_ckpts.click(lambda: gr.update(choices=list_trained_checkpoints()), outputs=ckpt_path_in)

                btn_upload_ds.click(handle_upload_to_dataset, inputs=[ds_files, ds_name], outputs=ds_status)
                btn_prep_ds.click(handle_prepare_dataset, inputs=[ds_path_in, trigger_in, ds_genre_in, ds_lang_in], outputs=ds_status)
                btn_start_train.click(
                    handle_start_training,
                    inputs=[ds_path_in, train_name, trigger_in, train_type, train_rank, train_alpha, train_lr, train_steps, train_save_every, train_preset],
                    outputs=train_logs,
                )
                btn_reg_lora.click(handle_register_lora, inputs=[ckpt_path_in, ckpt_alias_in], outputs=reg_status)

            # ---------------- TAB 4: MODELS & STORAGE ----------------
            with gr.Tab("⚙️ Models & Storage"):
                gpu_md = gr.Markdown(get_gpu_status())
                btn_refresh_sys = gr.Button("🔄 Refresh Status & VRAM", size="sm")
                gr.Markdown("### Download Community LoRA dari Hugging Face")
                with gr.Row():
                    hf_repo_in = gr.Textbox(label="Hugging Face Repo ID", placeholder="username/yue2-lora-repo")
                    hf_file_in = gr.Textbox(label="Filename", placeholder="adapter.gguf")
                    btn_dl_hf = gr.Button("⬇️ Download LoRA")
                hf_status = gr.Markdown("")

                gr.Markdown("### Google Drive Backup & Sync")
                btn_sync_drive = gr.Button("☁️ Sync Adapters, Library & Trained LoRAs ke Google Drive")
                drive_status = gr.Markdown("")

                btn_refresh_sys.click(lambda: (get_server_banner(), get_gpu_status()), outputs=[banner, gpu_md])
                btn_dl_hf.click(handle_download_hf_lora, inputs=[hf_repo_in, hf_file_in], outputs=hf_status)
                btn_sync_drive.click(handle_sync_drive, outputs=drive_status)

        btn_generate.click(
            generate_song,
            inputs=[
                title_in, style_in, lyrics_in, cot_mode, abc_in, semantic_in,
                lora_1, lora_1_w, lora_2, lora_2_w, lora_3, lora_3_w,
                steps_in, cfg_in, seed_in, fmt_in, bitrate_in,
                temp_in, top_p_in, top_k_in, rep_in, abc_temp_in,
            ],
            outputs=[audio_out, abc_in, semantic_in, gen_status, lib_table],
        )

        demo.load(get_server_banner, outputs=banner)

    return demo


def main() -> None:
    share = default_gradio_share(PLATFORM)
    port = int(os.environ.get("GRADIO_PORT", os.environ.get("PORT", "7860")))
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    main()
