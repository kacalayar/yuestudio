# YuE2-Studio (Colab, Kaggle, VPS & Hugging Face Spaces)

Dashboard Web **Gradio** (Bilingual ID/EN) + **LoRA Training Studio** untuk menjalankan **YuE2** di **Google Colab**, **Kaggle**, **VPS Linux CUDA**, maupun **Hugging Face Spaces**, terinspirasi dari [`YuE2-Studio`](https://github.com/kacalayar/YuE2-Studio).

## Arsitektur
- **Inference Engine:** [`timoncool/yue2.cpp`](https://github.com/timoncool/yue2.cpp) (`yue-server` dengan CUDA `GGML_CUDA=ON` & model `YuE2-3B-Q8_0.gguf`).
- **LoRA Trainer:** [`timoncool/HOT-Step-CPP`](https://github.com/timoncool/HOT-Step-CPP) (`ace-train` CUDA untuk YuE2 Joint Training AR + NAR).
- **Dashboard:** Gradio (`dashboard/app.py`) dengan auto-deteksi platform (`colab`, `kaggle`, `huggingface`, `vps`) + tab **Create + LoRA Mixer**, **Library & Exact Replay**, **LoRA Training Studio**, dan **Models & Storage**.

---

## Cara Menjalankan di Masing-Masing Platform

### 1. Google Colab (GPU T4 16 GB)
1. Upload folder `yuestudio/` ke Google Drive di `MyDrive/YuE2-Studio/yuestudio/`.
2. Buka [YuE2_Colab_Studio.ipynb](YuE2_Colab_Studio.ipynb) di Google Colab, pilih **Runtime → Change runtime type → T4 GPU**.
3. Jalankan **Sel 1 s/d Sel 5** (sesi pertama compile CUDA ~5–10 menit dan otomatis di-cache ke Google Drive).
4. Buka link publik `https://xxxx.gradio.live`. *(Opsional: Sel 6 membuka WebUI bawaan `yue2.cpp` via `cloudflared`)*

### 2. Kaggle (GPU T4 x2 / P100)
1. Upload folder `yuestudio/` sebagai **Kaggle Dataset** bernama `yuestudio` (atau upload langsung ke `/kaggle/working/yuestudio`).
2. Import [YuE2_Colab_Studio.ipynb](YuE2_Colab_Studio.ipynb) ke Kaggle Notebook, aktifkan **Accelerator: GPU T4** & **Internet: On**.
3. Jalankan **Sel 1 s/d Sel 5** — notebook otomatis mendeteksi `/kaggle/working` dan meluncurkan link Gradio publik.

### 3. VPS Sendiri (Linux + NVIDIA GPU)
**Opsi A — Docker Compose (Disarankan):**
```bash
docker compose up -d --build
```
Akses dashboard di `http://<IP-VPS>:7860` (dan API `yue-server` di port `8087`). Model GGUF otomatis diunduh saat boot pertama ke `./yue2-workspace/models`.

**Opsi B — Bare-metal Linux CUDA:**
```bash
chmod +x scripts/setup_linux.sh scripts/start.sh
./scripts/setup_linux.sh
./scripts/start.sh
```
*(Set `GRADIO_SHARE=1` sebelum `./scripts/start.sh` jika ingin link publik Gradio tanpa membuka port firewall).*

### 4. Hugging Face Spaces (Docker GPU)
1. Buat Space baru di Hugging Face dengan **SDK: Docker** dan pilih hardware GPU (minimal **T4 small / medium**).
2. Push seluruh isi folder `yuestudio/` (termasuk `Dockerfile`, `dashboard/`, `scripts/`) ke repository Space.
3. HF Spaces otomatis build `Dockerfile` dan menjalankan `scripts/start.sh` pada port `7860` (`SPACE_ID` terdeteksi otomatis sehingga `share=False` dipakai secara tepat).

---

## Cara Membuat LoRA Model Sendiri
1. Buka tab **🧬 LoRA Training Studio** di dashboard.
2. Upload kumpulan lagu (`.mp3`/`.wav`/`.flac`, beserta `.txt` lirik jika ada) atau arahkan ke folder dataset.
3. Isi **Trigger Word**, lalu klik **📋 Prepare Dataset** untuk menghasilkan `dataset.json` dan sidecar `.yue2.txt`.
4. Pilih preset (`Fast` / `Balanced` / `Thorough`), tipe (`lokr` / `lora`), lalu klik **🔥 Start LoRA Training**.
5. Setelah selesai, klik **➕ Daftarkan ke Create Tab** agar LoRA langsung muncul di **LoRA Mixer** pada tab **Create**.
