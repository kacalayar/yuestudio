"""Environment auto-detection for Colab, Kaggle, Hugging Face Spaces, and VPS/Local."""

from __future__ import annotations

import os
from pathlib import Path


def detect_platform() -> str:
    """Detect runtime platform: 'colab', 'kaggle', 'huggingface', or 'vps'."""
    if os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE"):
        return "huggingface"
    if Path("/kaggle/working").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if Path("/content").exists() or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "vps"


def default_workspace_root(platform: str | None = None) -> Path:
    """Return the default workspace directory for the current platform."""
    explicit = os.environ.get("YUE_WORKSPACE") or os.environ.get("YUE_COLAB_ROOT")
    if explicit:
        return Path(explicit)

    plat = platform or detect_platform()
    if plat == "kaggle":
        return Path("/kaggle/working/yue2-workspace")
    if plat == "colab":
        return Path("/content/yue2-workspace")
    if plat == "huggingface":
        if Path("/data").exists() and os.access("/data", os.W_OK):
            return Path("/data/yue2-workspace")
        return Path("/app/yue2-workspace")
    return Path.cwd() / "yue2-workspace"


def default_gradio_share(platform: str | None = None) -> bool:
    """Enable Gradio public share link by default only on Colab and Kaggle."""
    explicit = os.environ.get("GRADIO_SHARE")
    if explicit is not None:
        return explicit.strip() in {"1", "true", "True", "yes"}
    plat = platform or detect_platform()
    return plat in {"colab", "kaggle"}
