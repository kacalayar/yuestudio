"""HTTP client for timoncool/yue2.cpp `yue-server` API."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

SUPPORTED_ADAPTER_EXTS = {".gguf", ".safetensors", ".bin", ".pt"}


@dataclass
class SynthResult:
    """Parsed result from GET /job?id=N&result=1."""

    job_id: str
    audio_bytes: bytes
    audio_ext: str
    replay_request: dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_tokens(self) -> str:
        return str(self.replay_request.get("semantic_tokens", ""))

    @property
    def abc_score(self) -> str:
        return str(self.replay_request.get("abc", "") or self.replay_request.get("score", ""))

    @property
    def seed(self) -> Any:
        return self.replay_request.get("seed")


def parse_multipart_mixed(content_type: str, body: bytes) -> tuple[dict[str, Any], bytes, str]:
    """Parse a multipart/mixed response into (replay_json, audio_bytes, audio_ext)."""
    match = re.search(r'boundary="?([^";\s]+)"?', content_type or "")
    if not match:
        raise ValueError(f"Missing multipart boundary in Content-Type: {content_type!r}")

    boundary = match.group(1).encode("utf-8")
    delimiter = b"--" + boundary
    raw_parts = body.split(delimiter)

    replay_json: dict[str, Any] = {}
    audio_bytes = b""
    audio_ext = "mp3"

    for part in raw_parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].rstrip(b"\r\n")

        sep = b"\r\n\r\n" if b"\r\n\r\n" in part else b"\n\n"
        if sep not in part:
            continue
        header_raw, payload = part.split(sep, 1)
        headers_text = header_raw.decode("utf-8", errors="replace").lower()

        if "application/json" in headers_text:
            replay_json = json.loads(payload.decode("utf-8"))
        elif "audio/" in headers_text or "application/octet-stream" in headers_text:
            audio_bytes = payload
            if "wav" in headers_text:
                audio_ext = "wav"
            elif "flac" in headers_text:
                audio_ext = "flac"
            else:
                audio_ext = "mp3"

    if not audio_bytes:
        raise ValueError("No audio payload found in multipart/mixed response")

    return replay_json, audio_bytes, audio_ext


def list_adapters(adapters_dir: str | Path) -> list[str]:
    """Scan adapters directory for LoRA/LoKr adapter files."""
    root = Path(adapters_dir)
    if not root.exists():
        return []
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_ADAPTER_EXTS:
            found.append(str(path.relative_to(root)).replace("\\", "/"))
    return found


def build_synth_payload(
    style: str,
    lyrics: str,
    *,
    abc: str = "",
    cot: str = "auto",
    steps: int = 32,
    cfg_scale: float = -1.0,
    seed: int = -1,
    output_format: str = "mp3",
    mp3_bitrate: int = 192,
    semantic_tokens: str = "",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 100,
    repetition_penalty: float = 1.2,
    abc_temperature: float = 0.7,
    adapters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Yue2Request JSON payload compatible with yue2.cpp yue-server."""
    if not (style or "").strip() and not (lyrics or "").strip() and not (semantic_tokens or "").strip():
        raise ValueError("Style atau lyrics tidak boleh kosong semua / Style or lyrics cannot both be empty.")

    payload: dict[str, Any] = {
        "style": style.strip(),
        "lyrics": lyrics.strip(),
        "steps": max(1, int(steps)),
        "cfg_scale": float(cfg_scale),
        "output_format": output_format,
        "mp3_bitrate": int(mp3_bitrate),
        "abc_sampling": {
            "temperature": float(abc_temperature),
            "top_p": 0.9,
            "top_k": 30,
            "repetition_penalty": 1.005,
            "penalty_window": 100,
            "min_tokens": 32,
            "max_tokens": 4096,
        },
        "semantic_sampling": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "penalty_window": 50,
            "min_tokens": 200,
            "max_tokens": 9000,
        },
    }
    if seed >= 0:
        payload["seed"] = int(seed)
    if cot and cot != "auto":
        payload["cot"] = cot
    if (abc or "").strip():
        payload["abc"] = abc.strip()
    if (semantic_tokens or "").strip():
        payload["semantic_tokens"] = semantic_tokens.strip()
    if adapters:
        payload["lm_adapter"] = [
            {"path": str(a["path"]), "scale": float(a.get("scale", a.get("weight", 1.0)))}
            for a in adapters
        ]
    else:
        payload["lm_adapter"] = None
    return payload


class YueServerClient:
    """Thin wrapper around HOT-Step-CPP ace-server (/yue2/*) REST API."""

    def __init__(self, base_url: str | None = None, timeout: float = 15.0) -> None:
        self.base_url = (base_url or os.environ.get("YUE_SERVER_URL", "http://127.0.0.1:8087")).rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_online(self) -> bool:
        try:
            data = self.health()
            return data.get("status") == "ok"
        except Exception:
            return False

    def get_props(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/yue2/props", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def submit_synth(self, payload: dict[str, Any]) -> str:
        resp = requests.post(f"{self.base_url}/yue2/synth", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data["id"])

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/job", params={"id": job_id}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def cancel_job(self, job_id: str) -> None:
        requests.post(f"{self.base_url}/job", params={"id": job_id, "cancel": "1"}, timeout=self.timeout)

    def fetch_result(self, job_id: str) -> SynthResult:
        resp = requests.get(
            f"{self.base_url}/job",
            params={"id": job_id, "result": "1"},
            timeout=max(self.timeout, 120.0),
        )
        resp.raise_for_status()
        replay_json, audio_bytes, audio_ext = parse_multipart_mixed(
            resp.headers.get("Content-Type", ""),
            resp.content,
        )
        return SynthResult(
            job_id=job_id,
            audio_bytes=audio_bytes,
            audio_ext=audio_ext,
            replay_request=replay_json,
        )

    def generate_and_wait(
        self,
        payload: dict[str, Any],
        poll_interval: float = 1.5,
        on_status: Callable[[str], None] | None = None,
    ) -> SynthResult:
        job_id = self.submit_synth(payload)
        while True:
            info = self.get_job_status(job_id)
            status = info.get("status", "unknown")
            if on_status:
                on_status(status)
            if status == "done":
                return self.fetch_result(job_id)
            if status in {"failed", "cancelled"}:
                detail = info.get("error") or status
                raise RuntimeError(f"Job {job_id} ended with status={detail}")
            time.sleep(poll_interval)
