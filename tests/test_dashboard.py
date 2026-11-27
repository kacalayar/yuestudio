"""Unit tests for YuE2-Colab-Studio dashboard modules and Colab notebook."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.env import default_gradio_share, default_workspace_root, detect_platform
from dashboard.trainer import (  # noqa: E402
    LoraTrainConfig,
    build_train_command,
    prepare_dataset_manifest,
    register_trained_adapter,
    scan_audio_dataset,
)
from dashboard.yue_client import (  # noqa: E402
    build_synth_payload,
    list_adapters,
    parse_multipart_mixed,
)


class TestYueClient(unittest.TestCase):
    def test_parse_multipart_mixed(self) -> None:
        boundary = "yueboundary123"
        body = (
            f"--{boundary}\r\n"
            "Content-Type: application/json\r\n\r\n"
            '{"seed": 42, "semantic_tokens": "1,2,3", "abc": "X:1\\nK:C\\nC D E F"}\r\n'
            f"--{boundary}\r\n"
            "Content-Type: audio/mpeg\r\n\r\n"
            "FAKE_MP3_DATA"
            f"\r\n--{boundary}--\r\n"
        ).encode("utf-8")

        replay, audio, ext = parse_multipart_mixed(f'multipart/mixed; boundary="{boundary}"', body)
        self.assertEqual(replay["seed"], 42)
        self.assertEqual(replay["semantic_tokens"], "1,2,3")
        self.assertEqual(audio, b"FAKE_MP3_DATA")
        self.assertEqual(ext, "mp3")

    def test_build_synth_payload_with_adapters(self) -> None:
        payload = build_synth_payload(
            style="indie rock",
            lyrics="[verse]\nhello world",
            steps=24,
            seed=1234,
            adapters=[{"path": "/content/adapters/my.gguf", "weight": 0.8}],
        )
        self.assertEqual(payload["style"], "indie rock")
        self.assertEqual(payload["steps"], 24)
        self.assertEqual(payload["seed"], 1234)
        self.assertEqual(payload["adapters"][0]["weight"], 0.8)

    def test_build_synth_payload_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            build_synth_payload(style="", lyrics="")

    def test_list_adapters(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "rock.gguf").write_bytes(b"0")
            (p / "sub").mkdir()
            (p / "sub" / "vocal.safetensors").write_bytes(b"0")
            (p / "ignore.txt").write_text("hi", encoding="utf-8")
            found = list_adapters(p)
            self.assertEqual(found, ["rock.gguf", "sub/vocal.safetensors"])


class TestTrainer(unittest.TestCase):
    def test_prepare_dataset_manifest_and_register(self) -> None:
        with TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds"
            ds.mkdir()
            (ds / "song1.mp3").write_bytes(b"audio1")
            (ds / "song1.txt").write_text("[verse]\nla la", encoding="utf-8")
            (ds / "song2.wav").write_bytes(b"audio2")

            self.assertEqual(len(scan_audio_dataset(ds)), 2)
            manifest, count = prepare_dataset_manifest(ds, trigger_word="tok_style", default_genre="pop", language="id")
            self.assertEqual(count, 2)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(data["tracks"]), 2)
            self.assertTrue((ds / "song1.yue2.txt").exists())

            ckpt = Path(tmp) / "ckpt.gguf"
            ckpt.write_bytes(b"weights")
            adapters_dir = Path(tmp) / "adapters"
            dest = register_trained_adapter(ckpt, adapters_dir, alias="my_custom")
            self.assertTrue(dest.exists())
            self.assertEqual(dest.name, "my_custom.gguf")

    def test_build_train_command(self) -> None:
        cfg = LoraTrainConfig(
            dataset_dir="/content/ds",
            output_dir="/content/out",
            adapter_name="test_lora",
            trainer_bin="/usr/bin/ace-train",
        )
        cmd = build_train_command(cfg)
        self.assertEqual(cmd[0], "/usr/bin/ace-train")
        self.assertIn("--backend", cmd)
        self.assertIn("yue2", cmd)


class TestNotebookValidJson(unittest.TestCase):
    def test_notebook_structure(self) -> None:
        nb_path = ROOT / "YuE2_Colab_Studio.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(len(nb["cells"]), 6)


class TestEnvDetection(unittest.TestCase):
    def test_platform_and_share_defaults(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"SPACE_ID": "user/yue2-studio"}, clear=False):
            self.assertEqual(detect_platform(), "huggingface")
            self.assertFalse(default_gradio_share("huggingface"))

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRADIO_SHARE", None)
            self.assertTrue(default_gradio_share("colab"))
            self.assertTrue(default_gradio_share("kaggle"))
            self.assertFalse(default_gradio_share("vps"))
            self.assertEqual(default_workspace_root("kaggle"), Path("/kaggle/working/yue2-workspace"))


if __name__ == "__main__":
    unittest.main()
