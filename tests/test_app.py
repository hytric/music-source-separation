"""Input, path, output isolation and launch-setting regression checks."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from inference import app
from inference.runtime import SeparationResult


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.audio = self.data / "song.wav"
        self.audio.touch()
        self.output = self.root / "output"
        self.output.mkdir()
        self.patch = patch.multiple(app, DATA_DIR=self.data, OUTPUT_DIR=self.output)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_dataset_traversal_and_absolute_paths_are_rejected(self):
        outside = self.root / "private.wav"
        outside.touch()
        for path in ["../private.wav", str(outside)]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                app._resolve(path)

    def test_symlink_outside_dataset_is_not_listed(self):
        outside = self.root / "private.wav"
        outside.touch()
        try:
            (self.data / "link.wav").symlink_to(outside)
        except OSError:
            self.skipTest("Symlinks unavailable")
        self.assertEqual(app.list_audio(), ["song.wav"])
        with self.assertRaises(ValueError):
            app._resolve("link.wav")

    def test_external_dataset_preview_stages_only_selected_audio(self):
        self.audio.write_bytes(b"selected audio")
        with patch.object(app, "validate_audio"), patch.object(app, "spectrogram_image"):
            preview, _ = app.on_select("song.wav")
        self.assertTrue(Path(preview).is_relative_to(self.output))
        self.assertEqual(Path(preview).read_bytes(), self.audio.read_bytes())

    def test_oversize_and_overlong_audio(self):
        with patch.object(app, "MAX_MB", 1 / 1024 ** 2):
            self.audio.write_bytes(b"too big")
            with self.assertRaisesRegex(ValueError, "크기"):
                app.validate_audio(str(self.audio))
        with patch.object(app.librosa, "get_duration", return_value=app.MAX_SECONDS + 1):
            with self.assertRaisesRegex(ValueError, "길이"):
                app.validate_audio(str(self.audio))

    def test_default_host_and_false_flags(self):
        with patch.dict(os.environ, {"MSS_SHARE": "0"}, clear=True):
            options = app.launch_options()
        self.assertEqual(options["server_name"], "127.0.0.1")
        self.assertFalse(options["share"])
        self.assertEqual(options["allowed_paths"], [str(self.output)])

    def test_invalid_auth_is_rejected(self):
        with patch.dict(os.environ, {"MSS_AUTH": "invalid"}, clear=True):
            with self.assertRaises(ValueError):
                app.launch_options()

    def test_per_request_outputs_and_actual_cpu_status(self):
        jobs = []
        def run(path, output, model, notify):
            jobs.append(output)
            for name in ["song_(Vocals).wav", "song_(Instrumental).wav"]:
                (output / name).touch()
            return SeparationResult(
                ["song_(Vocals).wav", "song_(Instrumental).wav"], 1.0, "CPU", ["GPU 없음: CPU 사용"],
            )
        with patch.object(app, "validate_audio"), patch.object(app, "spectrogram_image"), \
             patch.object(app.RUNTIME, "separate", side_effect=run):
            first = app.separate(str(self.audio), None, next(iter(app.MODELS)))
            second = app.separate(str(self.audio), None, next(iter(app.MODELS)))
        self.assertNotEqual(jobs[0], jobs[1])
        self.assertIn("실제 장치 CPU", first[-1])
        self.assertIn("GPU 없음", second[-1])

    def test_error_does_not_expose_server_paths(self):
        with patch.object(app, "validate_audio"), \
             patch.object(app.RUNTIME, "separate", side_effect=RuntimeError("sensitive server details")):
            with self.assertRaises(gr.Error) as caught:
                app.separate(str(self.audio), None, next(iter(app.MODELS)))
        self.assertNotIn("sensitive server details", str(caught.exception))
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cleanup_only_removes_expired_web_jobs(self):
        old = self.output / "job-old"
        old.mkdir()
        fresh = self.output / "job-fresh"
        fresh.mkdir()
        os.utime(old, (1, 1))
        app.cleanup_outputs()
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(self.audio.exists())


if __name__ == "__main__":
    unittest.main()
