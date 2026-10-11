"""Regression checks for hardware selection and one-time CPU recovery."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from inference.runtime import SeparationRuntime, check_cuda_provider, make_separator, select_device
from inference.settings import env_bool


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.mem_get_info.return_value = (8 * 1024 ** 3, 12 * 1024 ** 3)
        self.ort = MagicMock()
        self.ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.modules = patch.dict("sys.modules", {"torch": self.torch, "onnxruntime": self.ort})
        self.modules.start()
        self.environment = patch.dict(os.environ, {"MSS_USE_GPU": "1"}, clear=True)
        self.environment.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(self.environment.stop)

    def test_explicit_zero_does_not_probe_cuda(self):
        os.environ["MSS_USE_GPU"] = "0"
        self.assertEqual(select_device("model.ckpt"), ("cpu", None))
        self.torch.cuda.is_available.assert_not_called()

    def test_invalid_boolean_is_rejected(self):
        os.environ["MSS_USE_GPU"] = "maybe"
        with self.assertRaises(ValueError):
            env_bool("MSS_USE_GPU")

    def test_no_gpu(self):
        self.torch.cuda.is_available.return_value = False
        device, notice = select_device("model.ckpt")
        self.assertEqual(device, "cpu")
        self.assertIn("GPU", notice)

    def test_insufficient_free_vram(self):
        self.torch.cuda.mem_get_info.return_value = (1 * 1024 ** 3, 12 * 1024 ** 3)
        device, notice = select_device("model.ckpt")
        self.assertEqual(device, "cpu")
        self.assertIn("VRAM", notice)

    def test_onnx_without_cuda_provider(self):
        self.ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        device, notice = select_device("model.onnx")
        self.assertEqual(device, "cpu")
        self.assertIn("ONNX", notice)

    def test_sufficient_free_vram(self):
        self.assertEqual(select_device("model.ckpt"), ("cuda", None))

    def test_cuda_initialization_error(self):
        self.torch.cuda.mem_get_info.side_effect = RuntimeError("CUDA driver failed")
        self.assertEqual(select_device("model.ckpt")[0], "cpu")

    def test_bad_vram_threshold_is_not_silently_ignored(self):
        os.environ["MSS_MIN_VRAM_GB"] = "nan"
        with self.assertRaises(ValueError):
            select_device("model.ckpt")

    def test_force_cpu_overrides_automatic_acceleration(self):
        class AutoSeparator:
            def __init__(self, **kwargs):
                self.setup_torch_device(None)

        self.torch.device.side_effect = lambda value: SimpleNamespace(type=value)
        with patch.dict("sys.modules", {"audio_separator.separator": SimpleNamespace(Separator=AutoSeparator)}):
            separator = make_separator("cpu", Path("unused"))
        self.assertEqual(separator.torch_device.type, "cpu")
        self.assertEqual(separator.onnx_execution_provider, ["CPUExecutionProvider"])
        self.torch.cuda.is_available.assert_not_called()

    def test_silent_onnx_provider_fallback_is_detected(self):
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        def model_run(data):
            return session.get_providers()
        separator = SimpleNamespace(model_instance=SimpleNamespace(
            uses_pytorch_inference=False, model_run=model_run,
        ))
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            check_cuda_provider(separator, "model.onnx")


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.input = Path(self.temp.name) / "input.wav"
        self.input.touch()
        self.output = Path(self.temp.name) / "out"
        self.runtime = SeparationRuntime()
        self.torch = MagicMock()
        self.modules = patch.dict("sys.modules", {"torch": self.torch})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def separator(self):
        separator = MagicMock()
        separator.separate.return_value = ["vocals.wav", "instrumental.wav"]
        return separator

    def recover(self, error_at):
        gpu, cpu = self.separator(), self.separator()
        getattr(gpu, error_at).side_effect = RuntimeError("CUDA out of memory")
        notices = []
        with patch("inference.runtime.select_device", return_value=("cuda", None)), \
             patch("inference.runtime.make_separator", side_effect=[gpu, cpu]) as factory:
            result = self.runtime.separate(self.input, self.output, "model.ckpt", notices.append)
        self.assertEqual(result.device, "CPU")
        self.assertEqual([call.args[0] for call in factory.call_args_list], ["cuda", "cpu"])
        self.assertEqual(len(notices), 1)
        self.torch.cuda.empty_cache.assert_called_once()
        self.assertIn("model.ckpt", self.runtime._cpu_models)

    def test_oom_during_model_load(self):
        self.recover("load_model")

    def test_oom_during_inference(self):
        self.recover("separate")

    def test_non_gpu_error_does_not_retry(self):
        separator = self.separator()
        separator.separate.side_effect = ValueError("bad input")
        with patch("inference.runtime.select_device", return_value=("cuda", None)), \
             patch("inference.runtime.make_separator", return_value=separator) as factory:
            with self.assertRaisesRegex(ValueError, "bad input"):
                self.runtime.separate(self.input, self.output, "model.ckpt")
        self.assertEqual(factory.call_count, 1)
        self.assertIsNone(self.runtime._separator)

    def test_cpu_memory_error_does_not_retry(self):
        separator = self.separator()
        separator.separate.side_effect = RuntimeError("out of memory")
        with patch("inference.runtime.select_device", return_value=("cpu", None)), \
             patch("inference.runtime.make_separator", return_value=separator) as factory:
            with self.assertRaises(RuntimeError):
                self.runtime.separate(self.input, self.output, "model.ckpt")
        self.assertEqual(factory.call_count, 1)

    def test_notification_and_device_when_gpu_missing(self):
        with patch("inference.runtime.select_device", return_value=("cpu", "GPU 없음: CPU 사용")), \
             patch("inference.runtime.make_separator", return_value=self.separator()):
            result = self.runtime.separate(self.input, self.output, "model.ckpt")
        self.assertEqual(result.device, "CPU")
        self.assertEqual(result.notices, ["GPU 없음: CPU 사용"])

    def test_missing_file_fails_before_model_load(self):
        with patch("inference.runtime.make_separator") as factory:
            with self.assertRaises(FileNotFoundError):
                self.runtime.separate(self.input.with_name("missing.wav"), self.output, "model.ckpt")
        factory.assert_not_called()

    def test_only_one_model_stays_cached(self):
        with patch("inference.runtime.select_device", return_value=("cpu", None)), \
             patch("inference.runtime.make_separator", side_effect=[self.separator(), self.separator()]) as factory:
            self.runtime.separate(self.input, self.output, "a.ckpt")
            self.runtime.separate(self.input, self.output, "a.ckpt")
            self.runtime.separate(self.input, self.output, "b.ckpt")
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(self.runtime._key, ("b.ckpt", "cpu"))


if __name__ == "__main__":
    unittest.main()
