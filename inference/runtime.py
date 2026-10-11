"""Shared, serialized inference with CUDA preflight and CPU retry."""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .settings import MODEL_DIR, env_bool, env_number

Notify = Callable[[str], None]
LOGGER = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    files: list[str]
    elapsed: float
    device: str
    notices: list[str]


def select_device(model_filename: str) -> tuple[str, str | None]:
    if not env_bool("MSS_USE_GPU"):
        return "cpu", None

    import torch

    try:
        if not torch.cuda.is_available():
            return "cpu", "사용 가능한 CUDA GPU가 없습니다. CPU로 실행합니다."
        if model_filename.lower().endswith(".onnx"):
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return "cpu", "ONNX Runtime의 CUDA 지원이 없습니다. CPU로 실행합니다."
        # A conservative preflight, not an exact prediction of peak VRAM.
        required_gb = env_number("MSS_MIN_VRAM_GB", 2 if model_filename.endswith(".onnx") else 6)
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
        if free_gb < required_gb:
            return "cpu", (
                f"GPU 여유 VRAM {free_gb:.1f}GB가 기준 {required_gb:g}GB보다 적습니다. "
                "CPU로 실행합니다."
            )
    except (RuntimeError, OSError):
        return "cpu", "CUDA 장치 초기화에 실패했습니다. CPU로 실행합니다."
    return "cuda", None


def is_gpu_failure(exc: Exception) -> bool:
    """Only GPU memory/backend errors qualify; invalid inputs must propagate."""
    message = str(exc).lower()
    oom = any(term in message for term in (
        "out of memory", "cudaerrormemoryallocation", "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed", "failed to allocate memory", "bfcarena::allocaterawinternal",
    ))
    backend = any(term in message for term in (
        "cuda driver", "cuda initialization", "cuda error", "cuda failure",
        "cudaexecutionprovider", "libcudnn", "libcublas", "cudnn_status_not_initialized",
    ))
    if oom or backend:
        return True
    return False


def check_cuda_provider(separator, model: str) -> None:
    if not model.lower().endswith(".onnx"):
        return
    instance = separator.model_instance
    if getattr(instance, "uses_pytorch_inference", True):
        return
    # In the pinned MDX implementation, model_run closes over its ORT session.
    for cell in instance.model_run.__closure__ or ():
        session = cell.cell_contents
        if hasattr(session, "get_providers"):
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("CUDAExecutionProvider could not activate")
            return
    raise RuntimeError("CUDAExecutionProvider could not be verified")


def make_separator(device: str, output_dir: Path):
    import torch
    from audio_separator.separator import Separator

    class DeviceSeparator(Separator):
        # audio-separator 0.47.0 has no public force-CPU constructor option.
        # Override its device hook before model loading, including on Apple Silicon.
        def setup_torch_device(self, system_info):
            self.torch_device_cpu = torch.device("cpu")
            self.torch_device_mps = None
            self.torch_device = torch.device(device)
            self.onnx_execution_provider = [
                "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            ]

    separator = DeviceSeparator(
        output_dir=str(output_dir), model_file_dir=str(MODEL_DIR),
        log_level=logging.ERROR, use_soundfile=True,
    )
    return separator


class SeparationRuntime:
    """Keep at most one model resident and retry once after a GPU failure."""

    def __init__(self):
        self._lock = threading.RLock()
        self._separator = None
        self._key = None
        self._cpu_models: set[str] = set()

    def _release(self):
        had_cuda = self._key is not None and self._key[1] == "cuda"
        self._separator = None
        self._key = None
        gc.collect()
        if had_cuda:
            import torch

            try:
                torch.cuda.empty_cache()
            except (RuntimeError, OSError):
                pass

    def _run(self, input_path: Path, output_dir: Path, model: str, device: str) -> list[str]:
        key = (model, device)
        if self._key != key:
            self._release()
            self._key = key
            self._separator = make_separator(device, output_dir)
            self._separator.load_model(model_filename=model)
            if device == "cuda":
                check_cuda_provider(self._separator, model)
        sep = self._separator
        sep.output_dir = str(output_dir)
        sep.model_instance.output_dir = str(output_dir)
        return sep.separate(str(input_path))

    def separate(self, input_path: str | Path, output_dir: str | Path,
                 model: str, notify: Notify | None = None) -> SeparationResult:
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError("입력 오디오 파일을 찾을 수 없습니다.")
        if Path(model).name != model or "/" in model or "\\" in model:
            raise ValueError("모델은 경로가 아닌 지원 모델 파일명으로 지정하세요.")
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        notices = []

        def report(message: str):
            notices.append(message)
            LOGGER.warning(message)
            if notify:
                notify(message)

        with self._lock:
            # Release a different cached model before checking free VRAM.
            if self._key and self._key[0] != model:
                self._release()
            if model in self._cpu_models and env_bool("MSS_USE_GPU"):
                device, notice = "cpu", "이 모델은 앞선 GPU 오류로 CPU를 사용합니다."
            elif self._key == (model, "cuda") and env_bool("MSS_USE_GPU"):
                device, notice = "cuda", None
            else:
                device, notice = select_device(model)
            if notice:
                report(notice)
            t0 = time.perf_counter()
            retry_cpu = False
            try:
                files = self._run(input_path, output_dir, model, device)
            except Exception as exc:
                if device != "cuda" or not is_gpu_failure(exc):
                    self._release()
                    raise
                # Release traceback frames holding GPU tensors before collecting.
                exc.__traceback__ = None
                retry_cpu = True
            if retry_cpu:
                self._release()
                self._cpu_models.add(model)
                report("GPU 메모리 부족 또는 CUDA 실행 오류가 발생했습니다. CPU로 다시 실행합니다.")
                device = "cpu"
                try:
                    files = self._run(input_path, output_dir, model, device)
                except Exception:
                    self._release()
                    raise
            return SeparationResult(files, time.perf_counter() - t0, device.upper(), notices)


RUNTIME = SeparationRuntime()
