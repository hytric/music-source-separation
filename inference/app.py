"""Portable Gradio demo: upload or select a local song, then separate it."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep direct script execution and `python -m inference.app` both supported.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.settings import AUDIO_EXTS, CACHE_DIR, DATA_DIR, env_bool, env_number

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(CACHE_DIR / "gradio"))

import logging
import shutil
import tempfile
import time
from io import BytesIO

import gradio as gr
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference.runtime import RUNTIME

MODELS = {
    "MDX-Net Inst HQ3 (경량·CPU 권장)": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "BS-RoFormer (고품질·CPU에서는 느림)": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}
MAX_MB = env_number("MSS_MAX_UPLOAD_MB", 100)
MAX_SECONDS = env_number("MSS_MAX_AUDIO_SECONDS", 600)
RETENTION_SECONDS = int(env_number("MSS_RETENTION_SECONDS", 3600))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Only this process's generated outputs are exposed, never pre-existing results/.
_WEB_TEMP = tempfile.TemporaryDirectory(prefix="web-", dir=CACHE_DIR)
OUTPUT_DIR = Path(_WEB_TEMP.name)


def list_audio() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return [
        str(p.relative_to(DATA_DIR))
        for p in sorted(DATA_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        and p.resolve().is_relative_to(DATA_DIR)
    ]


def _resolve(rel: str) -> str:
    path = (DATA_DIR / rel).resolve()
    if not path.is_relative_to(DATA_DIR) or not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
        raise ValueError("허용된 데이터셋 오디오를 선택하세요.")
    return str(path)


def validate_audio(path: str) -> None:
    audio = Path(path)
    if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTS:
        raise ValueError("지원하는 오디오 파일을 선택하세요.")
    if audio.stat().st_size > MAX_MB * 1024 ** 2:
        raise ValueError(f"파일 크기는 {MAX_MB:g}MB 이하여야 합니다.")
    try:
        duration = librosa.get_duration(path=path)
    except Exception as exc:
        raise ValueError("오디오를 읽을 수 없습니다. 파일 형식과 FFmpeg 설치를 확인하세요.") from exc
    if not np.isfinite(duration) or duration <= 0 or duration > MAX_SECONDS:
        raise ValueError(f"음원 길이는 0초 초과, {MAX_SECONDS:g}초 이하여야 합니다.")


def spectrogram_image(wav_path: str | None, title: str) -> Image.Image | None:
    if not wav_path:
        return None
    # Bound preview memory even for a long, high-sample-rate input.
    y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=60)
    spectrum = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    try:
        img = librosa.display.specshow(
            spectrum, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax,
        )
        ax.set_title(f"{title} (first 60s)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()
        with BytesIO() as buf:
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            with Image.open(buf) as image:
                return image.copy()
    finally:
        plt.close(fig)


def on_select(rel: str):
    if not rel:
        return None, None
    try:
        path = _resolve(rel)
        validate_audio(path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    # Stage only the selected audio, so external datasets need no broad file allowance.
    cleanup_outputs()
    preview_dir = Path(tempfile.mkdtemp(prefix="job-preview-", dir=OUTPUT_DIR))
    preview = preview_dir / Path(path).name
    try:
        shutil.copy2(path, preview)
        image = spectrogram_image(str(preview), "Mixture")
    except Exception:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise gr.Error("미리보기에 실패했습니다. 오디오와 디스크 공간을 확인하세요.") from None
    return str(preview), image


def on_upload(path: str | None):
    if not path:
        return None
    try:
        validate_audio(path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    return spectrogram_image(path, "Mixture")


def cleanup_outputs() -> None:
    cutoff = time.time() - RETENTION_SECONDS
    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder)


def separate(upload_path: str | None, rel: str | None, model_label: str):
    if not upload_path and not rel:
        return None, None, None, None, "음원을 업로드하거나 데이터셋에서 곡을 선택하세요."
    if model_label not in MODELS:
        raise gr.Error("지원하는 모델을 선택하세요.")
    try:
        path = upload_path or _resolve(rel)
        validate_audio(path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from None

    cleanup_outputs()
    job_dir = Path(tempfile.mkdtemp(prefix="job-", dir=OUTPUT_DIR))
    try:
        result = RUNTIME.separate(path, job_dir, MODELS[model_label], notify=gr.Warning)
        vocals = instrumental = None
        for filename in result.files:
            output = Path(filename)
            output = (output if output.is_absolute() else job_dir / output).resolve()
            if not output.is_relative_to(job_dir) or not output.is_file():
                raise RuntimeError("Unexpected model output path")
            low = output.name.lower()
            if "instrument" in low:
                instrumental = str(output)
            elif "vocal" in low:
                vocals = str(output)
        if not vocals or not instrumental:
            raise RuntimeError("Model did not return both stems")
        vocal_spec = spectrogram_image(vocals, "Vocals")
        inst_spec = spectrogram_image(instrumental, "Instrumental")
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        logging.getLogger(__name__).error("Separation failed (%s)", type(exc).__name__)
        raise gr.Error("분리에 실패했습니다. 모델 다운로드 연결, 메모리, 디스크 공간을 확인하세요.") from None
    # Retention starts at completion, which may be much later on CPU.
    os.utime(job_dir, None)
    status = (
        f"완료 · {model_label} · 실제 장치 {result.device} · 처리시간 {result.elapsed:.1f}s"
    )
    if result.notices:
        status += "\n" + "\n".join(result.notices)
    return vocals, vocal_spec, instrumental, inst_spec, status


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Music Source Separation (2-stem)", analytics_enabled=False,
        delete_cache=(RETENTION_SECONDS, RETENTION_SECONDS),
    ) as demo:
        gr.Markdown(
            "# 🎵 음악 소스 분리 (보컬 / 반주)\n"
            "음원을 업로드하거나 로컬 데이터셋에서 선택하세요. 업로드한 파일이 우선합니다.\n"
            f"최대 {MAX_MB:g}MB / {MAX_SECONDS:g}초 · 스펙트로그램은 앞 60초를 표시합니다. "
            "첫 실행에는 모델 다운로드가 필요하며 CPU에서는 시간이 걸릴 수 있습니다."
        )
        with gr.Row():
            dd = gr.Dropdown(choices=list_audio(), label="로컬 데이터셋 (선택 사항)", scale=4)
            refresh = gr.Button("새로고침", scale=1)
        with gr.Row():
            model = gr.Dropdown(choices=list(MODELS), value=next(iter(MODELS)), label="모델", scale=4)
            btn = gr.Button("분리 실행", variant="primary", scale=1)
        status = gr.Textbox(label="상태", interactive=False)
        with gr.Row():
            with gr.Column():
                orig = gr.Audio(label="원곡 업로드 / 재생", type="filepath", sources=["upload"])
                orig_spec = gr.Image(label="원곡 스펙트로그램")
            with gr.Column():
                out_voc = gr.Audio(label="보컬", interactive=False)
                voc_spec = gr.Image(label="보컬 스펙트로그램")
            with gr.Column():
                out_inst = gr.Audio(label="반주", interactive=False)
                inst_spec = gr.Image(label="반주 스펙트로그램")
        refresh.click(lambda: gr.update(choices=list_audio()), outputs=[dd], api_visibility="private")
        dd.change(on_select, inputs=[dd], outputs=[orig, orig_spec], concurrency_id="audio", concurrency_limit=1)
        orig.upload(on_upload, inputs=[orig], outputs=[orig_spec], concurrency_id="audio", concurrency_limit=1)
        btn.click(
            separate, inputs=[orig, dd, model],
            outputs=[out_voc, voc_spec, out_inst, inst_spec, status],
            concurrency_id="audio", concurrency_limit=1,
        )
    return demo.queue(max_size=8, default_concurrency_limit=1, api_open=False)


def launch_options() -> dict:
    auth = None
    if os.environ.get("MSS_AUTH"):
        username, sep, password = os.environ["MSS_AUTH"].partition(":")
        if not sep or not username.strip() or not password.strip():
            raise ValueError("MSS_AUTH는 사용자명:비밀번호 형식이어야 합니다.")
        auth = (username, password)
    port = int(os.environ.get("MSS_PORT", "7860"))
    if not 1 <= port <= 65535:
        raise ValueError("MSS_PORT는 1~65535 범위여야 합니다.")
    return dict(
        server_name=os.environ.get("MSS_HOST", "127.0.0.1"), server_port=port,
        share=env_bool("MSS_SHARE"), auth=auth, show_error=False,
        allowed_paths=[str(OUTPUT_DIR)], max_file_size=int(MAX_MB * 1024 ** 2),
    )


if __name__ == "__main__":
    build_ui().launch(**launch_options())
