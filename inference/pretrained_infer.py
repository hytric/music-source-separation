"""CLI inference with the same CUDA-to-CPU fallback used by the web demo."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from inference.runtime import RUNTIME, make_separator
from inference.settings import OUTPUT_DIR

DEFAULT_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"


def _audio_duration_sec(path: str | Path) -> float | None:
    import librosa

    try:
        return librosa.get_duration(path=str(path))
    except (OSError, RuntimeError, ValueError):
        return None


def separate(input_path: str | Path, output_dir: str | Path = OUTPUT_DIR,
             model_filename: str = DEFAULT_MODEL):
    """Return (output filenames, elapsed seconds including load/retry, RTF)."""
    result = RUNTIME.separate(input_path, output_dir, model_filename)
    duration = _audio_duration_sec(input_path)
    print(f"실제 실행 장치: {result.device}")
    return result.files, result.elapsed, result.elapsed / duration if duration else None


def list_models() -> None:
    models = make_separator("cpu", OUTPUT_DIR).list_supported_model_files()
    for architecture, entries in models.items():
        print(f"[{architecture}]")
        for name, metadata in entries.items():
            print(f"  {metadata['filename']}  — {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="보컬/반주 분리 (CPU 기본, CUDA 자동 폴백)")
    parser.add_argument("input", nargs="?", help="입력 오디오 경로 (wav/flac/mp3 등)")
    parser.add_argument("-o", "--output-dir", default=str(OUTPUT_DIR), help="출력 폴더")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="지원 모델 파일명")
    parser.add_argument("--list", action="store_true", help="지원 모델 목록 출력")
    args = parser.parse_args()
    try:
        if args.list:
            list_models()
            return
        if not args.input:
            parser.error("input 오디오 경로가 필요합니다 (또는 --list).")
        outputs, elapsed, rtf = separate(args.input, args.output_dir, args.model)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"분리 실패: {exc}\n")
    print(f"모델: {args.model}")
    print(f"전체 처리시간: {elapsed:.1f}s" + (f" (RTF={rtf:.2f})" if rtf else ""))
    for filename in outputs:
        output = Path(filename)
        print(output if output.is_absolute() else Path(args.output_dir) / output)


if __name__ == "__main__":
    main()
