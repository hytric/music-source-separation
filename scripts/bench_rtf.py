"""Measure end-to-end RTF (including cold model load and CPU retry)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.pretrained_infer import _audio_duration_sec
from inference.runtime import RUNTIME
from inference.settings import OUTPUT_DIR

MODELS = {
    "MDX-Net-InstHQ3": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "BS-RoFormer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, default=ROOT / "data/raw/test_mix.wav")
    parser.add_argument("--warm", action="store_true", help="모델별 캐시된 두 번째 실행도 측정")
    args = parser.parse_args()
    duration = _audio_duration_sec(args.audio)
    if not duration or duration <= 0:
        parser.error("오디오를 읽을 수 없습니다. scripts/make_test_audio.py로 테스트 음원을 생성하세요.")
    print(f"{'model':18s} {'device':>6s} {'cold(s)':>9s} {'RTF':>6s}", flush=True)
    for name, model in MODELS.items():
        result = RUNTIME.separate(args.audio, OUTPUT_DIR / "benchmark", model)
        print(f"{name:18s} {result.device:>6s} {result.elapsed:9.1f} {result.elapsed / duration:6.2f}", flush=True)
        if args.warm:
            result = RUNTIME.separate(args.audio, OUTPUT_DIR / "benchmark", model)
            print(f"{'  warm':18s} {result.device:>6s} {result.elapsed:9.1f} {result.elapsed / duration:6.2f}", flush=True)


if __name__ == "__main__":
    main()
