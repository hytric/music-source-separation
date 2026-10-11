"""MUSDB 샘플(실제 음악 + 정답 stem) 다운로드 → 데모/평가용 wav로 export.

- 믹스처  → data/raw/musdb_sample/*.wav        (Gradio 데모에서 선택할 데이터셋)
- 정답 stem(vocals) → data/processed/musdb_sample_stems/*__vocals.wav  (SDR 평가용)

MUSDB18-HQ 전체는 W2에서 별도 확보. 이건 즉시 들어보기 위한 무료 샘플.
"""
from pathlib import Path

import musdb
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/musdb_sample_stems"            # musdb 원본(.stem.mp4)
OUT = ROOT / "data/raw/musdb_sample"                  # 믹스처 wav (데모용)
STEMS = ROOT / "data/processed/musdb_sample_stems"    # 정답 stem wav (평가용)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STEMS.mkdir(parents=True, exist_ok=True)

    mus = musdb.DB(root=str(RAW), download=True)
    print(f"tracks: {len(mus)}")
    for track in mus:
        name = track.name.replace("/", "_")
        sf.write(str(OUT / f"{name}.wav"), track.audio, track.rate)
        sf.write(
            str(STEMS / f"{name}__vocals.wav"),
            track.targets["vocals"].audio,
            track.rate,
        )
        dur = track.audio.shape[0] / track.rate
        print(f"exported {name}  ({dur:.1f}s)")

    print("\nDATASET(mix):", OUT.resolve())
    print("STEMS(vocals):", STEMS.resolve())


if __name__ == "__main__":
    main()
