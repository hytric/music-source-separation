"""파이프라인 스모크 테스트용 합성 믹스 생성 (외부 의존성 없음, stdlib만 사용).

보컬 느낌 멜로디(비브라토 사인) + 베이스 사인 + 드럼 느낌 노이즈 클릭을 섞어
data/raw/test_mix.wav (스테레오, 44.1kHz)로 저장한다.
실제 음악은 아니지만 추론 파이프라인이 end-to-end로 도는지 확인용.
"""
import math
import random
import struct
import wave
from pathlib import Path

SR = 44100
DUR = 12.0
OUT = Path(__file__).resolve().parent.parent / "data/raw/test_mix.wav"


def main() -> None:
    n = int(SR * DUR)
    random.seed(0)
    left = [0.0] * n
    right = [0.0] * n

    # 보컬 느낌: 비브라토가 있는 사인 멜로디 (음정 변화)
    notes = [440, 494, 523, 587, 523, 494, 440, 392]
    for i in range(n):
        t = i / SR
        note = notes[int(t) % len(notes)]
        vib = 1.0 + 0.01 * math.sin(2 * math.pi * 6 * t)  # 6Hz 비브라토
        voc = 0.30 * math.sin(2 * math.pi * note * vib * t)

        # 베이스: 저역 사인
        bass = 0.20 * math.sin(2 * math.pi * 80 * t)

        left[i] = voc + bass
        right[i] = voc + bass

    # 드럼 느낌: 0.5초마다 짧은 노이즈 버스트
    burst = int(0.05 * SR)
    for k in range(int(DUR / 0.5)):
        start = int(k * 0.5 * SR)
        for j in range(burst):
            if start + j < n:
                env = 1.0 - j / burst
                noise = 0.25 * env * (random.random() * 2 - 1)
                left[start + j] += noise
                right[start + j] += noise

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for i in range(n):
            l = max(-1.0, min(1.0, left[i]))
            r = max(-1.0, min(1.0, right[i]))
            frames += struct.pack("<hh", int(l * 32767), int(r * 32767))
        w.writeframes(bytes(frames))
    print(f"saved {OUT}  ({DUR:.0f}s, {SR}Hz, stereo)")


if __name__ == "__main__":
    main()
