"""Create an isolated environment, install dependencies and start the local UI."""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="음악 분리 웹 데모 설치 및 시작")
    parser.add_argument("--gpu", action="store_true", help="별도 CUDA 환경 설치 및 GPU 사용")
    parser.add_argument("--install-only", action="store_true", help="설치 검증 후 종료")
    parser.add_argument("--reinstall", action="store_true", help="의존성 설치 다시 실행")
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        parser.exit(1, "Python 3.11 또는 3.12 (64비트)를 설치하세요.\n")
    if platform.architecture()[0] != "64bit":
        parser.exit(1, "64비트 Python이 필요합니다.\n")
    if not shutil.which("ffmpeg"):
        parser.exit(1, "FFmpeg를 설치하고 PATH에 추가하세요. OS별 설치 방법은 README.md에 있습니다.\n")
    env_dir = ROOT / (".venv-gpu" if args.gpu else ".venv")
    executable = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.exists():
        print(f"가상환경 생성: {env_dir.name}", flush=True)
        venv.EnvBuilder(with_pip=True).create(env_dir)
    requirement = ROOT / ("requirements-gpu.txt" if args.gpu else "requirements.txt")
    digest = hashlib.sha256(
        requirement.read_bytes() + (ROOT / "requirements-base.txt").read_bytes()
        + str(sys.version_info[:2]).encode()
    ).hexdigest()
    marker = env_dir / ".mss-dependencies"
    installed = marker.read_text().strip() if marker.exists() else ""
    if args.reinstall or installed != digest:
        subprocess.run([str(executable), "-m", "pip", "install", "--upgrade", "pip"], check=True, cwd=ROOT)
        if not args.gpu and sys.platform != "darwin":
            subprocess.run([
                str(executable), "-m", "pip", "install", "torch==2.14.0+cpu", "torchvision==0.29.1+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ], check=True, cwd=ROOT)
        subprocess.run([str(executable), "-m", "pip", "install", "-r", str(requirement)], check=True, cwd=ROOT)
        subprocess.run([str(executable), "-m", "pip", "check"], check=True, cwd=ROOT)
        subprocess.run([str(executable), "-c", "import torch, torchvision, onnx2torch"], check=True, cwd=ROOT)
        marker.write_text(digest + "\n")
    if args.install_only:
        return
    environment = os.environ.copy()
    environment["MSS_USE_GPU"] = "1" if args.gpu else "0"
    subprocess.run([str(executable), "-m", "inference.app"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("설치 또는 실행에 실패했습니다. 위 오류와 README의 설치 조건을 확인하세요.", file=sys.stderr)
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        pass
