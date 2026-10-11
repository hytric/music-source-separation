"""Build a source-only ZIP from an explicit manifest, never the entire folder."""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT_FILES = [
    "README.md", ".gitignore", ".gitattributes", ".dockerignore", ".env.example", "Dockerfile",
    "compose.yaml", "start.py", "requirements.txt", "requirements-base.txt",
    "requirements-gpu.txt", "requirements-data.txt",
]
# Explicit files prevent a newly added private script/document entering a release.
SOURCE_FILES = [
    "inference/__init__.py", "inference/settings.py", "inference/runtime.py",
    "inference/app.py", "inference/pretrained_infer.py", "scripts/package_release.py",
    "scripts/make_test_audio.py", "scripts/prepare_musdb_sample.py", "scripts/bench_rtf.py",
    "tests/test_runtime.py", "tests/test_app.py", "tests/test_release.py",
    "docs/deployment.md", "docs/git-deployment.md",
]
SECRET_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})"
    r"|\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}"
    r"|(?im:^[ \t]*MSS_AUTH[ \t]*=[ \t]*[^\s#]+)"
)
PERSONAL_PATH = re.compile(r"/(?:home|Users)/[\w.-]+/|[A-Za-z]:\\Users\\[^\\]+\\")


def release_files() -> list[Path]:
    files = []
    for relative in ROOT_FILES + SOURCE_FILES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or unsafe release file: {relative}")
        content = path.read_text(encoding="utf-8")
        if SECRET_PATTERN.search(content) or PERSONAL_PATH.search(content):
            raise ValueError(f"Potential credential or personal path in: {relative}")
        files.append(path)
    return files


def build_release(output: Path) -> list[str]:
    files = release_files()
    output = output.resolve()
    if output in files:
        raise ValueError("Release output must not overwrite a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="검사한 소스만 배포용 ZIP으로 생성")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "dist/music-source-separation.zip")
    args = parser.parse_args()
    names = build_release(args.output)
    print(f"Created {args.output.name}: {len(names)} files (no data, results, credentials or model weights)")


if __name__ == "__main__":
    main()
