"""Extract/validate official train sources without touching the existing test split."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import unicodedata
import zipfile
import zlib

import numpy as np
import soundfile as sf

from .common import SOURCES, atomic_json, digest, emit, exclusive, file_entry, read_json, sha256, verify_entry


def song_split(names, counts, salt):
    names = [unicodedata.normalize("NFC", n) for n in names]
    if len(names) != sum(counts.values()) or len(set(names)) != len(names):
        raise ValueError("unexpected song count or normalization collision")
    ordered = sorted(names, key=lambda n: (hashlib.sha256(salt.encode() + b"\0" + n.encode()).hexdigest(), n))
    result, offset = {}, 0
    for split in ("train", "validation", "confirmation"):
        for n in ordered[offset:offset + counts[split]]:
            result[n] = split
        offset += counts[split]
    return result


def train_members(archive):
    result = {}
    for info in archive.infolist():
        p = PurePosixPath(info.filename)
        if not p.parts or p.parts[0] != "train" or info.is_dir():
            continue
        if len(p.parts) != 3 or ".." in p.parts or "\\" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"unsafe train member: {info.filename}")
        if p.suffix != ".wav":
            continue
        if p.name not in [s + ".wav" for s in SOURCES] + ["mixture.wav"]:
            raise ValueError(f"unexpected train audio: {info.filename}")
        key = (p.parts[1], p.name)
        if key in result:
            raise ValueError(f"duplicate ZIP member: {key}")
        result[key] = info
    return result


def _existing_crc(path, size, crc):
    if not path.is_file() or path.stat().st_size != size:
        return False
    actual = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            actual = zlib.crc32(block, actual)
    return actual == crc


def _decode_check(path):
    count = 0
    with sf.SoundFile(path) as f:
        if f.samplerate != 44100 or f.channels != 2 or f.frames <= 0:
            raise ValueError(f"invalid full-length HQ source: {path}")
        info = {"sample_rate": f.samplerate, "channels": f.channels, "frames": f.frames, "subtype": f.subtype}
        for block in f.blocks(blocksize=262144, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError(f"nonfinite decoded source: {path}")
            count += len(block)
    if count != info["frames"]:
        raise ValueError(f"truncated decoded source: {path}")
    return info


def prepare_data(root, archive_path, config, baseline_runtime):
    root = Path(root)
    with exclusive(root, "prepare-data"):
        settings = {"data": config["data"], "audio": config["audio"], "source_order": list(SOURCES)}
        archive_path = Path(archive_path).resolve()
        archive_sha = sha256(archive_path)
        path = root / "manifests/dataset.json"
        if path.exists():
            manifest = read_json(path)
            if manifest["settings_sha256"] != digest(settings) or manifest["archive_sha256"] != archive_sha:
                raise ValueError("dataset producer inputs changed; use a new runtime")
            for row in manifest["tracks"]:
                for entry in row["files"].values():
                    verify_entry(root, entry)
            emit("dataset_reused", tracks=len(manifest["tracks"]))
            return manifest
        with zipfile.ZipFile(archive_path) as z:
            members = train_members(z)
            names = sorted({track for track, _ in members})
            if len(names) != 100 or len(members) != 500:
                raise ValueError("expected 100 train songs and 500 WAVs")
            old_test = read_json(Path(baseline_runtime) / "manifests/dataset.json")
            if set(names) & {r["name"] for r in old_test["tracks"]}:
                raise ValueError("official train/test name overlap")
            split = song_split(names, config["data"]["split_counts"], config["data"]["split_salt"])
            need = sum(i.file_size for i in members.values()) + int(config["data"]["minimum_free_gb"] * 1e9)
            if shutil.disk_usage(root).free < need:
                raise OSError(f"need {need / 1e9:.1f} GB free before extraction and caches")
            tracks = []
            for index, name in enumerate(names):
                files = {}
                for stem in (*SOURCES, "mixture"):
                    info = members[(name, stem + ".wav")]
                    target = root / "data/train" / name / (stem + ".wav")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not _existing_crc(target, info.file_size, info.CRC):
                        tmp = target.with_suffix(".wav.part")
                        with z.open(info) as src, tmp.open("wb") as dst:
                            shutil.copyfileobj(src, dst, 4 << 20)
                        if tmp.stat().st_size != info.file_size:
                            raise ValueError(f"extraction size mismatch: {name}/{stem}")
                        tmp.replace(target)
                    files[stem] = {**file_entry(target, root), **_decode_check(target), "zip_crc32": info.CRC}
                if len({files[s]["frames"] for s in SOURCES}) != 1:
                    raise ValueError(f"stem lengths differ: {name}")
                frames = files[SOURCES[0]]["frames"]
                retained = min(frames, 30 * 44100)
                left = (30 * 44100 - retained) // 2
                tracks.append({"name": name, "split": split[name], "files": files,
                               "crop_start_frame": max(0, (frames - 30 * 44100) // 2), "crop_frames": retained,
                               "pad_left_frames": left, "pad_right_frames": 30 * 44100 - retained - left,
                               "padding_policy": "symmetric_zero_pad_short_songs_without_repetition"})
                if (index + 1) % 5 == 0:
                    emit("train_sources_validated", done=index + 1, total=100)
        manifest = {"schema_version": 1, "dataset": "MUSDB18-HQ official train", "archive_sha256": archive_sha,
                    "settings_sha256": digest(settings), "settings": settings, "tracks": tracks,
                    "crc_decode_and_hash_checked": True, "baseline_pretraining_overlap": "unconfirmed",
                    "existing_test_usage": "posthoc-diagnostic-only"}
        atomic_json(path, manifest)
        emit("dataset_complete", tracks=100, splits=config["data"]["split_counts"])
        return manifest
