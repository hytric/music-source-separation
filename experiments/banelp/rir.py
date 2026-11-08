"""RIR acquisition: OpenAIR first, public full-band substitute next, seeded synthetic last.

The source list is a host-run input (`rir_sources.example.json` documents its schema). URLs,
checksums and licenses must be verified by the host-run before they are filled in; an entry
with no URL fails with a recorded reason instead of guessing an endpoint.
"""
from __future__ import annotations
import hashlib, json, os, shutil, stat, tarfile, tempfile, urllib.parse, urllib.request, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import soundfile as sf
from .contracts import SAMPLE_RATE, SOURCES, sha256_file, validate_runtime_path
from .data import atomic_json, prepare_rir

TARGET_RT60 = {"R1": 0.26, "R2": 0.89, "R3": 1.28, "R4": 1.96}
POSITIONS = ("P1", "P2", "P3", "P4")
# Reimplementation choice (the original mapping is unknown): stem i uses position i of the room.
STEM_POSITION = dict(zip(SOURCES, POSITIONS))
RESAMPLE_METHOD = "scipy.signal.resample_poly(up/gcd, down/gcd, axis=0), default Kaiser(5.0) FIR"
MAX_MEMBER_BYTES = 512 << 20

def download_resumable(url: str, dest: str | Path, *, sha256: str | None = None, expected_bytes: int | None = None,
                       min_free_bytes: int = 1 << 30, timeout: float = 60.0, chunk: int = 1 << 20) -> dict:
    """Resume into `<dest>.part` with HTTP Range, check disk space and checksum, then rename."""
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"https", "http", "file"}: raise ValueError(f"unsupported URL scheme: {scheme}")
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True); part = dest.with_name(dest.name + ".part")
    if dest.is_file():
        digest = sha256_file(dest)
        if sha256 is None or digest == sha256:
            return {"url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": digest, "provider_sha256": sha256,
                    "checksum": "provider-verified" if sha256 else "computed-only", "resumed_from": None, "reused": True}
        dest.unlink()
    offset = part.stat().st_size if part.is_file() else 0
    need = (expected_bytes - offset if expected_bytes else 0) + min_free_bytes
    if shutil.disk_usage(dest.parent).free < need: raise OSError(f"insufficient disk space for {url}: need {need} bytes")
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset and scheme != "file" else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", None) or 200
        if offset and status != 206: offset = 0  # server ignored Range; restart cleanly
        length = response.headers.get("Content-Length")
        if length is not None and shutil.disk_usage(dest.parent).free < int(length) + min_free_bytes:
            raise OSError(f"insufficient disk space for {url}: {length} bytes remaining")
        with part.open("ab" if offset else "wb") as stream:
            while block := response.read(chunk): stream.write(block)
    digest = sha256_file(part)
    if sha256 is not None and digest != sha256:
        part.unlink(); raise ValueError(f"checksum mismatch for {url}: {digest} != {sha256}")
    if expected_bytes is not None and part.stat().st_size != expected_bytes:
        part.unlink(); raise ValueError(f"size mismatch for {url}")
    os.replace(part, dest)
    return {"url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": digest, "provider_sha256": sha256,
            "checksum": "provider-verified" if sha256 else "computed-only", "resumed_from": offset or None, "reused": False}

def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or not member.parts or ".." in member.parts or ":" in member.parts[0]:
        raise ValueError(f"unsafe archive member path: {name}")
    return member

def safe_extract(archive: str | Path, members: list[str], dest: str | Path, *, max_bytes: int = MAX_MEMBER_BYTES) -> dict[str, Path]:
    """Extract only the named regular-file members; reject traversal, links and oversized entries."""
    archive, dest = Path(archive), Path(dest); dest.mkdir(parents=True, exist_ok=True); out = {}
    def target(name):
        path = (dest / _safe_member(name)).resolve()
        if dest.resolve() not in path.parents: raise ValueError(f"archive member escapes destination: {name}")
        path.parent.mkdir(parents=True, exist_ok=True); return path
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in members:
                info = zf.getinfo(name)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16): raise ValueError(f"not a regular file: {name}")
                if info.file_size > max_bytes: raise ValueError(f"archive member too large: {name}")
                path = target(name)
                with zf.open(info) as src, path.open("wb") as dst: shutil.copyfileobj(src, dst)
                out[name] = path
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for name in members:
                info = tf.getmember(name)
                if not info.isfile(): raise ValueError(f"not a regular file: {name}")
                if info.size > max_bytes: raise ValueError(f"archive member too large: {name}")
                path = target(name); src = tf.extractfile(info)
                with src, path.open("wb") as dst: shutil.copyfileobj(src, dst)
                out[name] = path
    else: raise ValueError(f"unsupported archive format: {archive}")
    return out

def synthetic_rir(room: str, position: str, *, salt: str = "banelp-v1", sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Deterministic exponentially decaying noise RIR; full band only because it is generated at 44.1 kHz."""
    target = TARGET_RT60[room]
    seed = int.from_bytes(hashlib.sha256(f"{salt}\0{room}\0{position}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed); index = POSITIONS.index(position)
    length = int(round((1.5 * target + 0.05) * sample_rate))
    t = np.arange(length) / sample_rate
    onset = int(round((0.002 + 0.001 * index) * sample_rate))  # first diffuse arrival after the direct path
    diffuse = rng.standard_normal(length) * np.exp(-3 * np.log(10) * t / target)  # 60 dB energy decay at RT60
    diffuse[:onset] = 0.0
    rir = np.clip(0.3 * diffuse, -0.9, 0.9); rir[0] = 1.0  # direct path is the unique peak at index 0, so onset offset 0 is the identity
    return rir, {"generator": "seeded-exponential-noise", "seed": seed, "salt": salt, "target_rt60_seconds": target,
                 "length_samples": length, "diffuse_onset_samples": onset, "diffuse_gain": 0.3}

def _write_rir(runtime: Path, room: str, position: str, rir: np.ndarray) -> tuple[str, str]:
    path = validate_runtime_path(runtime / "rir" / room / f"{position}.npy", runtime); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{position}.", dir=path.parent); os.close(fd)
    try:
        with open(tmp, "wb") as stream: np.save(stream, np.asarray(rir, dtype=np.float64), allow_pickle=False)
        os.replace(tmp, path)
    finally: Path(tmp).unlink(missing_ok=True)
    return str(path.relative_to(runtime)), sha256_file(path)

def _measured(info: dict) -> dict:
    t30 = info.pop("rt30_estimate_seconds")
    return {**info, "measured_t30_seconds": t30, "t30_failure": None if t30 is not None else "decay did not reach -35 dB"}

def _from_candidate(runtime: Path, room: str, index: int, entry: dict) -> dict:
    if entry.get("kind") not in {"openair", "substitute"}: raise ValueError("candidate kind must be openair or substitute")
    if not entry.get("url"): raise ValueError("no verified URL recorded for this candidate")
    if not entry.get("license"): raise ValueError("no license recorded for this candidate")
    work = runtime / "rir/downloads" / f"{room}-{index}"
    name = Path(urllib.parse.urlparse(entry["url"]).path).name or "download"
    fetched = download_resumable(entry["url"], work / name, sha256=entry.get("sha256"), expected_bytes=entry.get("bytes"))
    positions = entry.get("positions") or {}
    if set(positions) != set(POSITIONS): raise ValueError(f"candidate must name exactly positions {POSITIONS}")
    if entry.get("archive"): files = safe_extract(work / name, [p["member"] for p in positions.values()], work / "extracted")
    else:
        if len({p.get("member") for p in positions.values()}) != 1: raise ValueError("non-archive candidate must be one file")
        files = {positions["P1"].get("member"): work / name}
    records = {}
    for position, spec in positions.items():
        source = files[spec["member"]]
        audio, native = sf.read(source, always_2d=True, dtype="float64")
        if native < SAMPLE_RATE: raise ValueError(f"{source.name}: native {native} Hz is not full band; upsampling is not a substitute")
        rir, info = prepare_rir(audio, native, channel=spec.get("channel"))
        if audio.shape[1] > 1 and not spec.get("channel_note"): raise ValueError("multichannel RIR needs a documented channel_note (e.g. B-format W)")
        path, digest = _write_rir(runtime, room, position, rir)
        records[position] = {"path": path, "sha256": digest, "source_member": spec["member"], "source_sha256": sha256_file(source),
                             "channel_note": spec.get("channel_note"), "resample_method": RESAMPLE_METHOD if info["resampled"] else "none",
                             "onset_offset": 0, **_measured(info)}
    return {"kind": entry["kind"], "space": entry.get("space"), "url": entry["url"], "license": entry["license"],
            "download": fetched, "full_band": True, "positions": records}

def _synthetic_room(runtime: Path, room: str) -> dict:
    records = {}
    for position in POSITIONS:
        rir, gen = synthetic_rir(room, position)
        rir, info = prepare_rir(rir, SAMPLE_RATE)
        path, digest = _write_rir(runtime, room, position, rir)
        records[position] = {"path": path, "sha256": digest, "generator": gen, "resample_method": "none", "onset_offset": 0, **_measured(info)}
    return {"kind": "synthetic", "space": None, "url": None, "license": "generated", "full_band": True,
            "note": "synthetic substitute; not an OpenAIR reproduction; target and measured RT60 are kept separate", "positions": records}

def acquire_rirs(runtime: str | Path, *, sources: dict | None, synthetic_fallback: bool, manifest_path: str | Path | None = None) -> dict:
    """Try each room's candidates in order; a room with no usable candidate uses synthetic only if allowed."""
    runtime = Path(runtime).resolve(); rooms_spec = {r["room"]: r for r in (sources or {}).get("rooms", [])}
    if set(rooms_spec) - set(TARGET_RT60): raise ValueError("source spec names unknown rooms")
    rooms = {}
    for room, target in TARGET_RT60.items():
        spec = rooms_spec.get(room, {"candidates": []})
        if spec.get("target_rt60_seconds", target) != target: raise ValueError(f"{room}: target RT60 must be {target}")
        attempts, chosen = [], None
        for index, entry in enumerate(spec.get("candidates", [])):
            try: chosen = _from_candidate(runtime, room, index, entry); break
            except Exception as exc: attempts.append({"index": index, "kind": entry.get("kind"), "space": entry.get("space"),
                                                      "url": entry.get("url"), "failure": f"{type(exc).__name__}: {exc}"})
        if chosen is None:
            if not synthetic_fallback: raise RuntimeError(f"{room}: no RIR candidate succeeded and synthetic fallback is not enabled: {attempts}")
            chosen = _synthetic_room(runtime, room)
        rooms[room] = {"target_rt60_seconds": target, "attempts": attempts, **chosen}
    manifest = {"schema_version": 1, "sample_rate": SAMPLE_RATE, "onset_offset": 0, "positions": list(POSITIONS),
                "stem_position": STEM_POSITION, "rooms": rooms}
    atomic_json(manifest_path or runtime / "manifests/rirs.json", manifest)
    return manifest

def load_rirs(runtime: str | Path, manifest: dict) -> dict[str, dict[str, np.ndarray]]:
    """Load every RIR named by the manifest after checking its content hash."""
    runtime = Path(runtime).resolve(); out = {}
    if set(manifest.get("rooms", {})) != set(TARGET_RT60): raise ValueError("RIR manifest must name R1..R4")
    for room, record in manifest["rooms"].items():
        out[room] = {}
        for position in POSITIONS:
            entry = record["positions"][position]; path = validate_runtime_path(runtime / entry["path"], runtime)
            if sha256_file(path) != entry["sha256"]: raise ValueError(f"RIR content changed: {path}")
            out[room][position] = np.load(path, allow_pickle=False)
    return out

def load_sources(path: str | Path | None) -> dict | None:
    return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
