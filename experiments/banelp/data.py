"""Manifest validation and deterministic wet-target dataset construction."""
from __future__ import annotations
import hashlib, json, math, os, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from .contracts import SOURCES, SAMPLE_RATE, SAMPLES, deterministic_split, sha256_file, validate_runtime_path

def crop_center(audio: np.ndarray, length: int=SAMPLES) -> tuple[np.ndarray,int]:
    if audio.ndim != 2: raise ValueError("audio must be (samples,channels)")
    if len(audio)<length: raise ValueError(f"track too short: {len(audio)} < {length}")
    start=(len(audio)-length)//2
    return audio[start:start+length],start

def _rms(audio): return float(np.sqrt(np.mean(np.asarray(audio,dtype=np.float64)**2)))

def apply_rir(dry: np.ndarray, rir: np.ndarray, *, onset_offset: int=0, rir_channel: int|None=None) -> tuple[np.ndarray,dict]:
    """Same mono RIR on both channels, truncate to N and restore each stem RMS."""
    if dry.ndim != 2 or dry.shape[1] != 2: raise ValueError("dry stem must be stereo")
    selected_channel=None
    if rir.ndim == 2:
        if rir.shape[1] == 1: rir=rir[:,0]
        elif rir_channel is not None and 0 <= rir_channel < rir.shape[1]:
            selected_channel=rir_channel; rir=rir[:,rir_channel]
        else: raise ValueError("multichannel RIR requires an explicitly documented extraction channel")
    if rir.ndim != 1 or not np.isfinite(rir).all(): raise ValueError("invalid RIR")
    if onset_offset < 0 or onset_offset >= len(rir): raise ValueError("invalid onset offset")
    rir=rir[onset_offset:]
    dry_rms=_rms(dry)
    if dry_rms<=1e-12: return np.zeros_like(dry,dtype=np.float32),{"dry_rms":dry_rms,"wet_rms":0.0,"gain":0.0,"tail_energy":0.0,"dry_length":len(dry),"full_length":len(dry)+len(rir)-1}
    full=np.stack([fftconvolve(dry[:,ch],rir,mode="full") for ch in range(2)])
    cropped=full[:,:len(dry)].T
    wet_rms=_rms(cropped)
    if wet_rms<=1e-12: raise ValueError("non-silent dry stem produced near-zero wet RMS")
    gain=dry_rms/wet_rms; out=(cropped*gain).astype(np.float32)
    tail=float(np.sum(full[:,len(dry):]**2))
    return out,{"dry_rms":dry_rms,"wet_rms_before_gain":wet_rms,"gain":gain,"wet_rms_after_gain":_rms(out),"tail_energy":tail,"dry_length":len(dry),"full_length":full.shape[1],"onset_offset":onset_offset,"rir_channel":selected_channel}

def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int=SAMPLE_RATE) -> np.ndarray:
    if src_rate==dst_rate:return np.asarray(audio,dtype=np.float32)
    common=math.gcd(src_rate,dst_rate)
    return resample_poly(audio,dst_rate//common,src_rate//common,axis=0).astype(np.float32)

def make_sample(dry_stems: dict[str,np.ndarray], rir_by_source: dict[str,np.ndarray], *, start_samples: dict[str,int]|None=None) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    if set(dry_stems)!=set(SOURCES) or set(rir_by_source)!=set(SOURCES): raise ValueError("exactly four named stems and RIRs required")
    wet=[]; rows=[]
    for name in SOURCES:
        stem=np.asarray(dry_stems[name])
        if start_samples is None:
            segment,start=crop_center(stem,SAMPLES)
        else:
            start=start_samples[name]; segment=stem[start:start+SAMPLES]
        if len(segment)!=SAMPLES: raise ValueError(f"short stem {name}")
        result,info=apply_rir(segment,rir_by_source[name]); wet.append(result); rows.append({"source":name,"start_sample":start,**info})
    target=np.stack(wet).astype(np.float32)
    mixture=target.astype(np.float64).sum(axis=0).astype(np.float32)
    return mixture,target,rows

def build_split_manifest(names: list[str], salt="banelp-v1"):
    split=deterministic_split(names,salt)
    return {"schema_version":1,"dataset":"MUSDB18-HQ test","salt":salt,"tracks":[{"name":n,"split":split[n],"crop":"center-30s"} for n in sorted(split)]}

def atomic_json(path: str|Path, payload: dict):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)

def inspect_audio(path: str|Path) -> dict:
    info=sf.info(path)
    return {"path":str(path),"sha256":sha256_file(path),"bytes":Path(path).stat().st_size,"frames":info.frames,"samplerate":info.samplerate,"channels":info.channels}

def validate_acquired_dataset(track_root: str|Path, runtime_dir: str|Path, manifest_path: str|Path) -> dict:
    """Validate an already acquired HQ tree; acquisition credentials are host-run owned."""
    root=validate_runtime_path(track_root,runtime_dir)
    tracks=[]
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        files={name:directory/f"{name}.wav" for name in SOURCES}
        if any(not p.is_file() for p in files.values()): continue
        records={name:inspect_audio(path) for name,path in files.items()}
        if len({(v["frames"],v["samplerate"],v["channels"]) for v in records.values()})!=1: raise ValueError(f"unaligned stems: {directory.name}")
        if next(iter(records.values()))["channels"]!=2: raise ValueError(f"non-stereo stems: {directory.name}")
        if next(iter(records.values()))["samplerate"]!=SAMPLE_RATE: raise ValueError(f"unexpected source sample rate: {directory.name}")
        tracks.append({"name":directory.name,"sources":records})
    if len(tracks)!=50: raise ValueError(f"expected 50 complete HQ tracks, found {len(tracks)}")
    manifest=build_split_manifest([t["name"] for t in tracks])
    by_name={__import__("unicodedata").normalize("NFC",t["name"]):t for t in tracks}
    if len(by_name)!=len(tracks): raise ValueError("track names collide after NFC normalization")
    for entry in manifest["tracks"]: entry["sources"]=by_name[entry["name"]]["sources"]
    manifest["source_root"]="runtime-relative"
    atomic_json(manifest_path,manifest)
    return manifest

def prepare_rir(audio: np.ndarray, native_rate: int, *, channel: int|None=None) -> tuple[np.ndarray,dict]:
    """Convert one documented mono/W-channel RIR to 44.1 kHz and report T30."""
    raw=np.asarray(audio)
    if raw.ndim==2:
        if raw.shape[1]==1: extracted=raw[:,0]; selected=0
        elif channel is not None and 0<=channel<raw.shape[1]: extracted=raw[:,channel]; selected=channel
        else: raise ValueError("multichannel RIR requires documented channel selection")
    elif raw.ndim==1: extracted=raw; selected=None
    else: raise ValueError("RIR must be mono or channels-last")
    if native_rate<=0 or not np.isfinite(extracted).all(): raise ValueError("invalid RIR sample rate or values")
    converted=resample_audio(extracted,native_rate,SAMPLE_RATE)
    energy=converted.astype(np.float64)**2
    decay=np.cumsum(energy[::-1])[::-1]
    if not decay.size or decay[0]<=0: t30=None
    else:
        edc=10*np.log10(np.maximum(decay/decay[0],1e-30)); idx5=np.flatnonzero(edc<=-5); idx35=np.flatnonzero(edc<=-35)
        t30=float(2*(idx35[0]-idx5[0])/SAMPLE_RATE) if idx5.size and idx35.size and idx35[0]>idx5[0] else None
    return converted,{"native_sample_rate":native_rate,"sample_rate":SAMPLE_RATE,"channel_count":1 if raw.ndim==1 else raw.shape[1],"selected_channel":selected,"resampled":native_rate!=SAMPLE_RATE,"rt30_estimate_seconds":t30}

def save_wet_sample(runtime_dir: str|Path, *, split: str, track: str, room: str,
                   dry_stems: dict[str,np.ndarray], rir_by_source: dict[str,np.ndarray],
                   source_records: dict[str,dict], rir_records: dict[str,dict], start_samples: dict[str,int]|None=None) -> dict:
    """Atomically persist one wet-target sample and a compact content-bound record."""
    if split not in {"dev","evaluation"}: raise ValueError("split must be dev or evaluation")
    if room not in {"R1","R2","R3","R4"}: raise ValueError("room must be R1..R4")
    root=validate_runtime_path(Path(runtime_dir)/"cache/wet"/split/track/room,runtime_dir)
    mixture,targets,rows=make_sample(dry_stems,rir_by_source,start_samples=start_samples)
    if not np.isfinite(mixture).all() or not np.isfinite(targets).all(): raise ValueError("wet sample contains nonfinite samples")
    if not np.allclose(targets.astype(np.float64).sum(axis=0),mixture.astype(np.float64),rtol=0,atol=1e-6): raise ValueError("wet target sum does not equal mixture")
    root.mkdir(parents=True,exist_ok=True)
    for name,array in (("mixture.npy",mixture),("targets.npy",targets)):
        fd,tmp=tempfile.mkstemp(prefix=f".{name}.",dir=root); os.close(fd)
        try:
            with open(tmp,"wb") as stream: np.save(stream,array,allow_pickle=False)
            os.replace(tmp,root/name)
        finally: Path(tmp).unlink(missing_ok=True)
    record={"schema_version":1,"split":split,"track":track,"room":room,"sources":list(SOURCES),"sample_rate":SAMPLE_RATE,"samples":SAMPLES,
            "mixture_sha256":sha256_file(root/"mixture.npy"),"targets_sha256":sha256_file(root/"targets.npy"),
            "source_records":source_records,"rir_records":rir_records,"processing":rows,
            "wet_rms":{name:_rms(targets[i]) for i,name in enumerate(SOURCES)},"mixture_peak":float(np.max(np.abs(mixture)))}
    atomic_json(root/"complete.json",record)
    return record
