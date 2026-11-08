"""Manifest validation and deterministic wet-target dataset construction."""
from __future__ import annotations
import json, math, os, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from .contracts import SOURCES, SAMPLE_RATE, SAMPLES, cache_key, deterministic_split, sha256_file, validate_runtime_path, validate_tensors

ROOMS=("R1","R2","R3","R4")

def crop_center(audio: np.ndarray, length: int=SAMPLES) -> tuple[np.ndarray,int]:
    if audio.ndim != 2: raise ValueError("audio must be (samples,channels)")
    if len(audio)<length: raise ValueError(f"track too short: {len(audio)} < {length}")
    start=(len(audio)-length)//2
    return audio[start:start+length],start

SILENCE_RMS=1e-12

def _rms(audio): return float(np.sqrt(np.mean(np.asarray(audio,dtype=np.float64)**2)))

def apply_rir(dry: np.ndarray, rir: np.ndarray, *, onset_offset: int=0, rir_channel: int|None=None) -> tuple[np.ndarray,dict]:
    """Same mono RIR on both channels, full convolution, keep the first N samples, restore the stem RMS.

    `dry` is soundfile layout (N,2); the result is channels-first (2,N) like every tensor consumer."""
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
    dry_rms=_rms(dry); full_length=len(dry)+len(rir)-1
    base={"dry_rms":dry_rms,"dry_length":len(dry),"full_length":full_length,"kept_length":len(dry),"onset_offset":onset_offset,"rir_channel":selected_channel}
    if dry_rms<=SILENCE_RMS:
        return np.zeros((2,len(dry)),dtype=np.float32),{**base,"silent":True,"wet_rms_before_gain":0.0,"gain":0.0,"wet_rms_after_gain":0.0,"tail_energy":0.0,"tail_energy_fraction":0.0}
    full=np.stack([fftconvolve(np.asarray(dry[:,ch],dtype=np.float64),np.asarray(rir,dtype=np.float64),mode="full") for ch in range(2)])
    cropped=full[:,:len(dry)]
    wet_rms=_rms(cropped)
    if wet_rms<=SILENCE_RMS: raise ValueError("non-silent dry stem produced near-zero wet RMS")
    gain=dry_rms/wet_rms; out=(cropped*gain).astype(np.float32)
    tail=float(np.sum(full[:,len(dry):]**2)); total=float(np.sum(full**2))
    return out,{**base,"silent":False,"wet_rms_before_gain":wet_rms,"gain":gain,"wet_rms_after_gain":_rms(out),
                "tail_energy":tail,"tail_energy_fraction":tail/total if total>0 else 0.0}

def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int=SAMPLE_RATE) -> np.ndarray:
    if src_rate==dst_rate:return np.asarray(audio,dtype=np.float32)
    common=math.gcd(src_rate,dst_rate)
    return resample_poly(audio,dst_rate//common,src_rate//common,axis=0).astype(np.float32)

def make_sample(dry_stems: dict[str,np.ndarray], rir_by_source: dict[str,np.ndarray], *, start_samples: dict[str,int]|None=None,
                samples: int=SAMPLES) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    """Return mixture (2,N), wet targets (4,2,N) in SOURCES order, and per-stem processing records."""
    if set(dry_stems)!=set(SOURCES) or set(rir_by_source)!=set(SOURCES): raise ValueError("exactly four named stems and RIRs required")
    wet=[]; rows=[]
    for name in SOURCES:
        stem=np.asarray(dry_stems[name])
        if start_samples is None:
            segment,start=crop_center(stem,samples)
        else:
            start=start_samples[name]; segment=stem[start:start+samples]
        if len(segment)!=samples: raise ValueError(f"short stem {name}")
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

def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try:
        with open(tmp,"wb") as stream: np.save(stream,array,allow_pickle=False)
        os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)

def write_cache_entry(root: Path, arrays: dict[str,np.ndarray], record: dict) -> dict:
    """Arrays first, `complete.json` last; an old marker is removed before any array is replaced."""
    root.mkdir(parents=True,exist_ok=True); (root/"complete.json").unlink(missing_ok=True)
    for name,array in arrays.items(): _save_npy_atomic(root/f"{name}.npy",array)
    record={**record,"files":{name:{"sha256":sha256_file(root/f"{name}.npy"),"shape":list(array.shape),"dtype":str(array.dtype)} for name,array in arrays.items()}}
    atomic_json(root/"complete.json",record)
    return record

def read_cache_entry(root: Path, names: tuple[str,...], *, cache_key: str, identity: dict) -> tuple[dict[str,np.ndarray],dict]:
    """Load a cache entry only when its `complete.json` matches the key, identity and file hashes."""
    marker=root/"complete.json"
    if not marker.is_file(): raise FileNotFoundError(f"cache entry incomplete (no complete.json): {root}")
    record=json.loads(marker.read_text(encoding="utf-8"))
    if record.get("cache_key")!=cache_key: raise ValueError(f"stale cache entry (cache key mismatch): {root}")
    for key,value in identity.items():
        if record.get(key)!=value: raise ValueError(f"cache entry identity mismatch for {key}: {root}")
    arrays={}
    for name in names:
        path=root/f"{name}.npy"; expected=record.get("files",{}).get(name,{})
        if not path.is_file() or sha256_file(path)!=expected.get("sha256"): raise ValueError(f"cache file missing or changed: {path}")
        arrays[name]=np.load(path,allow_pickle=False)
    return arrays,record

SYNTHESIS_CONTRACT={"crop":"center floor((len-N)/2)","convolution":"full, same mono RIR on both channels","onset_offset":0,
                    "truncate":"keep first N","rms_match":"per-stem two-channel RMS to dry crop","mixture":"float64 sum -> float32",
                    "silence_rms":SILENCE_RMS,"clipping":"none","stem_position":"stem i -> room position i"}

def wet_cache_key(*, code_hash: str, lock_hash: str, manifest_hash: str, rir_manifest_hash: str, samples: int) -> str:
    return cache_key(code_revision=code_hash,lock_hash=lock_hash,source_manifest_hash=manifest_hash,rir_manifest_hash=rir_manifest_hash,
                     model_hashes=[],inference={},synthesis={**SYNTHESIS_CONTRACT,"samples":samples})

def wet_root(runtime_dir: str|Path, split: str, track: str, room: str) -> Path:
    if split not in {"dev","evaluation"}: raise ValueError("split must be dev or evaluation")
    if room not in ROOMS: raise ValueError("room must be R1..R4")
    return validate_runtime_path(Path(runtime_dir)/"cache/wet"/split/track/room,runtime_dir)

def save_wet_sample(runtime_dir: str|Path, *, split: str, track: str, room: str,
                   dry_stems: dict[str,np.ndarray], rir_by_source: dict[str,np.ndarray],
                   source_records: dict[str,dict], rir_records: dict[str,dict], start_samples: dict[str,int]|None=None,
                   cache_key: str, samples: int=SAMPLES) -> dict:
    """Atomically persist one channels-first wet-target sample and its content-bound `complete.json`."""
    root=wet_root(runtime_dir,split,track,room)
    mixture,targets,rows=make_sample(dry_stems,rir_by_source,start_samples=start_samples,samples=samples)
    validate_tensors(mixture,targets,samples=samples)
    if not np.allclose(targets.astype(np.float64).sum(axis=0),mixture.astype(np.float64),rtol=0,atol=1e-6): raise ValueError("wet target sum does not equal mixture")
    record={"schema_version":2,"cache_key":cache_key,"split":split,"track":track,"room":room,"sources":list(SOURCES),"sample_rate":SAMPLE_RATE,"samples":samples,
            "layout":{"mixture":"(2,N)","targets":"(4,2,N)"},"source_records":source_records,"rir_records":rir_records,"processing":rows,
            "wet_rms":{name:_rms(targets[i]) for i,name in enumerate(SOURCES)},"mixture_peak":float(np.max(np.abs(mixture))),
            "target_peak":float(np.max(np.abs(targets)))}
    return write_cache_entry(root,{"mixture":mixture,"targets":targets},record)

def load_wet(runtime_dir: str|Path, *, split: str, track: str, room: str, cache_key: str, samples: int=SAMPLES) -> tuple[np.ndarray,np.ndarray,dict]:
    arrays,record=read_cache_entry(wet_root(runtime_dir,split,track,room),("mixture","targets"),cache_key=cache_key,
                                   identity={"split":split,"track":track,"room":room,"samples":samples})
    x,targets=arrays["mixture"].astype(np.float32),arrays["targets"].astype(np.float32)
    validate_tensors(x,targets,samples=samples)
    return x,targets,record

def build_wet_cache(runtime_dir: str|Path, *, dataset: dict, rirs: dict[str,dict[str,np.ndarray]], rir_manifest: dict,
                    cache_key: str, samples: int=SAMPLES, splits=("dev","evaluation"), limit: int|None=None) -> dict:
    """Create every missing or stale wet entry; valid entries with the same key are reused."""
    runtime=Path(runtime_dir).resolve(); summary={"written":0,"reused":0,"tracks":0}
    stem_position=rir_manifest["stem_position"]
    tracks=[t for t in dataset["tracks"] if t["split"] in splits][:limit]
    for entry in tracks:
        pending=[]
        for room in ROOMS:
            try: read_cache_entry(wet_root(runtime,entry["split"],entry["name"],room),("mixture","targets"),cache_key=cache_key,
                                  identity={"split":entry["split"],"track":entry["name"],"room":room,"samples":samples}); summary["reused"]+=1
            except (FileNotFoundError,ValueError): pending.append(room)
        if pending:
            stems={}
            for name in SOURCES:
                src=entry["sources"][name]; path=validate_runtime_path(src["path"],runtime)
                if sha256_file(path)!=src["sha256"]: raise ValueError(f"source stem changed since manifest: {path}")
                stems[name],rate=sf.read(path,always_2d=True,dtype="float32")
                if rate!=SAMPLE_RATE: raise ValueError(f"unexpected sample rate {rate}: {path}")
            for room in pending:
                by_source={name:rirs[room][stem_position[name]] for name in SOURCES}
                rir_records={name:{"room":room,"position":stem_position[name],"sha256":rir_manifest["rooms"][room]["positions"][stem_position[name]]["sha256"],
                                   "kind":rir_manifest["rooms"][room]["kind"]} for name in SOURCES}
                save_wet_sample(runtime,split=entry["split"],track=entry["name"],room=room,dry_stems=stems,rir_by_source=by_source,
                                source_records={n:{"sha256":entry["sources"][n]["sha256"],"frames":entry["sources"][n]["frames"]} for n in SOURCES},
                                rir_records=rir_records,cache_key=cache_key,samples=samples)
                summary["written"]+=1
        summary["tracks"]+=1
    return summary
