"""FP32 CUDA-only separators (Demucs, MSST) and checkpoint provenance/eligibility checks."""
from __future__ import annotations
import hashlib, importlib, subprocess, sys, types
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from .contracts import SOURCES, SAMPLE_RATE, canonical_json, sha256_file

TRAIN_DATA_STATUSES = {"verified-excluded", "verified-included", "unknown", "documented-train-only"}
DOCUMENTED_LABEL_KO = "문서상 train-only, 곡별 감사 미확인"
DOCUMENTED_LABEL_EN = "documented train-only (distribution table); per-song audit unconfirmed"
FAMILIES = ("demucs", "msst")

def declaration_record(url: str|None, quote: str|None, release: str|None) -> dict|None:
    """`documented-train-only` evidence: the distribution table's own declaration, quoted, with its release."""
    if url is None and quote is None and release is None: return None
    if not (url and quote and release): raise ValueError("a declaration needs declaration_url, declaration_quote and release")
    return {"url": url, "quote": quote, "release": release}

def record_eligibility(record: dict) -> dict:
    """Eligibility recomputed from status + evidence, never from a stored flag.

    clean winner = verified-excluded with evidence; candidate use additionally allows documented-train-only with a full declaration."""
    status = record.get("train_data_status")
    if status not in TRAIN_DATA_STATUSES: raise ValueError(f"invalid training-data status {status!r}")
    verified = status == "verified-excluded" and bool(record.get("training_data_evidence"))
    declaration = record.get("declaration") or {}
    documented = status == "documented-train-only" and all(declaration.get(k) for k in ("url", "quote", "release"))
    return {"train_data_status": status, "clean_winner_eligible": verified, "candidate_eligible": verified or documented,
            "label_ko": DOCUMENTED_LABEL_KO if documented else None, "label_en": DOCUMENTED_LABEL_EN if documented else None}

def checkpoint_record(path: str|Path, *, model_id: str, source_order: list[str], train_data_status: str, url: str|None=None, revision: str|None=None,
                      training_data_evidence: list[str]|None=None, declaration: dict|None=None) -> dict:
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    if tuple(source_order)!=SOURCES: raise ValueError("checkpoint must emit exactly the contracted four-source order")
    if train_data_status not in TRAIN_DATA_STATUSES: raise ValueError("invalid training-data status")
    record={"model_id":model_id,"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size,
            "source_order":list(source_order),"train_data_status":train_data_status,"training_data_evidence":training_data_evidence or [],
            "declaration":declaration,"url":url,"revision":revision}
    record["clean_winner_eligible"]=record_eligibility(record)["clean_winner_eligible"]
    return record

def model_record(*, model_id: str, checkpoints: list[str|Path], native_source_order: list[str], samplerate: int,
                 train_data_status: str, training_data_evidence: list[str]|None=None, url: str|None=None, revision: str|None=None,
                 family: str="demucs", model_type: str|None=None, config_file: dict|None=None, declaration: dict|None=None,
                 msst_commit: str|None=None) -> dict:
    """One `manifests/models.json` entry: every bag member hashed, native order mapped to the contract order."""
    if not checkpoints: raise ValueError("at least one checkpoint file is required")
    if family not in FAMILIES: raise ValueError(f"family must be one of {FAMILIES}")
    if train_data_status == "documented-train-only" and not declaration: raise ValueError("documented-train-only needs declaration_url, declaration_quote and release")
    if family == "msst":
        if len(checkpoints) != 1: raise ValueError("an MSST model has exactly one checkpoint")
        if not (model_type and config_file and msst_commit): raise ValueError("an MSST model needs model_type, a hashed config file and the pinned MSST commit")
    native=tuple(native_source_order)
    if len(native)!=4 or set(native)!=set(SOURCES): raise ValueError(f"model is not an exact four-source model: {native}")
    if int(samplerate)!=SAMPLE_RATE: raise ValueError(f"model samplerate {samplerate} differs from contract {SAMPLE_RATE}")
    members=[checkpoint_record(path,model_id=model_id,source_order=list(SOURCES),train_data_status=train_data_status,url=url,
                               revision=revision,training_data_evidence=training_data_evidence,declaration=declaration) for path in checkpoints]
    hashes=[m["sha256"] for m in members]
    combined=hashes[0] if len(hashes)==1 else hashlib.sha256(canonical_json(sorted(hashes))).hexdigest()
    record={"model_id":model_id,"family":family,"sha256":combined,"members":members,"bag_size":len(members),"native_source_order":list(native),
            "source_order":list(SOURCES),"reorder_index":[native.index(name) for name in SOURCES],"samplerate":int(samplerate),
            "train_data_status":train_data_status,"training_data_evidence":training_data_evidence or [],"declaration":declaration,
            "url":url,"revision":revision,"model_type":model_type,"config_file":config_file,"msst_commit":msst_commit}
    eligibility=record_eligibility(record)
    record["clean_winner_eligible"]=eligibility["clean_winner_eligible"]; record["candidate_eligible"]=eligibility["candidate_eligible"]
    return record

def verify_model_record(record: dict) -> None:
    """Refuse to load weights whose bytes differ from the registered hash."""
    for member in record.get("members",[]):
        path=Path(member["path"])
        if not path.is_file() or sha256_file(path)!=member["sha256"]: raise ValueError(f"registered checkpoint missing or changed: {path}")
    config=record.get("config_file")
    if config and (not Path(config["path"]).is_file() or sha256_file(config["path"])!=config["sha256"]): raise ValueError(f"registered config missing or changed: {config['path']}")

def state_dict_sha256(state) -> str:
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()+b"\0"); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

# ---- Demucs: registered files <-> what get_model(name) would load -------------------------------------------------

def demucs_remote_root() -> Path:
    import demucs
    return Path(demucs.__file__).resolve().parent / "remote"

def demucs_expected_members(model_name: str, *, remote_root: str|Path|None=None) -> dict:
    """Member files `demucs.pretrained.get_model(model_name)` resolves: bag yaml (or single signature) -> files.txt URL -> basename.

    Mirrors RemoteRepo/BagOnlyRepo of demucs 4.0.1 without downloading anything."""
    import yaml
    root = Path(remote_root) if remote_root is not None else demucs_remote_root()
    urls, prefix = {}, ""
    for line in (root/"files.txt").read_text().split("\n"):
        line=line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("root:"): prefix=line.split(":",1)[1].strip(); continue
        urls[line.split("-",1)[0]]=prefix+line
    bag_file=root/f"{model_name}.yaml"
    if bag_file.is_file():
        bag=yaml.safe_load(bag_file.read_text()); signatures=list(bag["models"]); weights=bag.get("weights"); segment=bag.get("segment"); is_bag=True
    elif model_name in urls: signatures,weights,segment,is_bag=[model_name],None,None,False
    else: raise ValueError(f"{model_name} is neither a bag yaml nor a signature in demucs' files.txt")
    files=[]
    for sig in signatures:
        if sig not in urls: raise ValueError(f"signature {sig} of {model_name} has no entry in files.txt")
        name=Path(urlparse(urls[sig]).path).name
        stem=name.rsplit(".",1)[0]
        files.append({"signature":sig,"filename":name,"hash_prefix":stem.split("-",1)[1] if "-" in stem else "","url_path":urls[sig]})
    return {"model_name":model_name,"bag":is_bag,"weights":weights,"segment":segment,"members":files}

def match_registered_members(registered: dict, expected: dict) -> list[Path]:
    """Registered member basenames/hashes must equal the resolved set exactly; returns paths in the loader's member order."""
    members=registered.get("members",[])
    want=[m["filename"] for m in expected["members"]]
    have=[Path(m["path"]).name for m in members]
    if len(members)!=len(want) or sorted(have)!=sorted(want):
        raise ValueError(f"registered checkpoints {sorted(have)} differ from the files get_model({expected['model_name']!r}) resolves {sorted(want)}")
    by_name={Path(m["path"]).name:m for m in members}; ordered=[]
    for entry in expected["members"]:
        member=by_name[entry["filename"]]
        if entry["hash_prefix"] and not member["sha256"].startswith(entry["hash_prefix"]):
            raise ValueError(f"registered {entry['filename']} sha256 {member['sha256']} does not start with demucs' hash prefix {entry['hash_prefix']}")
        ordered.append(Path(member["path"]))
    return ordered

def load_registered_demucs(registered: dict, model_name: str, *, remote_root: str|Path|None=None, load_model=None, bag_cls=None):
    """Hash-verify, match against the resolved member set, then load from exactly those local files (no name resolution)."""
    if registered.get("model_id")!=model_name: raise ValueError("registered model record does not match requested model")
    verify_model_record(registered)
    expected=demucs_expected_members(model_name,remote_root=remote_root)
    paths=match_registered_members(registered,expected)
    if load_model is None:
        from demucs.states import load_model
    models=[load_model(path) for path in paths]
    if expected["bag"]:
        if bag_cls is None:
            from demucs.apply import BagOfModels as bag_cls
        model=bag_cls(models,expected["weights"],expected["segment"])
    else: model=models[0]
    model.eval()
    return model

# ---- Demucs official pre/post normalization (demucs 4.0.1 separate.py:168-177) --------------------------------------

STD_ZERO_POLICY = ("std(ref)==0 (silent/constant input, or a single sample): the official path would divide by zero and return NaN; "
                   "here the input is centred only (scale 1), the output is restored with +mean, and `centered_only` is recorded")

def official_demucs_apply(model, wav, *, device, shifts=0, split=True, overlap=.25, segment=None, apply_model=None):
    """Official CLI wrapper around `apply_model`: ref=wav.mean(0); centre and divide by ref.std(); apply; restore std then mean.

    `wav` is a (channels, samples) tensor on the compute device and is not modified; returns (sources (S,C,N), info)."""
    if apply_model is None:
        from demucs.apply import apply_model
    wav = wav.clone()
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    centered_only = not bool(std > 0)
    scale = std.new_ones(()) if centered_only else std
    wav -= mean
    wav /= scale
    sources = apply_model(model, wav[None], device=device, shifts=shifts, split=split, overlap=overlap, progress=False, num_workers=0, segment=segment)[0]
    sources *= scale
    sources += mean
    return sources, {"ref_mean": float(mean), "ref_std": float(std), "centered_only": centered_only}

def official_demucs_reference(model, wav, *, device, shifts=0, split=True, overlap=.25, segment=None, apply_model=None):
    """Verbatim copy of demucs 4.0.1 separate.py:168-177 (no std==0 handling); smoke compares the wrapper against it."""
    if apply_model is None:
        from demucs.apply import apply_model
    wav = wav.clone()
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    sources = apply_model(model, wav[None], device=device, shifts=shifts, split=split, overlap=overlap, progress=False, num_workers=0, segment=segment)[0]
    sources *= ref.std()
    sources += ref.mean()
    return sources

class HTDemucsCUDA:
    def __init__(self, *, model_name="htdemucs", segment=7.0, overlap=.25, registered: dict|None=None):
        try:
            import torch
        except ImportError as exc: raise RuntimeError("torch/demucs are not installed in the research environment; run bootstrap.sh") from exc
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        if registered is None: raise ValueError("a registered, hash-verified model record is required; weights are loaded only from its files")
        self.torch=torch; self.model=load_registered_demucs(registered,model_name)
        native_sources=tuple(getattr(self.model,"sources",()))
        if len(native_sources)!=4 or set(native_sources)!=set(SOURCES): raise ValueError(f"checkpoint is not an exact four-source model: {native_sources}")
        if list(native_sources)!=registered["native_source_order"] or getattr(self.model,"samplerate",None)!=registered["samplerate"]:
            raise ValueError("loaded model differs from its registered source order or samplerate")
        self.native_sources=native_sources
        self.source_order=[native_sources.index(name) for name in SOURCES]
        max_segment=getattr(self.model,"segment",None)
        if max_segment is not None and segment>max_segment: raise ValueError(f"requested segment {segment}s exceeds model limit {max_segment}s")
        self.segment,self.overlap=segment,overlap
        self.model.to("cuda").eval()
        self.device="cuda:0"; self.last_normalization=None
    def __call__(self, mixture: np.ndarray) -> np.ndarray:
        torch=self.torch
        if mixture.ndim!=2 or mixture.shape[0]!=2: raise ValueError("mixture must be (2,N)")
        audio=torch.from_numpy(np.ascontiguousarray(mixture,dtype=np.float32)).to(device="cuda",dtype=torch.float32)
        with torch.inference_mode():
            out,self.last_normalization=official_demucs_apply(self.model,audio,device="cuda",shifts=0,split=True,overlap=self.overlap,segment=self.segment)
        result=out[self.source_order].detach().to("cpu").numpy().astype(np.float32)
        if result.shape[:2]!=(4,2) or result.shape[-1]!=mixture.shape[-1] or not np.isfinite(result).all(): raise ValueError(f"invalid Demucs output {result.shape}")
        if audio.device.type!="cuda" or out.device.type!="cuda": raise RuntimeError("inference tensors were not on CUDA")
        return result

# ---- MSST (BS-RoFormer, SCNet) ----------------------------------------------------------------------------------------

_API_MODULES = ("utils", "utils.settings", "utils.model_utils", "utils.audio_utils")

def load_msst_api(root: str|Path, commit: str):
    """Import `get_model_from_config` and `demix` from an unmodified MSST checkout pinned at `commit` (kept under $R/src, not in Git)."""
    root=Path(root).resolve()
    if not (root/".git").exists(): raise FileNotFoundError(f"MSST checkout missing: {root}")
    def git(*a): return subprocess.run(["git","-C",str(root),*a],capture_output=True,text=True,check=True).stdout.strip()
    head=git("rev-parse","HEAD")
    if head!=commit: raise ValueError(f"MSST checkout is at {head}, registered pin is {commit}")
    if git("status","--porcelain","--untracked-files=no"): raise ValueError("MSST checkout has local modifications")
    sys.path.insert(0,str(root))
    try:
        found={}
        for name in _API_MODULES:
            try: module=importlib.import_module(name)
            except ImportError: continue
            for attr in ("get_model_from_config","demix"):
                if attr not in found and callable(getattr(module,attr,None)): found[attr]=getattr(module,attr)
    finally: sys.path.remove(str(root))
    missing=[a for a in ("get_model_from_config","demix") if a not in found]
    if missing: raise RuntimeError(f"MSST checkout {commit} does not expose {missing}")
    return types.SimpleNamespace(**found)

class MSSTSeparator:
    """BS-RoFormer / SCNet from the registered config + hash-verified checkpoint, FP32, MSST's own chunked overlap-add `demix`.

    Output is reordered to the contract source order. Production uses CUDA only; `device` exists so unit tests can use a fake model."""
    def __init__(self, *, registered: dict, runtime_dir: str|Path, device: str="cuda", api=None, model_name: str|None=None, segment=None, overlap=None):
        import torch
        if registered.get("family")!="msst": raise ValueError("registered model is not an MSST model")
        if device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        verify_model_record(registered)
        self.torch, self.device, self.registered = torch, device, registered
        self.api = api if api is not None else load_msst_api(Path(runtime_dir)/"src"/"MSST", registered["msst_commit"])
        config_path=registered["config_file"]["path"]; model_type=registered["model_type"]
        model, self.config = self.api.get_model_from_config(model_type, config_path)
        state=torch.load(registered["members"][0]["path"], map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict): state=state["state_dict"]
        model.load_state_dict(state)
        instruments=list(getattr(getattr(self.config,"training",None),"instruments",[]) or [])
        if instruments and instruments!=registered["native_source_order"]: raise ValueError(f"config instruments {instruments} differ from registered native order {registered['native_source_order']}")
        self.model=model.float().to(device).eval(); self.model_type=model_type; self.native_sources=tuple(registered["native_source_order"])
    def __call__(self, mixture: np.ndarray) -> np.ndarray:
        if mixture.ndim!=2 or mixture.shape[0]!=2: raise ValueError("mixture must be (2,N)")
        with self.torch.inference_mode():
            out=self.api.demix(self.config,self.model,np.ascontiguousarray(mixture,dtype=np.float32).copy(),self.device,model_type=self.model_type,pbar=False)
        if isinstance(out,dict):
            if set(out)!=set(SOURCES): raise ValueError(f"MSST output keys {sorted(out)} are not the four contract sources")
            result=np.stack([np.asarray(out[name]) for name in SOURCES])
        else:
            arr=np.asarray(out)
            if arr.ndim!=3 or arr.shape[0]!=4: raise ValueError(f"MSST output must be (4,2,N), got {arr.shape}")
            result=arr[[self.native_sources.index(name) for name in SOURCES]]
        result=result.astype(np.float32)
        if result.shape[:2]!=(4,2) or result.shape[-1]!=mixture.shape[-1] or not np.isfinite(result).all(): raise ValueError(f"invalid MSST output {result.shape}")
        return result

# ---- registration allow/deny table -----------------------------------------------------------------------------------

ELIGIBILITY_FILE = Path(__file__).resolve().parent / "checkpoint_eligibility.json"

def load_eligibility_table() -> dict:
    import json
    return json.loads(ELIGIBILITY_FILE.read_text(encoding="utf-8"))

def check_registration_allowed(*, model_id: str, filenames: list[str], status: str, release: str|None, family: str, table: dict|None=None) -> None:
    """Excluded checkpoints never register; documented-train-only needs a listed file with its recorded release."""
    table = table or load_eligibility_table()
    for entry in table["excluded"]:
        if entry.get("model_id") == model_id or entry.get("file") in filenames: raise ValueError(f"{model_id} / {filenames} is excluded: {entry['reason']}")
    if status != "documented-train-only": return
    allowed = {e["file"]: e for e in table["allowed"]}
    for name in filenames:
        if name not in allowed: raise ValueError(f"documented-train-only is only accepted for the files listed in checkpoint_eligibility.json; {name} is not listed")
        entry = allowed[name]
        if family != entry["family"]: raise ValueError(f"{name} is a {entry['family']} checkpoint")
        if entry["release"] is not None and entry["release"] != release: raise ValueError(f"{name}: release {release!r} differs from the recorded {entry['release']!r}")
