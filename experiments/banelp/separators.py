"""FP32 CUDA-only separators (Demucs, MSST) and checkpoint provenance/eligibility checks."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib, subprocess, sys, types
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
                 msst_commit: str|None=None, inference_overrides: dict|None=None) -> dict:
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
            "url":url,"revision":revision,"model_type":model_type,"config_file":config_file,"msst_commit":msst_commit,"inference_overrides":inference_overrides}
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

def load_registered_demucs(registered: dict, model_name: str, *, remote_root: str|Path|None=None, load_model=None, bag_cls=None, torch_load=None):
    """Hash-verify, match against the resolved member set, then load from exactly those local files (no name resolution)."""
    if registered.get("model_id")!=model_name: raise ValueError("registered model record does not match requested model")
    verify_model_record(registered)
    expected=demucs_expected_members(model_name,remote_root=remote_root)
    paths=match_registered_members(registered,expected)
    if load_model is None:
        from demucs.states import load_model
    if torch_load is None:
        import torch
        torch_load=torch.load
    # weights_only=False equals the official path (demucs/repo.py:65 torch.hub.load_state_dict_from_url -> load_model(package_dict)); it is applied only to files whose sha256 was verified above
    models=[load_model(torch_load(path, map_location="cpu", weights_only=False)) for path in paths]
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

MSST_PINNED_COMMIT = "f17924ddb1817cdbae6353706708c4518dd7cd14"  # tag v1.0.23 of ZFTurbo/Music-Source-Separation-Training
_API_ATTRS = ("get_model_from_config", "demix", "load_start_checkpoint", "normalize_audio", "denormalize_audio")
_API_MODULES = ("utils", "utils.settings", "utils.model_utils", "utils.audio_utils")

MSST_NORMALIZATION_POLICY = ("official inference.py:104-106,142-145: if config.inference['normalize'] is True the mixture goes through utils.audio_utils.normalize_audio before demix "
                             "and every estimate through denormalize_audio after it; otherwise nothing is applied")
MSST_STD_ZERO_POLICY = ("std(mono mix)==0 (silent/constant input): normalize_audio would divide by zero (NaN); here the mix is centred only, the estimates are restored with +mean "
                        "(std recorded as 1.0) and `centered_only` is recorded")
MSST_FP32_POLICY = "config.training.use_amp is forced to False before demix, so demix's autocast is disabled (FP32, same as the HTDemucs baseline); the original config value is recorded"

def load_msst_api(root: str|Path, commit: str):
    """Import the official MSST functions from an unmodified checkout pinned at `commit` (kept under $R/src, not in Git).

    `get_model_from_config` imports `models.*` lazily, so the namespace carries `sys_path()`, a context that puts the checkout on `sys.path` for model construction."""
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
            for attr in _API_ATTRS:
                if attr not in found and callable(getattr(module,attr,None)): found[attr]=getattr(module,attr)
    finally: sys.path.remove(str(root))
    missing=[a for a in _API_ATTRS if a not in found]
    if missing: raise RuntimeError(f"MSST checkout {commit} does not expose {missing}")
    @contextlib.contextmanager
    def sys_path():
        sys.path.insert(0,str(root))
        try: yield
        finally:
            try: sys.path.remove(str(root))
            except ValueError: pass
    return types.SimpleNamespace(**found,sys_path=sys_path,root=root,commit=commit)

def msst_normalize_decision(config) -> bool:
    """Official condition: `"normalize" in config.inference and config.inference["normalize"] is True`."""
    return "normalize" in config.inference and config.inference["normalize"] is True

def official_msst_separate(api, config, model, mixture: np.ndarray, device, *, model_type: str) -> tuple[dict, dict]:
    """MSST inference.py:104-106 / 142-145 around `demix`: optional normalize -> demix -> denormalize each instrument.

    `mixture` is (channels, samples) float32 and is not modified. A silent input (std==0) is centred only (MSST_STD_ZERO_POLICY).
    Returns ({instrument: estimate}, info). `config.training.use_amp` must already be False (checked here, not just assumed)."""
    if getattr(getattr(config,"training",None),"use_amp",True) is not False: raise RuntimeError("config.training.use_amp must be False: MSST demix would run under fp16 autocast")
    normalize = msst_normalize_decision(config); info = {"normalize": normalize, "centered_only": False, "mean": None, "std": None}
    mix = mixture
    if normalize:
        mono = mix.mean(0); mean, std = mono.mean(), mono.std()
        if bool(std > 0): mix, params = api.normalize_audio(mix)
        else:
            info["centered_only"] = True; mix = mix - mean; params = {"mean": mean, "std": np.float32(1.0)}
        info["mean"], info["std"] = float(params["mean"]), float(params["std"])
    waveforms = api.demix(config, model, mix, device, model_type=model_type, pbar=False)
    if not isinstance(waveforms, dict): raise ValueError("MSST demix must return a dict of instruments for a multi-stem model")
    if normalize: waveforms = {name: api.denormalize_audio(estimate, params) for name, estimate in waveforms.items()}
    return waveforms, info

def official_msst_reference(api, config, model, mixture, device, *, model_type):
    """Verbatim sequence of MSST inference.py:104-106,142-145 (no std==0 handling, bigshifts=1 equals one plain demix); tests compare the adapter against it."""
    mix = mixture
    if "normalize" in config.inference:
        if config.inference["normalize"] is True:
            mix, norm_params = api.normalize_audio(mix)
    waveforms_orig = api.demix(config, model, mix, device, model_type=model_type, pbar=False)
    out = {}
    for instr in waveforms_orig:
        estimates = waveforms_orig[instr]
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                estimates = api.denormalize_audio(estimates, norm_params)
        out[instr] = estimates
    return out

def load_msst_checkpoint(api, model, path: str|Path, model_type: str) -> None:
    """The official strict load: `torch.load(..., map_location='cpu', weights_only=False)` then `load_start_checkpoint(args, model, ckpt, type_='inference')` (handles state / state_dict / model_state_dict).

    The file is hash-verified by the caller. The official function prints a line; it is sent to stderr so CLI stdout stays JSON."""
    import torch
    args = argparse.Namespace(start_check_point=str(path), model_type=model_type, lora_checkpoint_loralib=None, lora_checkpoint=None, load_only_compatible_weights=False)
    checkpoint = torch.load(str(path), weights_only=False, map_location="cpu")
    with contextlib.redirect_stdout(sys.stderr): api.load_start_checkpoint(args, model, checkpoint, type_="inference")

class MSSTSeparator:
    """BS-RoFormer / SCNet from the registered config + hash-verified checkpoint, FP32, MSST's own inference sequence (normalize -> chunked overlap-add `demix` -> denormalize).

    Output is reordered to the contract source order. Production uses CUDA only; `device` exists so unit tests can use a fake model.
    `batch_size` overrides `config.inference.batch_size` (batching does not change eval-mode results); the override and the effective values are recorded in `self.attestation`."""
    def __init__(self, *, registered: dict, runtime_dir: str|Path, device: str="cuda", api=None, model_name: str|None=None, segment=None, overlap=None, batch_size: int|None=None):
        import torch
        if registered.get("family")!="msst": raise ValueError("registered model is not an MSST model")
        if device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        verify_model_record(registered)
        self.torch, self.device, self.registered = torch, device, registered
        self.api = api if api is not None else load_msst_api(Path(runtime_dir)/"src"/"MSST", registered["msst_commit"])
        config_path=registered["config_file"]["path"]; model_type=registered["model_type"]
        sys_path=getattr(self.api,"sys_path",contextlib.nullcontext)
        with sys_path(): model, self.config = self.api.get_model_from_config(model_type, config_path)
        if "model_type" in self.config.training: model_type=self.config.training.model_type  # inference.py:210
        load_msst_checkpoint(self.api, model, registered["members"][0]["path"], model_type)
        instruments=list(getattr(getattr(self.config,"training",None),"instruments",[]) or [])
        if instruments and instruments!=registered["native_source_order"]: raise ValueError(f"config instruments {instruments} differ from registered native order {registered['native_source_order']}")
        config_use_amp=getattr(self.config.training,"use_amp",True); self.config.training.use_amp=False
        config_batch=int(self.config.inference.batch_size)
        override=batch_size if batch_size is not None else (registered.get("inference_overrides") or {}).get("batch_size")
        if override is not None:
            if int(override)<1: raise ValueError("batch_size override must be >= 1")
            self.config.inference.batch_size=int(override)
        self.model=model.float().to(device).eval(); self.model_type=model_type; self.native_sources=tuple(registered["native_source_order"])
        self.attestation={"msst_commit":registered["msst_commit"],"model_type":model_type,"fp32":True,"autocast_enabled":False,"config_use_amp":bool(config_use_amp),"use_amp_forced_false":True,
                          "normalize":msst_normalize_decision(self.config),"config_batch_size":config_batch,"batch_size":int(self.config.inference.batch_size),"batch_size_overridden":override is not None,
                          "num_overlap":int(self.config.inference.num_overlap),"loader":"load_start_checkpoint(type_='inference'), strict","policies":[MSST_FP32_POLICY,MSST_NORMALIZATION_POLICY,MSST_STD_ZERO_POLICY]}
        self.last_normalization=None
    def __call__(self, mixture: np.ndarray) -> np.ndarray:
        if mixture.ndim!=2 or mixture.shape[0]!=2: raise ValueError("mixture must be (2,N)")
        out,self.last_normalization=official_msst_separate(self.api,self.config,self.model,np.ascontiguousarray(mixture,dtype=np.float32).copy(),self.device,model_type=self.model_type)
        if set(out)!=set(SOURCES): raise ValueError(f"MSST output keys {sorted(out)} are not the four contract sources")
        result=np.stack([np.asarray(out[name]) for name in SOURCES]).astype(np.float32)
        if result.shape[:2]!=(4,2) or result.shape[-1]!=mixture.shape[-1] or not np.isfinite(result).all(): raise ValueError(f"invalid MSST output {result.shape}")
        return result

# ---- registration allow/deny table -----------------------------------------------------------------------------------

ELIGIBILITY_FILE = Path(__file__).resolve().parent / "checkpoint_eligibility.json"

def load_eligibility_table() -> dict:
    import json
    return json.loads(ELIGIBILITY_FILE.read_text(encoding="utf-8"))

def check_registration_allowed(*, model_id: str, filenames: list[str], status: str, release: str|None, family: str, table: dict|None=None,
                               checkpoint_sha256: dict|None=None, config_name: str|None=None, config_sha256: str|None=None) -> None:
    """Excluded checkpoints never register; a listed file needs its recorded release, sha256 and (MSST) config name + sha256.

    documented-train-only is accepted only for listed files. `checkpoint_sha256` maps file name -> sha256 of the file being registered."""
    table = table or load_eligibility_table()
    for entry in table["excluded"]:
        if entry.get("model_id") == model_id or entry.get("file") in filenames: raise ValueError(f"{model_id} / {filenames} is excluded: {entry['reason']}")
    allowed = {e["file"]: e for e in table["allowed"]}
    if status == "documented-train-only":
        for name in filenames:
            if name not in allowed: raise ValueError(f"documented-train-only is only accepted for the files listed in checkpoint_eligibility.json; {name} is not listed")
    for name in filenames:
        entry = allowed.get(name)
        if entry is None: continue
        if family != entry["family"]: raise ValueError(f"{name} is a {entry['family']} checkpoint")
        if entry["release"] is not None and entry["release"] != release: raise ValueError(f"{name}: release {release!r} differs from the recorded {entry['release']!r}")
        have = (checkpoint_sha256 or {}).get(name)
        if entry.get("sha256") and have != entry["sha256"]: raise ValueError(f"{name}: sha256 {have} differs from the recorded {entry['sha256']}")
        want_config = entry.get("config")
        if want_config:
            if config_name != want_config["file"]: raise ValueError(f"{name}: config {config_name!r} differs from the recorded {want_config['file']!r}")
            if config_sha256 != want_config["sha256"]: raise ValueError(f"{name}: config sha256 {config_sha256} differs from the recorded {want_config['sha256']}")
