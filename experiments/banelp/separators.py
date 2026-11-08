"""FP32 CUDA-only Demucs inference and checkpoint provenance checks."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .contracts import SOURCES, SAMPLE_RATE, canonical_json, sha256_file

def checkpoint_record(path: str|Path, *, model_id: str, source_order: list[str], train_data_status: str, url: str|None=None, revision: str|None=None, training_data_evidence: list[str]|None=None) -> dict:
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    if tuple(source_order)!=SOURCES: raise ValueError("checkpoint must emit exactly the contracted four-source order")
    if train_data_status not in {"verified-excluded","verified-included","unknown"}: raise ValueError("invalid training-data status")
    return {"model_id":model_id,"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size,
            "source_order":list(source_order),"train_data_status":train_data_status,"training_data_evidence":training_data_evidence or [],
            "clean_winner_eligible":train_data_status=="verified-excluded" and bool(training_data_evidence),"url":url,"revision":revision}

def model_record(*, model_id: str, checkpoints: list[str|Path], native_source_order: list[str], samplerate: int,
                 train_data_status: str, training_data_evidence: list[str]|None=None, url: str|None=None, revision: str|None=None) -> dict:
    """One `manifests/models.json` entry: every bag member hashed, native order mapped to the contract order."""
    if not checkpoints: raise ValueError("at least one checkpoint file is required")
    native=tuple(native_source_order)
    if len(native)!=4 or set(native)!=set(SOURCES): raise ValueError(f"model is not an exact four-source model: {native}")
    if int(samplerate)!=SAMPLE_RATE: raise ValueError(f"model samplerate {samplerate} differs from contract {SAMPLE_RATE}")
    members=[checkpoint_record(path,model_id=model_id,source_order=list(SOURCES),train_data_status=train_data_status,url=url,
                               revision=revision,training_data_evidence=training_data_evidence) for path in checkpoints]
    hashes=[m["sha256"] for m in members]
    combined=hashes[0] if len(hashes)==1 else hashlib.sha256(canonical_json(sorted(hashes))).hexdigest()
    return {"model_id":model_id,"sha256":combined,"members":members,"bag_size":len(members),"native_source_order":list(native),
            "source_order":list(SOURCES),"reorder_index":[native.index(name) for name in SOURCES],"samplerate":int(samplerate),
            "train_data_status":train_data_status,"training_data_evidence":training_data_evidence or [],
            "clean_winner_eligible":all(m["clean_winner_eligible"] for m in members),"url":url,"revision":revision}

def verify_model_record(record: dict) -> None:
    """Refuse to load weights whose bytes differ from the registered hash."""
    for member in record.get("members",[]):
        path=Path(member["path"])
        if not path.is_file() or sha256_file(path)!=member["sha256"]: raise ValueError(f"registered checkpoint missing or changed: {path}")

def state_dict_sha256(state) -> str:
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()+b"\0"); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class HTDemucsCUDA:
    def __init__(self, *, model_name="htdemucs", segment=7.0, overlap=.25, registered: dict|None=None):
        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError as exc: raise RuntimeError("torch/demucs are not installed in the research environment; run bootstrap.sh") from exc
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        if registered is not None:
            if registered.get("model_id")!=model_name: raise ValueError("registered model record does not match requested model")
            verify_model_record(registered)
            # demucs resolves remote weights in torch.hub's checkpoint cache; require the verified files to be those.
            hub=Path(torch.hub.get_dir()).resolve()/"checkpoints"
            if any(Path(m["path"]).resolve().parent!=hub for m in registered["members"]):
                raise ValueError(f"registered checkpoints must live in the torch hub cache {hub} that get_model loads")
        self.torch=torch; self.model=get_model(model_name)
        native_sources=tuple(getattr(self.model,"sources",()))
        if len(native_sources)!=4 or set(native_sources)!=set(SOURCES): raise ValueError(f"checkpoint is not an exact four-source model: {native_sources}")
        if registered is not None and (list(native_sources)!=registered["native_source_order"] or getattr(self.model,"samplerate",None)!=registered["samplerate"]):
            raise ValueError("loaded model differs from its registered source order or samplerate")
        self.native_sources=native_sources
        self.source_order=[native_sources.index(name) for name in SOURCES]
        max_segment=getattr(self.model,"segment",None)
        if max_segment is not None and segment>max_segment: raise ValueError(f"requested segment {segment}s exceeds model limit {max_segment}s")
        self.segment,self.overlap=segment,overlap
        self.model.to("cuda").eval()
        self.device="cuda:0"
    def __call__(self, mixture: np.ndarray) -> np.ndarray:
        torch=self.torch
        if mixture.ndim!=2 or mixture.shape[0]!=2: raise ValueError("mixture must be (2,N)")
        audio=torch.from_numpy(np.ascontiguousarray(mixture,dtype=np.float32)).unsqueeze(0).to(device="cuda",dtype=torch.float32)
        from demucs.apply import apply_model
        with torch.inference_mode():
            out=apply_model(self.model,audio,shifts=0,split=True,overlap=self.overlap,
                            progress=False,device="cuda",segment=self.segment)
        result=out[0,self.source_order].detach().to("cpu").numpy().astype(np.float32)
        if result.shape[:2]!=(4,2) or result.shape[-1]!=mixture.shape[-1] or not np.isfinite(result).all(): raise ValueError(f"invalid Demucs output {result.shape}")
        if audio.device.type!="cuda" or out.device.type!="cuda": raise RuntimeError("inference tensors were not on CUDA")
        return result
