from pathlib import Path
import json,time
import numpy as np
import torch
from adapter import OfficialSCNet
root=Path('/home/iiplab/code/music-source-separation/.agent_runtime/parirset-official-v1')
torch.set_num_threads(2)
torch.manual_seed(20261113)
model=OfficialSCNet(root,'clean')
records=[]
for seconds in (5,20):
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    x=np.random.default_rng(42).normal(0,.01,(2,44100*seconds)).astype('float32')
    y=model(x)
    r={'seconds':seconds,'shape':list(y.shape),'elapsed':time.monotonic()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
    records.append(r);print(json.dumps(r),flush=True)
output=Path('/home/iiplab/code/music-source-separation/.agent_runtime/parirset-compare-v1/smoke.json')
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'torch':torch.__version__,'records':records},indent=2))
