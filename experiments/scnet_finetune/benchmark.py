from pathlib import Path
import sys,json,time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'공식SCNet검증_2026-11-13'))
from adapter import OfficialSCNet
from core import normalized_pair,reconstruction_loss,tensor_hash

torch.set_num_threads(2);torch.manual_seed(20261113)
root=Path('/home/iiplab/code/music-source-separation/.agent_runtime')
sep=OfficialSCNet(root/'parirset-official-v1','rev');model=sep.model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
y=torch.randn(1,4,2,6*44100,device='cuda')*.03
before=tensor_hash(model);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
for i in range(3):
 optimizer.zero_grad(set_to_none=True)
 for j in range(2):
  x,t=normalized_pair(y);out=model(x)[:,sep.order]
  loss,_=reconstruction_loss(out,t,model.stft_config)
  (loss/2).backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
 optimizer.step()
 print(json.dumps({'iteration':i,'loss':loss.item(),'gradient_norm':norm.item(),'elapsed':time.monotonic()-started}),flush=True)
record={'optimizer_updates':3,'seconds':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'state_changed':before!=tensor_hash(model),'parameters':sum(p.numel() for p in model.parameters()),'production_training_updates':0}
print(json.dumps(record),flush=True)
out=root/'scnet-finetune-v1';out.mkdir(exist_ok=True);(out/'benchmark.json').write_text(json.dumps(record,indent=2))
