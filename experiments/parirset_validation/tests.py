import unittest
from pathlib import Path
import torch
from adapter import official_modules

class AdapterTests(unittest.TestCase):
    def test_official_overlap_add_preserves_boundary_and_tail(self):
        _,api=official_modules(Path('/home/iiplab/code/music-source-separation/.agent_runtime/parirset-official-v1'))
        class Replica(torch.nn.Module):
            sources=['drums','bass','other','vocals']
            def forward(self,x):return x[:,None].repeat(1,4,1,1)
        x=torch.randn(1,2,13777);x[...,0]=3;x[...,-1]=-7
        out=api.apply_model(Replica(),x,shifts=0,segment=.1,overlap=.5,device=torch.device('cpu'))
        self.assertEqual(tuple(out.shape),(1,4,2,13777))
        torch.testing.assert_close(out,x[:,None].expand_as(out),atol=1e-6,rtol=1e-6)
    def test_official_padding_center_trim(self):
        _,api=official_modules(Path('/home/iiplab/code/music-source-separation/.agent_runtime/parirset-official-v1'))
        x=torch.arange(11.)[None,None]
        padded=api.TensorChunk(x,0,4).padded(8)
        torch.testing.assert_close(padded,torch.tensor([[[0.,0.,0.,1.,2.,3.,4.,5.]]]))
        self.assertEqual(api.center_trim(torch.arange(10.),7).tolist(),list(range(1,8)))

if __name__=='__main__':unittest.main()
