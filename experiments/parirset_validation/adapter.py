"""Adapter for the pinned official PaRIRset SCNet (no CLI execution)."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch
import yaml


def official_modules(root):
    source = Path(root) / 'source/scnet'
    names = ('separation', 'SCNet', 'utils')
    previous = {n: sys.modules.pop(n, None) for n in names}
    sys.path.insert(0, str(source))
    try:
        import SCNet
        spec = importlib.util.spec_from_file_location('parir_official_inference', source / 'inference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.th = torch  # Fix upstream tensor_chunk's undefined alias.
        return SCNet.SCNet, module
    finally:
        sys.path.remove(str(source))
        for n in names:
            sys.modules.pop(n, None)
            if previous[n] is not None:
                sys.modules[n] = previous[n]


class OfficialSCNet:
    def __init__(self, root, name, segment=20):
        root = Path(root)
        cls, self.api = official_modules(root)
        config = yaml.safe_load((root / 'conf' / ('ismir26_clean.yaml' if name == 'clean' else 'ismir26_rev.yaml')).read_text())
        self.model = cls(**config['model']).eval()
        checkpoint = torch.load(root / 'models' / ('M1_clean' if name == 'clean' else 'M2_reverberant') / 'model.th', map_location='cpu', weights_only=True)
        state = checkpoint.get('best_state', checkpoint)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to('cuda')
        self.segment = segment
        self.order = [self.model.sources.index(s) for s in ('vocals', 'drums', 'bass', 'other')]

    def __call__(self, x):
        if x.ndim != 2 or x.shape[0] != 2 or not np.isfinite(x).all():
            raise ValueError('expected finite stereo mixture')
        mix = torch.from_numpy(np.ascontiguousarray(x)).float()
        mono = mix.mean(0)
        mean, std = mono.mean(), mono.std()
        if std < 1e-10:
            raise ValueError('near-silent mixture requires explicit handling')
        with torch.inference_mode():
            out = self.api.apply_model(self.model, ((mix - mean) / std)[None], shifts=0,
                split=True, segment=self.segment, samplerate=44100, overlap=.5,
                device=torch.device('cuda'))[0]
            out = out * std + mean
        result = out[self.order].cpu().numpy().astype(np.float32)
        if result.shape != (4, 2, x.shape[-1]) or not np.isfinite(result).all():
            raise ValueError('invalid SCNet output')
        return result
