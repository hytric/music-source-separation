"""A small complex residual network; its forward inputs never include labels."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def stft(wave, n_fft, hop):
    shape = wave.shape[:-1]
    window = torch.hann_window(n_fft, periodic=True, device=wave.device, dtype=wave.dtype)
    result = torch.stft(wave.reshape(-1, wave.shape[-1]), n_fft, hop_length=hop,
                        win_length=n_fft, window=window, center=True, pad_mode="constant",
                        normalized=False, onesided=True, return_complex=True)
    return result.reshape(*shape, *result.shape[-2:])


def istft(spectrum, n_fft, hop, length):
    shape = spectrum.shape[:-2]
    window = torch.hann_window(n_fft, periodic=True, device=spectrum.device, dtype=spectrum.real.dtype)
    wave = torch.istft(spectrum.reshape(-1, *spectrum.shape[-2:]), n_fft, hop_length=hop,
                       win_length=n_fft, window=window, center=True, normalized=False,
                       onesided=True, length=length, return_complex=False)
    return wave.reshape(*shape, length)


def project_mixture(mixture, estimates):
    return estimates + (mixture - estimates.sum(dim=1)).unsqueeze(1) / 4


def conv(inc, outc, stride=1):
    return nn.Sequential(nn.Conv2d(inc, outc, 3, stride=stride, padding=1),
                         nn.GroupNorm(8, outc), nn.SiLU())


class Dilated(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=(1, dilation),
                                            dilation=(1, dilation), groups=channels),
                                   nn.Conv2d(channels, channels, 1),
                                   nn.GroupNorm(8, channels), nn.SiLU())

    def forward(self, x):
        return x + self.block(x)


class JointRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        a, b, c = config["widths"]
        self.enc1 = nn.Sequential(conv(30, a), conv(a, a))
        self.enc2 = nn.Sequential(conv(a, b, 2), conv(b, b))
        self.enc3 = nn.Sequential(conv(b, c, 2), conv(c, c))
        self.context = nn.Sequential(*(Dilated(c, d) for d in config["dilations"]))
        self.dec2 = nn.Sequential(conv(c + b, b), conv(b, b))
        self.dec1 = nn.Sequential(conv(b + a, a), conv(a, a))
        self.head = nn.Conv2d(a, 16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mixture, estimates):
        if mixture.ndim != 3 or mixture.shape[1] != 2 or estimates.shape != (mixture.shape[0], 4, 2, mixture.shape[-1]):
            raise ValueError("expected mixture (B,2,N) and estimates (B,4,2,N)")
        n_fft, hop = self.config["n_fft"], self.config["hop"]
        scale = mixture.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(self.config["common_rms_floor"])
        projected = project_mixture(mixture, estimates)
        all_wave = torch.cat((mixture.unsqueeze(1), projected), dim=1) / scale.unsqueeze(1)
        spec = stft(all_wave, n_fft, hop)
        batch, _, _, freq, frames = spec.shape
        features = torch.cat((spec.real.flatten(1, 2), spec.imag.flatten(1, 2),
                              torch.log1p(spec.abs()).flatten(1, 2)), dim=1)
        features = F.pad(features, (0, -frames % 4, 0, -freq % 4))
        e1 = self.enc1(features)
        e2 = self.enc2(e1)
        e3 = self.context(self.enc3(e2))
        d2 = self.dec2(torch.cat((F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        raw = self.head(d1)[..., :freq, :frames].reshape(batch, 4, 2, 2, freq, frames)
        endpoints = raw.new_ones(freq)
        endpoints[0] = endpoints[-1] = 0
        delta = torch.complex(raw[:, :, :, 0], raw[:, :, :, 1] * endpoints[:, None])
        delta = delta * self.config["residual_spectral_scale"]
        delta = delta - delta.mean(dim=1, keepdim=True)
        out = istft(spec[:, 1:] + delta, n_fft, hop, mixture.shape[-1]) * scale.unsqueeze(1)
        # Enforce the waveform invariant after numerical STFT roundoff as well.
        out = project_mixture(mixture, out)
        silent = mixture.square().mean(dim=(-2, -1), keepdim=True) <= 1e-16
        return torch.where(silent.unsqueeze(1), torch.zeros_like(out), out)


@torch.no_grad()
def infer_record(model, mixture, estimates, chunk_samples, device):
    """Common-window overlap-add, including short records and a partial final chunk."""
    n = mixture.shape[-1]
    if n < 1:
        raise ValueError("empty inference record")
    result = torch.zeros(4, 2, n, dtype=torch.float32)
    norm = torch.zeros(n, dtype=torch.float32)
    weights = torch.hann_window(chunk_samples).clamp_min(1e-3)
    step = chunk_samples // 2
    for start in range(0, n, step):
        size = min(chunk_samples, n - start)
        x = F.pad(mixture[:, start:start + size], (0, chunk_samples - size))
        z = F.pad(estimates[:, :, start:start + size], (0, chunk_samples - size))
        out = model(x[None].to(device), z[None].to(device))[0].cpu()[..., :size]
        result[..., start:start + size] += out * weights[:size]
        norm[start:start + size] += weights[:size]
        if start + size == n:
            break
    return result / norm
