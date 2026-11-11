"""Stable, scale-sensitive supervised reconstruction objectives."""
from __future__ import annotations

import torch
from torch import nn

from .model import stft


class ReconstructionLoss(nn.Module):
    def __init__(self, config, sample_rate=44100):
        super().__init__()
        self.config = dict(config)
        self.sample_rate = sample_rate

    def forward(self, prediction, target, mixture):
        c = self.config
        q = mixture.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-5)
        p, y = prediction / q.unsqueeze(1), target / q.unsqueeze(1)
        stem_scale = y.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(c["stem_floor_fraction"])
        wave = ((p - y).abs() / stem_scale).mean()
        spectral = wave.new_zeros(())
        for size in c["fft_sizes"]:
            ps, ys = stft(p, size, size // 4), stft(y, size, size // 4)
            pm, ym = ps.abs(), ys.abs()
            numerator = torch.linalg.vector_norm(pm - ym, dim=(-2, -1))
            floor = c["stem_floor_fraction"] * (pm.shape[-1] * pm.shape[-2]) ** 0.5
            denominator = torch.linalg.vector_norm(ym, dim=(-2, -1)).clamp_min(floor)
            sc = numerator / denominator
            log = ((pm + c["log_floor"]).log() - (ym + c["log_floor"]).log()).abs().mean(dim=(-2, -1))
            active = y.square().mean(dim=-1) > 1e-12
            # No log ratio is used for an exactly silent target channel.
            quiet = p.square().mean(dim=-1)
            spectral = spectral + torch.where(active, sc + log, quiet).mean() / len(c["fft_sizes"])
        total = c["wave"] * wave + c["spectral"] * spectral
        terms = {"wave": wave, "spectral": spectral}
        if c["variant"] == "extended":
            diff = stft(p, 2048, 512) - stft(y, 2048, 512)
            error = (diff.real.abs() + diff.imag.abs()) / stem_scale.unsqueeze(-1)
            complex_loss = error.mean()
            frequencies = torch.arange(diff.shape[-2], device=diff.device) * self.sample_rate / 2048
            keep = (frequencies >= c["low_hz"][0]) & (frequencies <= c["low_hz"][1])
            low = error[..., keep, :].mean()
            total = total + c["complex"] * complex_loss + c["low"] * low
            terms.update(complex=complex_loss, low=low)
        return total, {**terms, "total": total}
