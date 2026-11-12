"""Double-precision block operators; no trainable parameters or label-based EQ."""
from __future__ import annotations

import numpy as np
from scipy import fft, signal

FS, CORE = 44100, 1323000


def core_slice(n):
    start = (n - CORE) // 2
    if start < 0:
        raise ValueError("block shorter than the retained core")
    return slice(start, start + CORE)


def pad_core(audio, n):
    if audio.shape[-1] != CORE:
        raise ValueError("expected a 30 second core")
    result = np.zeros((*audio.shape[:-1], n), dtype=np.float32)
    result[..., core_slice(n)] = audio
    return result


def apply_filter(audio, response, stats=None):
    original = audio.shape[-1]
    n = 2 * (len(response) - 1)
    if n < original:
        raise ValueError("filter grid shorter than signal block")
    offset = (n - original) // 2
    out = np.empty(audio.shape, dtype=np.float32)
    discarded = []
    for index in np.ndindex(audio.shape[:-1]):
        padded = np.zeros(n, dtype=np.float64)
        padded[offset:offset+original] = audio[index]
        spectrum = fft.rfft(padded, workers=1)
        del padded
        spectrum *= response
        result = fft.irfft(spectrum, n=n, workers=1)
        del spectrum
        out[index] = result[offset:offset+original]
        if stats is not None:
            total = float(np.dot(result,result))
            kept = result[offset:offset+original]
            retained = float(np.dot(kept,kept))
            discarded.append(max(0.,1-retained/total) if total>1e-24 else 0.)
    if stats is not None:
        stats['maximum_discarded_energy_fraction'] = max(discarded,default=0.)
    return out


def factors(h, delay, n, only=None):
    H = fft.rfft(np.asarray(h, dtype=np.float64), n=n, workers=1)
    f = fft.rfftfreq(n, 1 / FS)
    D = np.exp(-2j * np.pi * f * delay / FS)
    del f
    # Endpoints must represent real signals, including the integer delay sign.
    D[0], D[-1] = 1, (-1.) ** delay
    M = np.abs(H)
    A = H.copy()
    valid = M > 1e-12 * M.max()
    A /= D
    np.divide(A,M,out=A,where=valid)
    A[~valid] = 1
    A[0] = complex(float(A[0].real >= 0) * 2 - 1)
    A[-1] = complex(float(A[-1].real >= 0) * 2 - 1)
    reconstructed=D*M
    reconstructed*=A
    reconstructed-=H
    error = float(np.max(np.abs(reconstructed)) / np.max(np.abs(H)))
    del reconstructed
    unit = float(np.max(np.abs(np.abs(A) - 1)))
    if error > 1e-10 or unit > 1e-10:
        raise ValueError(f"factorization failed: {error}, {unit}")
    responses = {"D":lambda:D,"M":lambda:D*M,"A":lambda:D*A,"H":lambda:H}
    chosen = {k:responses[k]() for k in ([only] if only else responses)}
    return chosen, M, {
        "relative_reconstruction": error, "unit_magnitude": unit, "undefined_phase_bins": int((~valid).sum())}


def smooth_log(values, frequencies):
    low = np.searchsorted(frequencies, frequencies / 2 ** (1 / 6), side="left")
    high = np.searchsorted(frequencies, frequencies * 2 ** (1 / 6), side="right")
    high = np.maximum(high, low + 1)
    high = np.minimum(high, len(values))
    sums = np.r_[0., np.cumsum(values, dtype=np.float64)]
    return (sums[high] - sums[low]) / (high - low)


def bounded_gain(log_gain, frequencies):
    limit = np.log(10) * 6 / 20
    value = np.clip(log_gain, -limit, limit)
    window = np.zeros_like(frequencies)
    window[(frequencies >= 20) & (frequencies <= 1000)] = 1
    lo = (frequencies > 10) & (frequencies < 20)
    hi = (frequencies > 1000) & (frequencies < 1500)
    window[lo] = 0.5 - 0.5 * np.cos(np.pi * (frequencies[lo] - 10) / 10)
    window[hi] = 0.5 + 0.5 * np.cos(np.pi * (frequencies[hi] - 1000) / 500)
    return np.exp(value * window)


def known_q(magnitude, n):
    frequencies = fft.rfftfreq(n, 1 / FS)
    band = (frequencies >= 20) & (frequencies <= 16000)
    log_m = np.log(np.maximum(magnitude, magnitude.max() * 1e-12))
    log_m -= log_m[band].mean()
    return bounded_gain(smooth_log(-log_m, frequencies), frequencies)


def log_power(audio):
    if audio.shape != (2, CORE):
        raise ValueError("reference power must use the stereo 30s core")
    frequencies, _, z = signal.stft(audio, fs=FS, window="hann", nperseg=4096,
                                    noverlap=3072, boundary=None, padded=False, axis=-1)
    power = np.mean(z.real ** 2 + z.imag ** 2, axis=(0, 2), dtype=np.float64)
    band = (frequencies >= 20) & (frequencies <= 16000)
    if power[band].sum() <= 1e-20:
        raise ValueError("silent mixture has no reference spectral shape")
    power /= power[band].sum()
    return frequencies, np.log(np.maximum(power, power.max() * 1e-8))


def blind_q(audio, reference_log_power, n):
    frequencies, current = log_power(audio)
    gain = smooth_log(0.5 * (np.asarray(reference_log_power) - current), frequencies)
    long_f = fft.rfftfreq(n, 1 / FS)
    return bounded_gain(np.interp(long_f, frequencies, gain), long_f)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def sum_stems(stems):
    out = np.zeros(stems.shape[1:], dtype=np.float64)
    for s in stems:
        out += s
    return out.astype(np.float32)


def boundary_check(dry, h, delay, n, response_name):
    # Check DSP only. Do not change Demucs chunk positions for this comparison.
    small, _, _ = factors(h, delay, n, only=response_name)
    x1 = apply_filter(pad_core(dry, n), small[response_name])[..., core_slice(n)]
    del small
    large, _, _ = factors(h, delay, n * 2, only=response_name)
    x2 = apply_filter(pad_core(dry, n * 2), large[response_name])[..., core_slice(n * 2)]
    error = rms(x1.astype(np.float64) - x2) / max(rms(x2), 1e-12)
    if error > 1e-4:
        raise ValueError(f"block boundary check failed for {response_name}: {error}")
    return error
