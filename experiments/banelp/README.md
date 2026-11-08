# Banelp reproducible separation scaffold

This isolated research package uses the wet reverberant targets in `plan.md`'s contract and never imports or modifies the demo runtime. Its tensor order is `(vocals, drums, bass, other)`, stereo, 44.1 kHz, 1,323,000 samples. Dataset, cache, checkpoint, and environment payloads belong under `/home/iiplab/code/music-source-separation/.agent_runtime/banelp-v1`; artifacts contain compact manifests, hashes, and reports only.

## Environment

Provision Python 3.12 in the host research environment. Then run:

```bash
R=/home/iiplab/code/music-source-separation/.agent_runtime/banelp-v1
A=/home/iiplab/code/music-source-separation/.agent_reports/campaigns/2026-11-08_banelp-separation-poster-results/2026-11-08_framed-separation-gpu-route-rt-d43a08-8c67895ef9-2/artifacts
bash experiments/banelp/bootstrap.sh --runtime-dir "$R" --artifact-dir "$A"
P="$R/env/bin/python"
"$P" -m experiments.banelp.cli validate --runtime-dir "$R" --artifact-dir "$A" --config experiments/banelp/config.json --synthetic
"$P" -m unittest discover -s tests -p 'test_banelp_*.py' -v
"$P" -m unittest discover -s tests -p 'test_runtime.py' -v
```

Bootstrap refuses a missing Python 3.12 interpreter and installs only into the runtime venv. `requirements.lock` pins direct dependencies; pip resolves transitives at install time, and the resulting exact environment must be captured in `evidence/environment.json` and the generated `requirements.resolved.lock` before claiming a fully reproducible environment. Package-index URLs are written in the script. Cache variables are redirected beneath the runtime directory.

## Dataset and model provenance

`prepare --track-root` validates an already acquired MUSDB18-HQ tree containing 50 directories with four stereo WAV stems named exactly `vocals.wav`, `drums.wav`, `bass.wav`, and `other.wav`. It records per-file SHA256, bytes, sample rate, channels, length, and a stable NFC/hash 10/40 split under `runtime/manifests/dataset.json`. It does not treat the free sample pack as HQ and does not silently omit short stems. RIR origin, channel selection, location, license, T30 and fallback details must be recorded in the host-run RIR manifest before wet targets are made. `data.apply_rir` applies the same selected mono response to both stereo channels, full convolution, N-sample truncation, then per-stem RMS restoration; no mixture peak normalization is done.

For each checkpoint retain model ID, URL/revision, file SHA256/bytes, explicit source order and training-data evidence. `verified-included` and `unknown` checkpoints are diagnostic only and cannot be clean-winner eligible. No architecture is accepted as a four-source checkpoint by name alone.

## Candidate boundaries

`candidates.py` exposes identity, T1 uniform mixture consistency, T2 estimated-PSD ratio masking (alpha configurable), T5 waveform blending, norbert Wiener, and a bounded C2 consistency iteration. The oracle function is separately marked and requires target tensors. Practical adapter entry points never accept target, RIR, or T60 arguments. All STFT paths use periodic Hann, 4096/1024, zero boundary, padded frames, spectrum scaling and stereo channel independence. Oracle namespaces are dev-only.

The holdout selection seal remains blocked while `selection.sar_policy_status` is `pending-owner-confirmation`. A seal binds the policy, data manifest, checkpoint, code, and lock hashes. Evaluation verifies the seal and never chooses settings from evaluation outcomes.

## Runtime commands

```bash
"$P" -m experiments.banelp.cli prepare --runtime-dir "$R" --artifact-dir "$A" --config experiments/banelp/config.json --track-root "$R/data/musdb18hq/test"
"$P" -m experiments.banelp.cli smoke --runtime-dir "$R" --artifact-dir "$A" --config experiments/banelp/config.json --device cuda --require-cuda --split dev --track-index 0 --seconds 5
```

Dataset acquisition, RIR download/synthesis, model cache creation, CUDA smoke and any full dev/evaluation runs require the registered host-run/lab stages. CUDA smoke intentionally fails closed instead of switching to CPU. Incomplete cache entries must never be reused; downstream cache keys use code, lock, source/crop/RIR manifest, synthesis, weights and inference configuration hashes. Oracle outputs must use an independent `oracle-dev` namespace.
