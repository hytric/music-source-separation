# Banelp reproducible separation scaffold

This isolated research package uses the wet reverberant targets in `plan.md`'s contract and never imports or modifies the demo runtime. Its tensor order is `(vocals, drums, bass, other)`, stereo, 44.1 kHz, 1,323,000 samples. Dataset, cache, checkpoint, and environment payloads belong under `/home/iiplab/code/music-source-separation/.agent_runtime/banelp-v1`; artifacts contain compact manifests, hashes, and reports only.

## Environment

Provision Python 3.12 in the host research environment. Then run:

```bash
R=/home/iiplab/code/music-source-separation/.agent_runtime/banelp-v1
A=/home/iiplab/code/music-source-separation/.agent_reports/campaigns/2026-11-08_banelp-separation-poster-results/2026-11-08_autopilot-code-separation-claude-rout-400060999c/artifacts
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

The holdout selection seal remains blocked while `selection.sar_policy_status` is `pending-owner-confirmation` (a confirmed policy also needs a numeric `sar_floor_db`). `seal` never takes a hand-edited choice (`config.selection.selected` is rejected): it loads `results/develop.json`, checks that its code/lock/config/manifest hashes still match, applies `selection.select_candidate` to the per-candidate rows `develop` wrote (SDR gain and bootstrap CI, per-stem SAR deltas, Wilcoxon p, Holm-adjusted p), and writes `selection-seal.json` binding the develop.json sha256, policy, dataset/RIR/model manifest hashes, code hash and lock hash. `evaluate` recomputes the same bindings through `provenance.verify_binding` (one helper shared with `seal`), refuses any change, takes candidates from the same `candidate_registry` that `develop` used, and never chooses settings from evaluation outcomes.

## Runtime commands

Host-run order: `prepare -> rirs -> build-wet -> register-model -> smoke -> develop -> seal -> evaluate`. Every command takes `--runtime-dir "$R" --artifact-dir "$A" --config experiments/banelp/config.json`; `-h` on each shows its extra flags.

```bash
C="--runtime-dir $R --artifact-dir $A --config experiments/banelp/config.json"
"$P" -m experiments.banelp.cli prepare $C --track-root "$R/data/musdb18hq/test"        # manifests/dataset.json
"$P" -m experiments.banelp.cli rirs $C --sources rir_sources.json --synthetic-fallback   # manifests/rirs.json (or --offline-synthetic)
"$P" -m experiments.banelp.cli build-wet $C                                              # cache/wet/<split>/<track>/R1..R4 + complete.json
"$P" -m experiments.banelp.cli register-model $C --model-id htdemucs --checkpoint "$TORCH_HOME/hub/checkpoints/<file>.th" \
    --introspect --train-data-status verified-excluded --evidence <doc>                  # manifests/models.json
"$P" -m experiments.banelp.cli smoke $C --device cuda --require-cuda --split dev --track-index 0 --seconds 5
"$P" -m experiments.banelp.cli develop $C                                                # results/develop.json (dev only)
"$P" -m experiments.banelp.cli seal $C                                                   # selection-seal.json
"$P" -m experiments.banelp.cli evaluate $C --seal "$A/selection-seal.json"               # results/evaluation.json, one claimed run
```

- `rirs`: `rir_sources.example.json` documents the source-list schema. Its URLs, checksums, licenses and archive members must be verified by the host-run (OpenAIR terms) before use. Each room tries its candidates in order (resumable download with checksum and disk-space check, safe archive extraction, native rate below 44.1 kHz rejected rather than upsampled); a room with no usable candidate uses the seeded synthetic RIR only when `--synthetic-fallback`/`--offline-synthetic` is given. The manifest keeps the target RT60 and the measured T30 (or the failure reason) separate, and records the native rate, extracted channel, resample method, hashes and license.
- `build-wet`: dry 30 s centre crop, full convolution with the manifest RIR (offset 0), keep the first N samples, per-stem RMS match to dry, float64 sum to a float32 mixture, no clipping. Entries are channels-first (`mixture (2,N)`, `targets (4,2,N)`) and `complete.json` is written last with the cache key and file hashes.
- `develop`, `smoke`, `seal`, `evaluate` and every cache reader (`data.read_cache_entry`) refuse a wet or separation entry without a matching `complete.json` (key, identity, file hashes); `develop`/`evaluate` regenerate a stale baseline-estimate entry from the practical mixture only. The wet cache key includes the code hash, so editing code requires `build-wet` again.
- `register-model` hashes every checkpoint file (bag members included) and records the native source order, sample rate and training-data status; `--introspect` loads the model with demucs and takes those facts from it. Checkpoints are loaded only after their registered sha256 is re-verified, and must be the files in the torch hub cache that `get_model` resolves.
- Holdout claim: `evaluate` validates the seal bindings, every wet entry, every baseline estimate and the sealed candidate (preflight) before it writes `claims/evaluation-40.claim` as `started`; the claim becomes `completed` only after `results/evaluation.json` is written. Retry rule: a `started` claim for the same seal may be retried (an interrupted run persisted no holdout metrics); if results for that seal already exist the retry only finalizes the claim. A `completed` claim, or one bound to a different seal, is refused.

Dataset acquisition, RIR download, model cache creation, CUDA smoke and the full dev/evaluation runs belong to the registered host-run/lab stages. CUDA smoke fails closed instead of switching to CPU. Downstream cache keys use code, lock, source/crop/RIR manifest, synthesis, weights and inference configuration hashes. Oracle outputs use the independent `oracle-dev` namespace and never enter selection.
