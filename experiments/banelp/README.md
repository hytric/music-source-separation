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

For each checkpoint retain model ID, URL/revision, file SHA256/bytes, explicit source order and training-data evidence. `verified-included` and `unknown` checkpoints cannot be clean-winner eligible; `verified-included` is refused by `seal`/`evaluate`, and `unknown` is accepted only under the disclosure rule below. No architecture is accepted as a four-source checkpoint by name alone.

## Candidate boundaries

`candidates.py` exposes identity, T1 uniform mixture consistency, T2 estimated-PSD ratio masking (alpha configurable), T5 waveform blending, norbert Wiener, and a bounded C2 consistency iteration. The oracle function is separately marked and requires target tensors. Practical adapter entry points never accept target, RIR, or T60 arguments. All STFT paths use periodic Hann, 4096/1024, zero boundary, padded frames, spectrum scaling and stereo channel independence. Oracle namespaces are dev-only.

Selection policy (fixed in `config.json`): `sar_policy_status: confirmed`, `sar_floor_db: -0.5` (per-stem median SAR delta floor), `minimum_sdr_gain_db: 0.2`, `baseline_train_data_policy: allow-unknown-with-disclosure`. A pending policy or a non-numeric floor still blocks `seal` and `evaluate`. The baseline separator (htdemucs) has no verified training-data exclusion, so it is registered `unknown`; the clean-winner ineligibility no longer stops `seal`, but the result carries no clean-winner claim: `seal` writes `baseline_disclosure` (`clean_winner_claim: not-eligible`, Korean and English statement that the baseline's training data is unverified and MUSDB18 test non-inclusion has no direct evidence) into `selection-seal.json`, `evaluate` refuses a seal whose disclosure no longer matches the registered model, and `results/evaluation.json` repeats it. A `verified-included` baseline is always refused. Without `baseline_train_data_policy: allow-unknown-with-disclosure` an unverified baseline is refused too. `seal` never takes a hand-edited choice (`config.selection.selected` is rejected): it loads `results/develop.json`, checks that its code/lock/config/manifest hashes still match, applies `selection.select_candidate` to the per-candidate rows `develop` wrote (SDR gain and bootstrap CI, per-stem SAR deltas, Wilcoxon p, Holm-adjusted p), and writes `selection-seal.json` (schema 3) binding the develop.json sha256, policy, disclosure, dataset/RIR/model manifest hashes, code hash and lock hash. `evaluate` recomputes the same bindings through `provenance.verify_binding` (one helper shared with `seal`), refuses any change, takes candidates from the same `candidate_registry` that `develop` used, and never chooses settings from evaluation outcomes.

## gpuRIR fixed-layout RIRs

Since 2026-11-09 the primary RIR set is simulated, not downloaded: gpuRIR image-source method on the interim report's layout (`rir_geometry.json`: room 7.5 x 6.0 x 3.0 m, one omni microphone, four sources, all heights 1.5 m, fs 44,100, target T60 0.26/0.89/1.28/1.96 s for R1-R4). The simulation constants are those of the reference `rir_gen_gpuRIR.py` (`abs_weights` 0.9x5 + 0.5, diffuse model from 25 dB, end at 60 dB, c 343 m/s, LUT on, mixed precision off, one source x one receiver per call, no normalisation); `rir.load_geometry` refuses any deviation, including a T60 that is not the target. The Sabine absorption is not calibrated against the result: the T60 given to the simulator is the target (`t60_input_policy: "target (oracle); measured reported alongside"`), and measured EDT/T20/T30 are stored next to it. A short measured R1 is recorded, not regenerated.

`bash experiments/banelp/bootstrap_gpurir.sh --runtime-dir "$R" --artifact-dir "$A"` builds gpuRIR (pinned commit `06da27dd`, AGPL-3.0) with a CUDA 12.6 conda toolchain under `$R/gpurir-toolchain`, source under `$R/src/gpuRIR`, CMake arch patched to sm_75, and writes `build-record.json`. There is no CPU fallback; a failed build stops the run. gpuRIR source and binaries stay in the runtime directory and are not redistributed; this repository holds only the script and the commit record. cuRAND makes the diffuse tail GPU/driver dependent, so the generated files are pinned by sha256 in `manifests/rirs.json`; regenerating produces a new manifest hash and a new wet cache key.

RT measurement (`data.decay_times`, recorded as `rt_method`): onset at the energy peak, 10 ms envelope, iterative noise-floor crosspoint (Lundeby-style), Schroeder integral cut at the crosspoint plus the extrapolated tail, ISO 3382-1 regressions (EDT 0 to -10 dB, T20 -5 to -25, T30 -5 to -35). A value that cannot be measured is `None` with a reason (`rir too short or silent`, `decay did not reach <x> dB`, `insufficient dynamic range above noise floor` when the noise floor is above -20/-35/-45 dB for EDT/T20/T30); nothing is guessed.

Archive of the previous (substitute RIR) condition, made by renaming in the runtime directory by the host-run: `rir-substitute-v1/`, `manifests/rirs-substitute-v1.json`, `cache/wet-substitute-v1/` (and `cache/separation-substitute-v1/` if present). The renamed manifest points at `rir/...` paths, so new code does not read it; read it with the `7b7f70a` checkout.

## Runtime commands

Host-run order: `prepare -> rirs -> build-wet -> register-model -> smoke -> develop -> seal -> evaluate`. Every command takes `--runtime-dir "$R" --artifact-dir "$A" --config experiments/banelp/config.json`; `-h` on each shows its extra flags.

```bash
C="--runtime-dir $R --artifact-dir $A --config experiments/banelp/config.json"
"$P" -m experiments.banelp.cli prepare $C --track-root "$R/data/musdb18hq/test"        # manifests/dataset.json
"$P" -m experiments.banelp.cli rirs $C --gpurir experiments/banelp/rir_geometry.json \
    --gpurir-build-record "$R/gpurir-toolchain/build-record.json"                        # manifests/rirs.json (gpuRIR)
# legacy: rirs $C --sources rir_sources.json --synthetic-fallback (or --offline-synthetic)
"$P" -m experiments.banelp.cli build-wet $C                                              # cache/wet/<split>/<track>/R1..R4 + complete.json
"$P" -m experiments.banelp.cli register-model $C --model-id htdemucs --checkpoint "$TORCH_HOME/hub/checkpoints/<file>.th" \
    --introspect --train-data-status verified-excluded --evidence <doc>                  # manifests/models.json
"$P" -m experiments.banelp.cli smoke $C --device cuda --require-cuda --split dev --track-index 0 --seconds 5
"$P" -m experiments.banelp.cli develop $C                                                # results/develop.json (dev only)
"$P" -m experiments.banelp.cli seal $C                                                   # selection-seal.json
"$P" -m experiments.banelp.cli evaluate $C --seal "$A/selection-seal.json"               # results/evaluation.json, one claimed run
```

- `rirs --gpurir`: see the gpuRIR section; it cannot be combined with `--sources`, `--synthetic-fallback` or `--offline-synthetic`.
- `rirs` (legacy): `rir_sources.example.json` documents the source-list schema. Its URLs, checksums, licenses and archive members must be verified by the host-run (OpenAIR terms) before use. Each room tries its candidates in order (resumable download with checksum and disk-space check, safe archive extraction, native rate below 44.1 kHz rejected rather than upsampled); a room with no usable candidate uses the seeded synthetic RIR only when `--synthetic-fallback`/`--offline-synthetic` is given. The manifest keeps the target RT60 and the measured EDT/T20/T30 (or the failure reasons) separate, and records the native rate, extracted channel, resample method, hashes and license.
- `build-wet`: dry 30 s centre crop, full convolution with the manifest RIR (offset 0), keep the first N samples, per-stem RMS match to dry, float64 sum to a float32 mixture, no clipping. Entries are channels-first (`mixture (2,N)`, `targets (4,2,N)`) and `complete.json` is written last with the cache key and file hashes.
- `develop`, `smoke`, `seal`, `evaluate` and every cache reader (`data.read_cache_entry`) refuse a wet or separation entry without a matching `complete.json` (key, identity, file hashes); `develop`/`evaluate` regenerate a stale baseline-estimate entry from the practical mixture only. The wet cache key includes the code hash, so editing code requires `build-wet` again.
- `register-model` hashes every checkpoint file (bag members included) and records the native source order, sample rate and training-data status; `--introspect` loads the model with demucs and takes those facts from it. Checkpoints are loaded only after their registered sha256 is re-verified, and must be the files in the torch hub cache that `get_model` resolves.
- Holdout claim: `evaluate` validates the seal bindings, every wet entry, every baseline estimate and the sealed candidate (preflight) before it writes `claims/evaluation-40.claim` as `started`; the claim becomes `completed` only after `results/evaluation.json` is written. Retry rule: a `started` claim for the same seal may be retried (an interrupted run persisted no holdout metrics); if results for that seal already exist the retry only finalizes the claim. A `completed` claim, or one bound to a different seal, is refused.

Dataset acquisition, RIR download, model cache creation, CUDA smoke and the full dev/evaluation runs belong to the registered host-run/lab stages. CUDA smoke fails closed instead of switching to CPU. Downstream cache keys use code, lock, source/crop/RIR manifest, synthesis, weights and inference configuration hashes. Oracle outputs use the independent `oracle-dev` namespace and never enter selection.
