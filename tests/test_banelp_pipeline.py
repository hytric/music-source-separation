"""CPU tests for the wet cache, RIR acquisition and the develop -> seal -> evaluate guard path."""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
import numpy as np
import soundfile as sf
from experiments.banelp import cli, provenance, rir
from experiments.banelp.contracts import SOURCES, canonical_json, sha256_file, validate_tensors
from experiments.banelp.data import (ROOMS, apply_rir, load_wet, make_sample, read_cache_entry, save_wet_sample,
                                     wet_cache_key)
from experiments.banelp.selection import claim_complete, claim_start, select_candidate

ROOT = Path(__file__).parents[1]
SAMPLES = 8192
STEMS = list(SOURCES)


def _run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main([str(a) for a in argv])
    assert code == 0
    return json.loads(out.getvalue())


class WetSynthesisTests(unittest.TestCase):
    def test_writer_and_readers_share_channels_first_layout(self):
        rng = np.random.default_rng(1)
        dry = {n: rng.normal(0, .1, (SAMPLES + 400, 2)).astype(np.float32) for n in SOURCES}
        rirs = {n: rir.synthetic_rir("R1", p)[0] for n, p in rir.STEM_POSITION.items()}
        mixture, targets, rows = make_sample(dry, rirs, samples=SAMPLES)
        validate_tensors(mixture, targets, samples=SAMPLES)
        self.assertEqual((mixture.shape, targets.shape), ((2, SAMPLES), (4, 2, SAMPLES)))
        self.assertTrue(np.allclose(targets.astype(np.float64).sum(axis=0), mixture, atol=1e-6))
        self.assertEqual([r["source"] for r in rows], list(SOURCES))

    def test_rms_match_tail_removal_and_silence(self):
        rng = np.random.default_rng(2)
        dry = rng.normal(0, .1, (2048, 2)).astype(np.float32)
        wet, info = apply_rir(dry, np.r_[1., np.zeros(99), .5 * np.ones(400)])
        self.assertAlmostEqual(info["wet_rms_after_gain"], info["dry_rms"], places=5)
        self.assertEqual((info["dry_length"], info["full_length"], info["kept_length"]), (2048, 2048 + 499 - 0, 2048))
        self.assertGreater(info["tail_energy"], 0); self.assertGreater(info["tail_energy_fraction"], 0)
        silent, sinfo = apply_rir(np.zeros((2048, 2), np.float32), np.r_[1., .3])
        self.assertTrue(sinfo["silent"]); self.assertEqual(float(np.abs(silent).max()), 0.0); self.assertEqual(silent.shape, (2, 2048))
        with self.assertRaises(ValueError): apply_rir(dry, np.array([np.nan, 1.]))

    def test_synthetic_rir_is_deterministic_and_target_kept_apart(self):
        a, ga = rir.synthetic_rir("R3", "P2"); b, _ = rir.synthetic_rir("R3", "P2"); c, _ = rir.synthetic_rir("R3", "P3")
        self.assertTrue(np.array_equal(a, b)); self.assertFalse(np.array_equal(a, c))
        self.assertEqual(ga["target_rt60_seconds"], 1.28); self.assertEqual(a[0], 1.0)

    def test_complete_json_is_required_and_hash_checked(self):
        rng = np.random.default_rng(3)
        dry = {n: rng.normal(0, .1, (SAMPLES, 2)).astype(np.float32) for n in SOURCES}
        rirs = {n: rir.synthetic_rir("R1", p)[0] for n, p in rir.STEM_POSITION.items()}
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp).resolve()
            save_wet_sample(runtime, split="dev", track="t", room="R1", dry_stems=dry, rir_by_source=rirs, source_records={}, rir_records={},
                            cache_key="k", samples=SAMPLES)
            x, targets, record = load_wet(runtime, split="dev", track="t", room="R1", cache_key="k", samples=SAMPLES)
            self.assertEqual((x.shape, targets.shape), ((2, SAMPLES), (4, 2, SAMPLES)))
            with self.assertRaises(ValueError): load_wet(runtime, split="dev", track="t", room="R1", cache_key="other", samples=SAMPLES)
            root = runtime / "cache/wet/dev/t/R1"
            np.save(root / "mixture.npy", x + 1)
            with self.assertRaises(ValueError): load_wet(runtime, split="dev", track="t", room="R1", cache_key="k", samples=SAMPLES)
            (root / "complete.json").unlink()
            with self.assertRaises(FileNotFoundError): load_wet(runtime, split="dev", track="t", room="R1", cache_key="k", samples=SAMPLES)
            with self.assertRaises(FileNotFoundError): read_cache_entry(root, ("mixture",), cache_key="k", identity={})


class RirAcquisitionTests(unittest.TestCase):
    def test_download_resume_checksum_and_safe_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve(); src = tmp / "src.bin"; src.write_bytes(b"abcdef" * 1000)
            good = sha256_file(src)
            got = rir.download_resumable(src.as_uri(), tmp / "d/out.bin", sha256=good, min_free_bytes=0)
            self.assertEqual((got["checksum"], got["sha256"]), ("provider-verified", good))
            self.assertTrue(rir.download_resumable(src.as_uri(), tmp / "d/out.bin", sha256=good, min_free_bytes=0)["reused"])
            with self.assertRaises(ValueError): rir.download_resumable(src.as_uri(), tmp / "e/out.bin", sha256="0" * 64, min_free_bytes=0)
            self.assertFalse((tmp / "e/out.bin").exists())
            with self.assertRaises(ValueError): rir.download_resumable("ftp://example.invalid/x", tmp / "f", min_free_bytes=0)
            archive = tmp / "a.zip"
            with zipfile.ZipFile(archive, "w") as zf: zf.writestr("ok/a.wav", b"1"); zf.writestr("../evil", b"2")
            self.assertEqual(set(rir.safe_extract(archive, ["ok/a.wav"], tmp / "x")), {"ok/a.wav"})
            with self.assertRaises(ValueError): rir.safe_extract(archive, ["../evil"], tmp / "x")

    def test_offline_synthetic_manifest_records_target_and_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp).resolve()
            manifest = rir.acquire_rirs(runtime, sources=None, synthetic_fallback=True)
            self.assertEqual(set(manifest["rooms"]), set(ROOMS))
            for room, record in manifest["rooms"].items():
                self.assertEqual(record["kind"], "synthetic"); self.assertEqual(record["target_rt60_seconds"], rir.TARGET_RT60[room])
                for entry in record["positions"].values(): self.assertIn("measured_t30_seconds", entry)
            first = rir.load_rirs(runtime, manifest)["R2"]["P1"]
            self.assertEqual(first[0], 1.0); self.assertLess(float(np.abs(first[1:]).max()), 1.0)
            with self.assertRaises(RuntimeError): rir.acquire_rirs(runtime, sources=None, synthetic_fallback=False)

    def test_failed_candidate_is_recorded_then_synthetic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp).resolve()
            sources = {"rooms": [{"room": "R1", "candidates": [{"kind": "openair", "space": "x", "license": "l"}]}]}
            manifest = rir.acquire_rirs(runtime, sources=sources, synthetic_fallback=True)
            self.assertEqual(manifest["rooms"]["R1"]["kind"], "synthetic")
            self.assertIn("no verified URL", manifest["rooms"]["R1"]["attempts"][0]["failure"])


class StubSeparator:
    """CPU stand-in for HTDemucsCUDA: sees only the mixture, like the real adapter."""
    lookup: dict = {}
    calls = 0

    def __init__(self, *, model_name, segment, overlap, registered): self.registered = registered

    def __call__(self, mixture):
        StubSeparator.calls += 1
        mixture = np.ascontiguousarray(mixture, dtype=np.float32)
        targets = self.lookup.get(mixture.tobytes())
        if targets is None: return np.stack([mixture / 4] * 4)  # donor re-separation of a stem estimate
        rng = np.random.default_rng(int(np.abs(mixture).sum() * 1e6) % (2 ** 32))
        return (.7 * targets + rng.normal(0, .02, targets.shape)).astype(np.float32)


class PipelineTests(unittest.TestCase):
    TRAIN_DATA_STATUS = "verified-excluded"
    EVIDENCE = ["test-evidence"]
    SELECTION_EXTRA = {}
    WEIGHTS = b"stub-weights"

    @classmethod
    def selection_extra(cls):
        return dict(cls.SELECTION_EXTRA)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.tmp.cleanup)
        base = Path(cls.tmp.name).resolve(); cls.runtime, cls.artifact = base / "runtime", base / "artifacts"
        config = json.loads((ROOT / "experiments/banelp/config.json").read_text())
        config["samples"] = SAMPLES; config["selection"] = {"sar_policy_status": "confirmed", "sar_floor_db": -100.0, "minimum_sdr_gain_db": .1, **cls.selection_extra()}
        cls.config = base / "config.json"; cls.config.write_text(json.dumps(config))
        cls.common = ["--runtime-dir", cls.runtime, "--artifact-dir", cls.artifact, "--config", cls.config]
        rng = np.random.default_rng(5); tracks = cls.runtime / "data/test"
        for i in range(50):
            for stem in SOURCES:
                path = tracks / f"track-{i:02d}" / f"{stem}.wav"; path.parent.mkdir(parents=True, exist_ok=True)
                audio = np.zeros((SAMPLES + 500, 2), np.float32) if (i, stem) == (0, "other") else rng.normal(0, .1, (SAMPLES + 500, 2))
                sf.write(path, audio, 44100, subtype="FLOAT")
        _run("prepare", *cls.common, "--track-root", tracks)
        _run("rirs", *cls.common, "--offline-synthetic")
        summary = _run("build-wet", *cls.common)
        assert (summary["written"], summary["tracks"]) == (200, 50), summary
        ckpt = cls.runtime / "weights.th"; ckpt.write_bytes(cls.WEIGHTS)
        _run("register-model", *cls.common, "--model-id", "htdemucs", "--checkpoint", ckpt, "--native-source-order", "drums,bass,other,vocals",
             "--samplerate", "44100", "--train-data-status", cls.TRAIN_DATA_STATUS, *[a for e in cls.EVIDENCE for a in ("--evidence", e)])
        cls.models_bytes = (cls.runtime / "manifests/models.json").read_bytes()
        binding = provenance.current_binding(cls.runtime, cls.config); wet_key = wet_cache_key(
            code_hash=binding["code_hash"], lock_hash=binding["lock_hash"], manifest_hash=binding["manifest_hash"],
            rir_manifest_hash=binding["rir_manifest_hash"], samples=SAMPLES)
        manifest = json.loads((cls.runtime / "manifests/dataset.json").read_text())
        StubSeparator.lookup = {}
        for entry in manifest["tracks"]:
            for room in ROOMS:
                x, targets, _ = load_wet(cls.runtime, split=entry["split"], track=entry["name"], room=room, cache_key=wet_key, samples=SAMPLES)
                StubSeparator.lookup[x.tobytes()] = targets
        patcher = mock.patch.object(cli, "SEPARATOR_FACTORY", StubSeparator); patcher.start(); cls.addClassCleanup(patcher.stop)
        cls.lock = _run("lock-policy", *cls.common)
        cls.lock_bytes = (cls.runtime / "manifests/selection-policy.json").read_bytes()
        cls.develop = _run("develop", *cls.common)

    def setUp(self):
        (self.runtime / "manifests/models.json").write_bytes(self.models_bytes)
        (self.runtime / "manifests/selection-policy.json").write_bytes(self.lock_bytes)
        shutil.rmtree(self.runtime / "claims", ignore_errors=True)
        (self.runtime / "results/evaluation.json").unlink(missing_ok=True)
        (self.artifact / "selection-seal.json").unlink(missing_ok=True)

    def test_wet_cache_matches_reader_contract_and_is_reused(self):
        summary = _run("build-wet", *self.common)
        self.assertEqual((summary["written"], summary["reused"]), (0, 200))
        manifest = json.loads((self.runtime / "manifests/models.json").read_text())
        self.assertEqual(manifest["models"][0]["bag_size"], 1)
        self.assertEqual(manifest["models"][0]["clean_winner_eligible"], self.TRAIN_DATA_STATUS == "verified-excluded")

    def test_develop_seal_evaluate_round_trip_with_claim_rules(self):
        develop = self.develop
        results = json.loads((self.runtime / "results/develop.json").read_text())
        self.assertEqual(develop["sha256"], sha256_file(self.runtime / "results/develop.json"))
        ok = {r["name"]: r for r in results["candidates"] if r["status"] == "ok"}
        self.assertIn("T1-uniform", ok)
        for row in ok.values():
            self.assertEqual(set(row["stem_median_delta_sar"]), set(SOURCES)); self.assertIn("holm_adjusted_p", row)
        seal = _run("seal", *self.common)
        sealed = json.loads((self.artifact / "selection-seal.json").read_text())
        self.assertEqual(sealed["develop_sha256"], develop["sha256"]); self.assertEqual(sealed["code_hash"], provenance.code_hash())
        self.assertNotEqual(seal["selected"], "identity")
        seal_path = self.artifact / "selection-seal.json"; claim = self.runtime / "claims/evaluation-40.claim"
        # A code change after sealing is refused before the holdout claim is taken.
        with mock.patch.object(provenance, "code_hash", return_value="0" * 64):
            with self.assertRaises(ValueError): _run("evaluate", *self.common, "--seal", seal_path)
        self.assertFalse(claim.exists())
        # An inference/metric crash leaves a `started` claim and no results.
        real = cli.project_metrics; state = {"n": 0}
        def flaky(*args, **kwargs):
            state["n"] += 1
            if state["n"] == 30: raise MemoryError("simulated crash")
            return real(*args, **kwargs)
        with mock.patch.object(cli, "project_metrics", flaky):
            with self.assertRaises(MemoryError): _run("evaluate", *self.common, "--seal", seal_path)
        self.assertEqual(json.loads(claim.read_text())["status"], "started"); self.assertFalse((self.runtime / "results/evaluation.json").exists())
        result = _run("evaluate", *self.common, "--seal", seal_path)
        final = json.loads(claim.read_text())
        self.assertEqual((final["status"], final["attempt"]), ("completed", 2)); self.assertEqual(len(result["records"]), 160)
        with self.assertRaises(RuntimeError): _run("evaluate", *self.common, "--seal", seal_path)

    def test_seal_requires_develop_results_and_ignores_hand_edited_choice(self):
        develop = self.runtime / "results/develop.json"; held = develop.read_bytes(); develop.unlink()
        try:
            with self.assertRaises(FileNotFoundError): _run("seal", *self.common)
        finally: develop.write_bytes(held)
        config = json.loads(self.config.read_text()); config["selection"]["selected"] = "C2-iter2"
        edited = self.config.with_name("edited.json"); edited.write_text(json.dumps(config))
        with self.assertRaises(ValueError): _run("develop", "--runtime-dir", self.runtime, "--artifact-dir", self.artifact, "--config", edited)

    # ---- P2-1: develop.json detail ---------------------------------------------------------------------------------
    def test_develop_json_keeps_per_observation_scores_and_candidate_detail(self):
        results = json.loads((self.runtime / "results/develop.json").read_text())
        self.assertEqual((results["schema_version"], results["metrics_config"]), (3, json.loads(self.config.read_text())["metrics"]))
        base = results["baseline"]["observations"]; self.assertEqual(len(base), 160)
        self.assertEqual({(o["track"], o["room"], o["stem"]) for o in base}, {(t, r, s) for t in results["tracks"] for r in ROOMS for s in SOURCES})
        for o in base: self.assertTrue({"si_sdr", "si_sir", "si_sar", "target_gain", "state", "metric_states"} <= set(o))
        if "track-00" in results["tracks"]:  # the fixture's silent `other` stem keeps its reason instead of vanishing
            silent = [o for o in base if o["track"] == "track-00" and o["stem"] == "other"]
            self.assertEqual({o["state"] for o in silent}, {"silent-target"}); self.assertTrue(all(o["si_sdr"] is None and o["metric_states"]["si_sir"] == "silent-target" for o in silent))
        self.assertTrue(any(o["si_sdr"] is not None and o["si_sir"] is not None and o["si_sar"] is not None for o in base))
        self.assertEqual(set(results["baseline"]["summary"]), {"si_sdr", "si_sir", "si_sar"}); self.assertEqual(set(results["baseline"]["summary"]["si_sir"]["stem_median"]), set(SOURCES))
        self.assertEqual(set(results["candidate_observations"]), {r["name"] for r in results["candidates"] if r["status"] == "ok"})
        self.assertEqual(len(results["candidate_observations"]["T1-uniform"]), 160)
        oracle = results["oracle_diagnostics"]; self.assertEqual(len(oracle), 160); self.assertTrue(all({"si_sdr", "si_sir", "si_sar", "target_gain", "metric_states"} <= set(o) for o in oracle))
        row = [r for r in results["candidates"] if r["name"] == "T1-uniform"][0]
        self.assertEqual(set(row["delta_detail"]["stem_median"]), {"si_sdr", "si_sir", "si_sar"}); self.assertEqual(set(row["delta_detail"]["improvement_ratio"]["si_sdr"]), {"improved", "defined", "ratio"})
        self.assertLessEqual(row["delta_detail"]["improvement_ratio"]["si_sdr"]["defined"], 160); self.assertIn("undefined_by_reason", row["delta_detail"])
        self.assertEqual(row["sample_median_delta_db"], row["summary"]["sample_median_delta_db"]); self.assertIn("track_median_delta_db", row["summary"])

    # ---- P2-4: practical outputs do not depend on the targets --------------------------------------------------------
    def test_practical_outputs_are_identical_when_only_the_targets_differ(self):
        def digests(perturb):
            seen = []; real_run, real_load = cli.run_candidate, cli.load_wet
            def spy(spec, x, estimates, by_model=None):
                out = real_run(spec, x, estimates, by_model); seen.append(hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()); return out
            def load(*a, **k):
                x, targets, record = real_load(*a, **k); return x, perturb(targets), record
            with mock.patch.object(cli, "run_candidate", spy), mock.patch.object(cli, "load_wet", load), mock.patch.object(cli, "atomic_json"):
                _run("develop", *self.common)
            return seen
        rng = np.random.default_rng(99)
        same = digests(lambda t: t); different = digests(lambda t: (.3 * t + rng.normal(0, .05, t.shape)).astype(np.float32))
        self.assertGreater(len(same), 100); self.assertEqual(same, different)

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _config_with(self, name, **selection):
        config = json.loads(self.config.read_text()); config["selection"] = {**config["selection"], **selection}
        path = self.config.with_name(name); path.write_text(json.dumps(config)); return ["--runtime-dir", self.runtime, "--artifact-dir", self.artifact, "--config", path]

    def _register(self, model_id, payload, status, *evidence, extra=()):
        ckpt = self.runtime / f"{model_id}.th"; ckpt.write_bytes(payload)
        _run("register-model", *self.common, "--model-id", model_id, "--checkpoint", ckpt, "--native-source-order", "drums,bass,other,vocals", "--samplerate", "44100",
             "--train-data-status", status, *[a for e in evidence for a in ("--evidence", e)], *extra)

    # ---- P1-2: pre-registered lock ---------------------------------------------------------------------------------
    def test_lock_content_and_develop_record(self):
        results = json.loads((self.runtime / "results/develop.json").read_text()); lock = json.loads((self.runtime / "manifests/selection-policy.json").read_text())
        self.assertEqual(results["policy_lock_sha256"], lock["sha256"]); self.assertEqual(self.lock["sha256"], lock["sha256"])
        content = lock["content"]
        self.assertEqual(content["adoption"]["sar_floor_db"], -100.0); self.assertEqual(content["statistics"]["seed"], 20261108); self.assertEqual(content["statistics"]["bootstrap"], 10000)
        self.assertIn("median over dev track x room samples", content["statistics"]["effect_estimand"]); self.assertIn("track-cluster bootstrap", content["statistics"]["ci"])
        self.assertIn("per-stem median over dev track x room samples", content["statistics"]["sar_estimand"])
        self.assertEqual(content["baseline"]["model_id"], "htdemucs"); self.assertIn("T1-uniform", [c["name"] for c in content["candidates"]])
        self.assertTrue(all(set(c) >= {"name", "params", "config_hash", "models"} for c in content["candidates"]))
        self.assertTrue((self.artifact / "evidence/manifests/selection-policy.json").is_file())
        self.assertEqual(_run("lock-policy", *self.common)["sha256"], lock["sha256"])  # identical content is idempotent

    def test_develop_without_lock_is_refused(self):
        lock = self.runtime / "manifests/selection-policy.json"; lock.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "lock-policy"): _run("develop", *self.common)

    def test_develop_refuses_unconfirmed_policy_and_edited_lock(self):
        with self.assertRaisesRegex(ValueError, "confirms SAR policy"): _run("develop", *self._config_with("pending.json", sar_policy_status="pending-owner-confirmation"))
        lock = self.runtime / "manifests/selection-policy.json"; edited = json.loads(lock.read_text()); edited["content"]["adoption"]["sar_floor_db"] = -3.0; lock.write_text(json.dumps(edited))
        with self.assertRaisesRegex(ValueError, "hash mismatch"): _run("develop", *self.common)

    def test_lock_overwrite_with_different_content_is_refused(self):
        before = (self.runtime / "manifests/selection-policy.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"): _run("lock-policy", *self._config_with("other-floor.json", minimum_sdr_gain_db=.05))
        self.assertEqual((self.runtime / "manifests/selection-policy.json").read_bytes(), before)

    def test_threshold_changes_after_develop_refuse_seal_and_develop(self):
        # project-verify counterexample: a +0.1 dB / SAR -2 dB candidate must not seal under a post-hoc minimum=.05, sar_floor=-3.
        for name, edits in (("posthoc.json", {"minimum_sdr_gain_db": .05, "sar_floor_db": -3.0}), ("floor.json", {"sar_floor_db": -3.0}), ("minimum.json", {"minimum_sdr_gain_db": .05})):
            common = self._config_with(name, **edits)
            with self.assertRaisesRegex(ValueError, "pre-registered lock"): _run("seal", *common)
            with self.assertRaisesRegex(ValueError, "pre-registered lock"): _run("develop", *common)
            self.assertFalse((self.artifact / "selection-seal.json").exists())

    def test_candidate_set_change_refuses_develop_and_seal(self):
        self._register("other-model", b"other-weights", "verified-excluded", "evidence-doc")
        common = self._config_with("candidates.json", model_candidates=["replace:other-model"])
        with self.assertRaisesRegex(ValueError, "pre-registered lock"): _run("develop", *common)
        with self.assertRaisesRegex(ValueError, "pre-registered lock"): _run("seal", *common)

    def test_seal_requires_develop_made_under_the_same_lock(self):
        develop = self.runtime / "results/develop.json"; held = develop.read_bytes(); data = json.loads(held); data["policy_lock_sha256"] = "0" * 64
        develop.write_text(json.dumps(data))
        try:
            with self.assertRaisesRegex(ValueError, "different policy lock"): _run("seal", *self.common)
        finally: develop.write_bytes(held)

    def test_seal_embeds_lock_and_states_the_estimand(self):
        _run("seal", *self.common); sealed = json.loads((self.artifact / "selection-seal.json").read_text()); lock = json.loads((self.runtime / "manifests/selection-policy.json").read_text())
        self.assertEqual((sealed["policy_lock_sha256"], sealed["policy_lock"]), (lock["sha256"], lock["content"]))
        self.assertIn("median over dev track x room samples of the 4-stem mean dSI-SDR", sealed["statistic"]["effect_estimand"])
        self.assertIn("seed=20261108", sealed["statistic"]["ci"])
        if sealed["selection"]["selected"] != "identity": self.assertIn("gates", sealed["selection"]["candidate"])

    def test_evaluate_refuses_a_seal_with_a_different_lock(self):
        _run("seal", *self.common); seal_path = self.artifact / "selection-seal.json"; seal = json.loads(seal_path.read_text())
        seal["policy_lock_sha256"] = "1" * 64; seal["sha256"] = hashlib.sha256(canonical_json({k: v for k, v in seal.items() if k != "sha256"})).hexdigest()
        forged = self.artifact / "forged-lock-seal.json"; forged.write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError, "policy lock differs"): _run("evaluate", *self.common, "--seal", forged)
        self.assertFalse((self.runtime / "claims/evaluation-40.claim").exists())

    # ---- P1-3: eligibility per actual dependency -----------------------------------------------------------------
    def test_unrelated_model_changes_neither_seal_nor_evaluate_acceptance(self):
        first = _run("seal", *self.common); seal_path = self.artifact / "selection-seal.json"
        self._register("unrelated-clean", b"unrelated", "verified-excluded", "evidence-doc")
        again = _run("seal", *self.common)
        self.assertEqual((first["sha256"], first["selected"]), (again["sha256"], again["selected"]))
        with mock.patch.object(cli, "project_metrics", side_effect=RuntimeError("reached-holdout-computation")):  # every gate and the preflight passed
            with self.assertRaisesRegex(RuntimeError, "reached-holdout-computation"): _run("evaluate", *self.common, "--seal", seal_path)
        (self.runtime / "manifests/models.json").write_bytes(self.models_bytes)
        shutil.rmtree(self.runtime / "claims", ignore_errors=True)
        with mock.patch.object(cli, "project_metrics", side_effect=RuntimeError("reached-holdout-computation")):
            with self.assertRaisesRegex(RuntimeError, "reached-holdout-computation"): _run("evaluate", *self.common, "--seal", seal_path)

    def test_seal_model_hashes_list_the_depended_on_models_with_status(self):
        _run("seal", *self.common); sealed = json.loads((self.artifact / "selection-seal.json").read_text())
        ids = [m["model_id"] for m in sealed["model_hashes"]]; self.assertEqual(ids, ["htdemucs"])
        entry = sealed["model_hashes"][0]; self.assertEqual(entry["train_data_status"], self.TRAIN_DATA_STATUS); self.assertIn("route", entry)

    def test_verified_included_baseline_is_always_refused(self):
        self._register("htdemucs", self.WEIGHTS, "verified-included", "contaminated-doc")
        with self.assertRaisesRegex(ValueError, "verified-included"): _run("seal", *self.common)


class UnknownBaselinePipelineTests(PipelineTests):
    """The same round trip with a baseline whose training data is unverified: allowed only with the disclosure and a bound user decision."""
    TRAIN_DATA_STATUS = "unknown"
    EVIDENCE = []
    SELECTION_EXTRA = {"baseline_train_data_policy": "allow-unknown-with-disclosure"}

    @classmethod
    def decision(cls, sha=None):
        return {"model_id": "htdemucs", "sha256": sha or hashlib.sha256(cls.WEIGHTS).hexdigest(), "decision": "use-with-disclosure (training data unverified)", "decided_by": "user", "date": "2026-11-09"}

    @classmethod
    def selection_extra(cls):
        return {**cls.SELECTION_EXTRA, "baseline_decisions": [cls.decision()]}

    def test_seal_and_evaluation_carry_the_disclosure(self):
        _run("seal", *self.common); seal_path = self.artifact / "selection-seal.json"; sealed = json.loads(seal_path.read_text())
        disclosure = sealed["baseline_disclosure"]
        self.assertEqual((disclosure["clean_winner_claim"], disclosure["clean_winner_eligible"], disclosure["baseline_train_data_status"]), ("not-eligible", False, "unknown"))
        self.assertIn("clean winner가 아니다", disclosure["disclosure_ko"]); self.assertIn("not a clean winner", disclosure["disclosure_en"])
        self.assertIn("800", disclosure["disclosure_ko"]); self.assertIn("800 extra songs were not audited per song", disclosure["disclosure_en"])
        self.assertEqual(disclosure["decision"], self.decision())
        self.assertEqual(sealed["schema_version"], 4); self.assertEqual([(m["model_id"], m["route"]) for m in sealed["model_hashes"]], [("htdemucs", "user-decision")])
        self.assertEqual(json.loads((self.runtime / "results/develop.json").read_text())["baseline_disclosure"], disclosure)
        result = _run("evaluate", *self.common, "--seal", seal_path)
        self.assertEqual(result["baseline_disclosure"], disclosure)
        self.assertEqual(json.loads((self.runtime / "results/evaluation.json").read_text())["baseline_disclosure"], disclosure)

    def test_seal_refuses_unknown_baseline_without_policy_flag(self):
        with self.assertRaisesRegex(ValueError, "baseline_train_data_policy"):
            _run("seal", *self._config_with("plain.json", baseline_train_data_policy="refuse"))

    def test_unknown_baseline_without_matching_decision_is_refused_even_with_an_unrelated_clean_model(self):
        self._register("unrelated-clean", b"unrelated", "verified-excluded", "evidence-doc")
        for name, decisions in (("none.json", []), ("other-sha.json", [self.decision("f" * 64)]), ("other-id.json", [{**self.decision(), "model_id": "unrelated-clean"}]),
                                ("not-user.json", [{**self.decision(), "decided_by": "agent"}]), ("signature.json", [{**self.decision(), "demucs_signature": "deadbeef"}])):
            with self.assertRaisesRegex(ValueError, "decision"): _run("seal", *self._config_with(name, baseline_decisions=decisions))
        self.assertFalse((self.artifact / "selection-seal.json").exists())

    def test_tampered_disclosure_is_refused_before_the_claim(self):
        _run("seal", *self.common); seal_path = self.artifact / "selection-seal.json"; seal = json.loads(seal_path.read_text())
        seal["baseline_disclosure"] = {**seal["baseline_disclosure"], "clean_winner_claim": "eligible", "disclosure_ko": None, "disclosure_en": None}
        seal["sha256"] = hashlib.sha256(canonical_json({k: v for k, v in seal.items() if k != "sha256"})).hexdigest()
        forged = self.artifact / "forged-seal.json"; forged.write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError, "baseline disclosure changed"): _run("evaluate", *self.common, "--seal", forged)
        self.assertFalse((self.runtime / "claims/evaluation-40.claim").exists())


class SelectionAndRowTests(unittest.TestCase):
    def test_ragged_rows_are_padded_not_dropped_silently(self):
        shape = (3, 16); spec = {"params": {}, "additional_inference_calls": 0}
        base_sdr = np.full(shape, 1.0); base_sar = np.full(shape, 2.0); base_sdr[0, :4] = np.nan; base_sar[0, :4] = np.nan  # silent stems
        sdr = base_sdr + .5; sar = base_sar + .1
        row = cli._candidate_row("c", spec, sdr, sar, base_sdr, base_sar)
        self.assertEqual((row["status"], row["observations"]), ("ok", 44))
        self.assertAlmostEqual(row["summary"]["track_median_delta_db"], .5)
        bad = sdr.copy(); bad[1, 6] = np.nan
        self.assertEqual(cli._candidate_row("c", spec, bad, sar, base_sdr, base_sar)["status"], "invalid-observations")
        empty = np.full(shape, np.nan)
        self.assertEqual(cli._candidate_row("c", spec, empty, empty, empty, empty)["status"], "track-without-observations")

    def test_baseline_disclosure_rules(self):
        base = {"model_id": "htdemucs", "sha256": "a" * 64, "members": []}; allow = {"baseline_train_data_policy": "allow-unknown-with-disclosure"}
        decision = {"model_id": "htdemucs", "sha256": "a" * 64, "decision": "use-with-disclosure", "decided_by": "user", "date": "2026-11-09"}
        with self.assertRaises(ValueError): cli._baseline_disclosure({**base, "train_data_status": "verified-included", "clean_winner_eligible": False}, allow)
        with self.assertRaises(ValueError): cli._baseline_disclosure({**base, "train_data_status": "unknown", "clean_winner_eligible": False}, {"baseline_decisions": [decision]})
        with self.assertRaisesRegex(ValueError, "decision"): cli._baseline_disclosure({**base, "train_data_status": "unknown", "clean_winner_eligible": False}, allow)
        unknown = cli._baseline_disclosure({**base, "train_data_status": "unknown", "clean_winner_eligible": False}, {**allow, "baseline_decisions": [decision]})
        self.assertEqual(unknown["clean_winner_claim"], "not-eligible"); self.assertIn("htdemucs", unknown["disclosure_ko"])
        evidence = {"training_data_evidence": ["doc"]}
        verified = cli._baseline_disclosure({**base, "train_data_status": "verified-excluded", "clean_winner_eligible": True, **evidence}, {})
        self.assertEqual((verified["clean_winner_claim"], verified["disclosure_ko"]), ("eligible", None))
        # review finding 3: the stored flag must agree with status + evidence
        with self.assertRaisesRegex(ValueError, "disagrees"): cli._baseline_disclosure({**base, "train_data_status": "verified-excluded", "clean_winner_eligible": True}, {})
        with self.assertRaisesRegex(ValueError, "disagrees"): cli._baseline_disclosure({**base, "train_data_status": "unknown", "clean_winner_eligible": True}, {**allow, "baseline_decisions": [decision]})

    def test_claim_state_machine(self):
        with tempfile.TemporaryDirectory() as tmp:
            claim, results = Path(tmp) / "c", Path(tmp) / "r.json"
            self.assertEqual(claim_start(claim, seal_sha256="s", results_path=results, now="t0")["action"], "run")
            self.assertEqual(claim_start(claim, seal_sha256="s", results_path=results, now="t1")["action"], "run")
            with self.assertRaises(RuntimeError): claim_start(claim, seal_sha256="other", results_path=results, now="t2")
            results.write_text(json.dumps({"seal_sha256": "s"}))
            self.assertEqual(claim_start(claim, seal_sha256="s", results_path=results, now="t3")["action"], "finalize")
            self.assertEqual(claim_complete(claim, seal_sha256="s", results_sha256="h", now="t4")["status"], "completed")
            with self.assertRaises(RuntimeError): claim_start(claim, seal_sha256="s", results_path=results, now="t5")

    def test_registry_is_shared_and_names_all_norbert_variants(self):
        names = set(cli.candidate_registry())
        self.assertTrue({"C1-norbert-0", "C1-norbert-1", "C1-norbert-2", "identity"} <= names)


if __name__ == "__main__":
    unittest.main()
