"""CPU tests for the wet cache, RIR acquisition and the develop -> seal -> evaluate guard path."""
import contextlib
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
from experiments.banelp.contracts import SOURCES, sha256_file, validate_tensors
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
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.tmp.cleanup)
        base = Path(cls.tmp.name).resolve(); cls.runtime, cls.artifact = base / "runtime", base / "artifacts"
        config = json.loads((ROOT / "experiments/banelp/config.json").read_text())
        config["samples"] = SAMPLES; config["selection"] = {"sar_policy_status": "confirmed", "sar_floor_db": -100.0, "minimum_sdr_gain_db": .1}
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
        ckpt = cls.runtime / "weights.th"; ckpt.write_bytes(b"stub-weights")
        _run("register-model", *cls.common, "--model-id", "htdemucs", "--checkpoint", ckpt, "--native-source-order", "drums,bass,other,vocals",
             "--samplerate", "44100", "--train-data-status", "verified-excluded", "--evidence", "test-evidence")
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
        cls.develop = _run("develop", *cls.common)

    def setUp(self):
        (self.runtime / "manifests/models.json").write_bytes(self.models_bytes)
        shutil.rmtree(self.runtime / "claims", ignore_errors=True)
        (self.runtime / "results/evaluation.json").unlink(missing_ok=True)
        (self.artifact / "selection-seal.json").unlink(missing_ok=True)

    def test_wet_cache_matches_reader_contract_and_is_reused(self):
        summary = _run("build-wet", *self.common)
        self.assertEqual((summary["written"], summary["reused"]), (0, 200))
        manifest = json.loads((self.runtime / "manifests/models.json").read_text())
        self.assertEqual(manifest["models"][0]["bag_size"], 1); self.assertTrue(manifest["models"][0]["clean_winner_eligible"])

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

    def test_seal_refuses_stale_develop_results(self):
        ckpt = self.runtime / "weights2.th"; ckpt.write_bytes(b"other-weights")
        _run("register-model", *self.common, "--model-id", "other-model", "--checkpoint", ckpt, "--native-source-order", "drums,bass,other,vocals",
             "--samplerate", "44100", "--train-data-status", "unknown")
        with self.assertRaises(ValueError): _run("seal", *self.common)


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

    def test_select_candidate_uses_develop_schema_and_gates(self):
        good = {"name": "A", "status": "ok", "summary": {"track_median_delta_db": 1.0, "track_bootstrap_95_ci": [.5, 1.5]}, "wilcoxon_p": .01,
                "stem_median_delta_sar": {s: 0.0 for s in SOURCES}, "additional_inference_calls": 0}
        weak = {**good, "name": "B", "summary": {"track_median_delta_db": .05, "track_bootstrap_95_ci": [-.1, .2]}}
        low_sar = {**good, "name": "C", "stem_median_delta_sar": {**{s: 0.0 for s in SOURCES}, "bass": -3.0}}
        self.assertEqual(select_candidate([good, weak, low_sar], sar_policy_status="confirmed", sar_floor_db=-1.0)["selected"], "A")
        self.assertEqual(select_candidate([weak, low_sar], sar_policy_status="confirmed", sar_floor_db=-1.0)["selected"], "identity")

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
