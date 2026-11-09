"""CPU tests: demucs official pre/post normalization (P1-4), registered-file <-> loaded-file matching (P2-2), MSST adapter and registration (#4)."""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
from experiments.banelp import cli, separators
from experiments.banelp.contracts import SOURCES, sha256_file

try:
    import torch
    from demucs.apply import apply_model
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

ROOT = Path(__file__).parents[1]
NATIVE = ["drums", "bass", "other", "vocals"]


def _run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out): code = cli.main([str(a) for a in argv])
    assert code == 0
    return json.loads(out.getvalue())


if HAVE_TORCH:
    class TinyModel(torch.nn.Module):
        """Deterministic, NOT scale-equivariant (tanh), so missing normalization changes the result."""
        samplerate = 100; segment = 1.0; audio_channels = 2; sources = NATIVE

        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.tensor([1.5, -0.7, 0.9, 2.1])); self.register_buffer("b", torch.tensor([0.05, -0.1, 0.0, 0.2]))

        def forward(self, x):  # (B, C, L) -> (B, S, C, L)
            return torch.tanh(self.w[None, :, None, None] * x[:, None] + self.b[None, :, None, None])


def official_reference(model, wav, **kw):
    """Verbatim copy of demucs 4.0.1 separate.py:168-177 (the test's own copy, independent of the package's)."""
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    sources = apply_model(model, wav[None], device="cpu", shifts=0, split=True, overlap=0.25, progress=False, num_workers=0, segment=None)[0]
    sources *= ref.std()
    sources += ref.mean()
    return sources


@unittest.skipUnless(HAVE_TORCH, "torch/demucs not installed")
class OfficialNormalizationTests(unittest.TestCase):
    FIXTURES = {"asymmetric": lambda r: np.stack([r.normal(0.3, .5, 260), r.normal(-0.1, .05, 260)]),
                "dc-offset": lambda r: r.normal(0.4, .1, (2, 260)) + 0.35, "x0.01": lambda r: .01 * r.normal(0, 1, (2, 260)), "x10": lambda r: 10 * r.normal(0, 1, (2, 260))}

    def wrapper(self, wav):
        return separators.official_demucs_apply(TinyModel(), wav.clone(), device="cpu", shifts=0, split=True, overlap=.25, segment=None)

    def test_wrapper_matches_the_official_lines_on_fixtures(self):
        for name, make in self.FIXTURES.items():
            wav = torch.from_numpy(make(np.random.default_rng(len(name))).astype(np.float32))
            expected = official_reference(TinyModel(), wav.clone())
            got, info = self.wrapper(wav)
            self.assertTrue(torch.equal(got, expected), (name, float((got - expected).abs().max())))
            self.assertFalse(info["centered_only"])
            self.assertAlmostEqual(info["ref_mean"], float(wav.mean(0).mean()), places=6)
            reference = separators.official_demucs_reference(TinyModel(), wav.clone(), device="cpu", segment=None)  # the package's verbatim copy used by smoke
            self.assertTrue(torch.equal(reference, expected))

    def test_the_fixtures_would_fail_without_the_normalization(self):
        for name, make in self.FIXTURES.items():
            wav = torch.from_numpy(make(np.random.default_rng(len(name))).astype(np.float32))
            expected = official_reference(TinyModel(), wav.clone())
            raw = apply_model(TinyModel(), wav.clone()[None], device="cpu", shifts=0, split=True, overlap=.25, progress=False, num_workers=0, segment=None)[0]
            self.assertGreater(float((raw - expected).abs().max()), 1e-3, name)

    def test_input_is_not_modified_and_split_seams_are_covered(self):
        wav = torch.from_numpy(self.FIXTURES["asymmetric"](np.random.default_rng(1)).astype(np.float32)); before = wav.clone()
        self.wrapper(wav); self.assertTrue(torch.equal(wav, before))
        self.assertGreater(wav.shape[-1], int(TinyModel.samplerate * TinyModel.segment))  # more than one segment

    def test_zero_std_input_is_centred_only_and_stays_finite(self):
        for value in (0.0, 0.25):
            wav = torch.full((2, 260), value)
            got, info = self.wrapper(wav)
            self.assertTrue(info["centered_only"]); self.assertEqual(info["ref_std"], 0.0); self.assertTrue(torch.isfinite(got).all())
            # centred input is all zero; the model output for 0 is tanh(b), restored with + mean
            expected = torch.tanh(TinyModel().b)[:, None, None].expand(4, 2, 260) + value
            self.assertTrue(torch.allclose(got, expected, atol=1e-6))
            with torch.no_grad(): nan = official_reference(TinyModel(), wav.clone())
            self.assertTrue(torch.isnan(nan).any())  # the official path itself would give NaN here
        self.assertIn("centred only", separators.STD_ZERO_POLICY)


@unittest.skipUnless(HAVE_TORCH, "torch/demucs not installed")
class RegisteredFileMatchingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); base = Path(self.tmp.name).resolve()
        self.remote, self.hub = base / "remote", base / "hub"; self.remote.mkdir(); self.hub.mkdir()
        self.files = []
        for sig, payload in (("aaaa1111", b"member-one"), ("cccc3333", b"member-two")):
            digest = hashlib.sha256(payload).hexdigest(); name = f"{sig}-{digest[:8]}.th"; (self.hub / name).write_bytes(payload); self.files.append((sig, name, payload))
        (self.remote / "files.txt").write_text("# comment\nroot: hybrid_transformer/\n" + "\n".join(name for _, name, _ in self.files) + "\n")
        (self.remote / "bagname.yaml").write_text("models: ['aaaa1111', 'cccc3333']\nweights: [[1,1,1,1],[0,1,0,1]]\n")

    def record(self, names=None, model_id="bagname"):
        members = [{"path": str(self.hub / n), "sha256": sha256_file(self.hub / n)} for n in (names or [f for _, f, _ in self.files])]
        return {"model_id": model_id, "members": members}

    def test_expected_members_follow_yaml_and_files_txt(self):
        expected = separators.demucs_expected_members("bagname", remote_root=self.remote)
        self.assertEqual([m["filename"] for m in expected["members"]], [f for _, f, _ in self.files]); self.assertTrue(expected["bag"])
        self.assertEqual(expected["weights"], [[1, 1, 1, 1], [0, 1, 0, 1]])
        single = separators.demucs_expected_members("aaaa1111", remote_root=self.remote); self.assertFalse(single["bag"]); self.assertEqual(len(single["members"]), 1)
        with self.assertRaises(ValueError): separators.demucs_expected_members("missing", remote_root=self.remote)

    def test_real_htdemucs_resolves_to_one_known_file(self):
        expected = separators.demucs_expected_members("htdemucs")
        self.assertEqual([m["filename"] for m in expected["members"]], ["955717e8-8726e21a.th"])
        self.assertEqual(len(separators.demucs_expected_members("htdemucs_ft")["members"]), 4)

    def test_load_uses_exactly_the_verified_files_in_resolver_order(self):
        loaded = []
        class Bag:
            def __init__(self, models, weights, segment): self.models, self.weights, self.segment = models, weights, segment
            def eval(self): self.evaluated = True; return self
        model = separators.load_registered_demucs(self.record(list(reversed([f for _, f, _ in self.files]))), "bagname", remote_root=self.remote,
                                                  load_model=lambda path: loaded.append(Path(path).name) or Path(path).name, bag_cls=Bag)
        self.assertEqual(loaded, [f for _, f, _ in self.files]); self.assertEqual(model.models, loaded); self.assertTrue(model.evaluated)

    def test_unrelated_file_with_the_right_hash_is_refused(self):
        # hub holds unrelated.th (hash registered correctly) while get_model would resolve the two real members.
        (self.hub / "unrelated.th").write_bytes(self.files[0][2])
        with self.assertRaisesRegex(ValueError, "differ from the files get_model"):
            separators.load_registered_demucs(self.record(["unrelated.th", self.files[1][1]]), "bagname", remote_root=self.remote, load_model=lambda p: p, bag_cls=list)

    def test_bag_size_mismatch_and_wrong_hash_are_refused(self):
        with self.assertRaisesRegex(ValueError, "differ from the files"): separators.load_registered_demucs(self.record([self.files[0][1]]), "bagname", remote_root=self.remote, load_model=lambda p: p)
        record = self.record(); record["members"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "registered checkpoint missing or changed"): separators.load_registered_demucs(record, "bagname", remote_root=self.remote, load_model=lambda p: p)
        # file content changed AND re-registered with its new hash: the name's hash prefix no longer matches
        (self.hub / self.files[0][1]).write_bytes(b"swapped"); record = self.record()
        with self.assertRaisesRegex(ValueError, "hash prefix"): separators.load_registered_demucs(record, "bagname", remote_root=self.remote, load_model=lambda p: p)
        with self.assertRaisesRegex(ValueError, "does not match"): separators.load_registered_demucs(self.record(model_id="other"), "bagname", remote_root=self.remote)


def _msst_record(base, *, status="documented-train-only", ckpt_name="model_bs_roformer_ep_17_sdr_9.6568.ckpt", order=NATIVE, extra_declaration=True):
    ckpt, cfg = base / ckpt_name, base / "config.yaml"
    if HAVE_TORCH:
        model = FakeMSSTModel(); torch.save(model.state_dict(), ckpt)
    cfg.write_text("training:\n  instruments: [drums, bass, other, vocals]\n")
    declaration = separators.declaration_record("https://example.invalid/docs", "train only", "v1.0.12") if extra_declaration else None
    return separators.model_record(model_id="bsr", checkpoints=[ckpt], native_source_order=order, samplerate=44100, train_data_status=status, family="msst", model_type="bs_roformer",
                                   config_file={"path": str(cfg), "sha256": sha256_file(cfg), "url": None}, declaration=declaration, msst_commit="a" * 40)


if HAVE_TORCH:
    class FakeMSSTModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.scale = torch.nn.Parameter(torch.tensor([1., 2., 3., 4.]))
        def forward(self, x): return x

    class FakeAPI:
        """Stands in for MSST utils: native-order outputs per instrument, like demix()."""
        def __init__(self, instruments=NATIVE, keys=None): self.instruments, self.keys, self.calls = instruments, keys, []
        def get_model_from_config(self, model_type, config_path):
            self.calls.append(("get_model", model_type, config_path))
            return FakeMSSTModel(), types.SimpleNamespace(training=types.SimpleNamespace(instruments=list(self.instruments)))
        def demix(self, config, model, mix, device, model_type=None, pbar=False):
            self.calls.append(("demix", mix.shape, device, model_type, pbar))
            scales = model.scale.detach().numpy()
            return {name: (mix * float(scales[i])).astype(np.float32) for i, name in enumerate(self.keys or NATIVE)}


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class MSSTAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.base = Path(self.tmp.name).resolve()

    def test_reorders_to_the_contract_source_order(self):
        record = _msst_record(self.base); api = FakeAPI()
        sep = separators.MSSTSeparator(registered=record, runtime_dir=self.base, device="cpu", api=api)
        x = np.random.default_rng(0).normal(size=(2, 500)).astype(np.float32); out = sep(x)
        self.assertEqual(out.shape, (4, 2, 500)); native_scale = dict(zip(NATIVE, [1., 2., 3., 4.]))
        for i, name in enumerate(SOURCES): self.assertTrue(np.allclose(out[i], x * native_scale[name]), name)
        self.assertEqual(api.calls[0], ("get_model", "bs_roformer", record["config_file"]["path"])); self.assertEqual(api.calls[1][3:], ("bs_roformer", False))

    def test_hash_and_config_checks(self):
        record = _msst_record(self.base)
        Path(record["members"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "missing or changed"): separators.MSSTSeparator(registered=record, runtime_dir=self.base, device="cpu", api=FakeAPI())
        record = _msst_record(self.base); Path(record["config_file"]["path"]).write_text("training:\n  instruments: [vocals]\n")
        with self.assertRaisesRegex(ValueError, "config missing or changed"): separators.MSSTSeparator(registered=record, runtime_dir=self.base, device="cpu", api=FakeAPI())
        record = _msst_record(self.base)
        with self.assertRaisesRegex(ValueError, "instruments"): separators.MSSTSeparator(registered=record, runtime_dir=self.base, device="cpu", api=FakeAPI(instruments=SOURCES))
        with self.assertRaisesRegex(ValueError, "not an MSST"): separators.MSSTSeparator(registered={**record, "family": "demucs"}, runtime_dir=self.base, device="cpu", api=FakeAPI())

    def test_bad_outputs_are_refused_and_cuda_is_never_a_silent_fallback(self):
        record = _msst_record(self.base); x = np.zeros((2, 100), np.float32)
        sep = separators.MSSTSeparator(registered=record, runtime_dir=self.base, device="cpu", api=FakeAPI(keys=["a", "b", "c", "d"]))
        with self.assertRaisesRegex(ValueError, "contract sources"): sep(x)
        with self.assertRaises(ValueError): sep(np.zeros((1, 100), np.float32))
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is prohibited"): separators.MSSTSeparator(registered=record, runtime_dir=self.base, api=FakeAPI())

    def test_load_msst_api_from_a_pinned_clean_checkout(self):
        repo = self.base / "msst"; repo.mkdir()
        (repo / "utils.py").write_text("def get_model_from_config(model_type, config_path): return ('m', 'c')\ndef demix(*a, **k): return {}\n")
        git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q"); git("add", "."); git("commit", "-q", "-m", "x"); commit = git("rev-parse", "HEAD")
        try:
            api = separators.load_msst_api(repo, commit); self.assertEqual(api.get_model_from_config("t", "p"), ("m", "c"))
            with self.assertRaisesRegex(ValueError, "registered pin"): separators.load_msst_api(repo, "0" * 40)
            (repo / "utils.py").write_text("changed = 1\n")
            with self.assertRaisesRegex(ValueError, "local modifications"): separators.load_msst_api(repo, commit)
            with self.assertRaises(FileNotFoundError): separators.load_msst_api(self.base / "nowhere", commit)
        finally: sys.modules.pop("utils", None)


class EligibilityVocabularyTests(unittest.TestCase):
    def test_documented_train_only_requires_the_full_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = _msst_record(base)
            self.assertEqual(record["declaration"], {"url": "https://example.invalid/docs", "quote": "train only", "release": "v1.0.12"})
            self.assertEqual((record["clean_winner_eligible"], record["candidate_eligible"]), (False, True))
            eligibility = separators.record_eligibility(record); self.assertEqual(eligibility["label_ko"], "문서상 train-only, 곡별 감사 미확인")
            with self.assertRaisesRegex(ValueError, "needs declaration"): _msst_record(base, extra_declaration=False)
            for args in (("u", None, "r"), ("u", "q", None), (None, "q", "r")):
                with self.assertRaises(ValueError): separators.declaration_record(*args)
            self.assertIsNone(separators.declaration_record(None, None, None))
            unknown = separators.record_eligibility({"train_data_status": "unknown"}); self.assertEqual((unknown["clean_winner_eligible"], unknown["candidate_eligible"]), (False, False))
            excluded = separators.record_eligibility({"train_data_status": "verified-excluded", "training_data_evidence": ["doc"]}); self.assertEqual((excluded["clean_winner_eligible"], excluded["candidate_eligible"]), (True, True))
            self.assertFalse(separators.record_eligibility({"train_data_status": "verified-excluded"})["candidate_eligible"])
            with self.assertRaises(ValueError): separators.record_eligibility({"train_data_status": "made-up"})

    def test_msst_record_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m.ckpt"; ckpt.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "model_type"): separators.model_record(model_id="m", checkpoints=[ckpt], native_source_order=NATIVE, samplerate=44100, train_data_status="unknown", family="msst")
            with self.assertRaisesRegex(ValueError, "exactly one"): separators.model_record(model_id="m", checkpoints=[ckpt, ckpt], native_source_order=NATIVE, samplerate=44100, train_data_status="unknown", family="msst")
            with self.assertRaisesRegex(ValueError, "family"): separators.model_record(model_id="m", checkpoints=[ckpt], native_source_order=NATIVE, samplerate=44100, train_data_status="unknown", family="other")


class RegistrationTableTests(unittest.TestCase):
    def test_table_lists_the_five_allowed_and_four_excluded(self):
        table = separators.load_eligibility_table()
        self.assertEqual({e["file"] for e in table["allowed"]}, {"model_bs_roformer_ep_17_sdr_9.6568.ckpt", "scnet_checkpoint_musdb18.ckpt", "SCNet-large_starrytong_fixed.ckpt", "model_scnet_sdr_9.3244.ckpt", "model_scnet_ep_54_sdr_9.8051.ckpt"})
        self.assertEqual(len(table["excluded"]), 4); self.assertTrue(all(e["reason"] for e in table["excluded"]))
        self.assertEqual({e["file"]: e["release"] for e in table["allowed"]}["model_bs_roformer_ep_17_sdr_9.6568.ckpt"], "v1.0.12")

    def test_check_registration_allowed(self):
        ok = dict(model_id="bsr", filenames=["model_bs_roformer_ep_17_sdr_9.6568.ckpt"], status="documented-train-only", release="v1.0.12", family="msst")
        separators.check_registration_allowed(**ok)
        for edit, message in (({"release": "v9"}, "release"), ({"filenames": ["other.ckpt"]}, "not listed"), ({"family": "demucs"}, "msst checkpoint"),
                              ({"model_id": "mdx_extra", "filenames": ["x.th"], "status": "unknown"}, "excluded"),
                              ({"filenames": ["MelBandRoformer.ckpt"], "status": "unknown"}, "excluded"),
                              ({"filenames": ["model_bs_roformer_ep_317_sdr_12.9755.ckpt"], "status": "documented-train-only"}, "excluded")):
            with self.assertRaisesRegex(ValueError, message): separators.check_registration_allowed(**{**ok, **edit})
        separators.check_registration_allowed(model_id="other", filenames=["unlisted.th"], status="unknown", release=None, family="demucs")

    def test_register_model_cli_for_msst_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); runtime, artifact = base / "runtime", base / "artifacts"; common = ["--runtime-dir", runtime, "--artifact-dir", artifact, "--config", ROOT / "experiments/banelp/config.json"]
            ckpt = base / "model_bs_roformer_ep_17_sdr_9.6568.ckpt"; ckpt.write_bytes(b"weights"); cfg = base / "bsr.yaml"; cfg.write_text("training: {}\n")
            args = ["register-model", *common, "--model-id", "bsr", "--checkpoint", ckpt, "--family", "msst", "--model-type", "bs_roformer", "--msst-config", cfg, "--msst-commit", "a" * 40,
                    "--native-source-order", "drums,bass,other,vocals", "--samplerate", "44100", "--train-data-status", "documented-train-only", "--release", "v1.0.12",
                    "--declaration-url", "https://example.invalid/pretrained_models.md", "--declaration-quote", "MUSDB18HQ train 100 only", "--url", "https://example.invalid/ckpt", "--msst-config-url", "https://example.invalid/cfg"]
            result = _run(*args)["model"]
            self.assertEqual((result["family"], result["model_type"], result["msst_commit"], result["candidate_eligible"], result["clean_winner_eligible"]), ("msst", "bs_roformer", "a" * 40, True, False))
            self.assertEqual(result["config_file"]["sha256"], sha256_file(cfg)); self.assertEqual(result["members"][0]["sha256"], sha256_file(ckpt)); self.assertEqual(result["declaration"]["release"], "v1.0.12")
            self.assertTrue((artifact / "evidence/manifests/models.json").is_file())
            with self.assertRaisesRegex(ValueError, "declaration"): _run(*[a for i, a in enumerate(args) if args[i - 1] != "--declaration-quote" and a != "--declaration-quote"])
            with self.assertRaisesRegex(ValueError, "needs --model-type"): _run(*[a for i, a in enumerate(args) if args[i - 1] != "--msst-commit" and a != "--msst-commit"])
            with self.assertRaisesRegex(ValueError, "release"): _run(*[("v9" if a == "v1.0.12" else a) for a in args])
            other = base / "viperx.ckpt"; other.write_bytes(b"w")
            with self.assertRaisesRegex(ValueError, "not listed"): _run(*[(str(other) if a == ckpt else a) for a in args])
            excluded = base / "MelBandRoformer.ckpt"; excluded.write_bytes(b"w")
            with self.assertRaisesRegex(ValueError, "excluded"): _run(*[(str(excluded) if a == ckpt else a) for a in args])
            self.assertEqual(len(json.loads((runtime / "manifests/models.json").read_text())["models"]), 1)

    def test_smoke_accepts_a_model_flag(self):
        args = cli._parser().parse_args(["smoke", "--runtime-dir", "r", "--artifact-dir", "a", "--config", "c", "--device", "cuda", "--require-cuda", "--model", "bsr", "--seconds", "30"])
        self.assertEqual((args.model, args.seconds), ("bsr", 30))
        self.assertIsNone(cli._parser().parse_args(["smoke", "--runtime-dir", "r", "--artifact-dir", "a", "--config", "c"]).model)


if __name__ == "__main__":
    unittest.main()
