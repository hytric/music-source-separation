"""CPU tests: demucs official pre/post normalization (P1-4), registered-file <-> loaded-file matching (P2-2), MSST adapter and registration (#4)."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
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


class AttrDict(dict):
    """Config stand-in with MSST's access styles: `"k" in cfg.inference`, `cfg.inference["k"]` and `cfg.training.use_amp`."""
    def __getattr__(self, name):
        try: return self[name]
        except KeyError as exc: raise AttributeError(name) from exc
    def __setattr__(self, name, value): self[name] = value


def fake_config(*, normalize=True, use_amp=True, batch_size=3, instruments=NATIVE, chunk=100, overlap=2):
    inference = {"batch_size": batch_size, "num_overlap": overlap}
    if normalize is not None: inference["normalize"] = normalize
    return AttrDict(audio=AttrDict(chunk_size=chunk), training=AttrDict(instruments=list(instruments), use_amp=use_amp), inference=AttrDict(inference))


def _msst_record(base, *, status="documented-train-only", ckpt_name="model_bs_roformer_ep_17_sdr_9.6568.ckpt", order=NATIVE, extra_declaration=True, overrides=None):
    ckpt, cfg = base / ckpt_name, base / "config.yaml"
    if HAVE_TORCH:
        model = FakeMSSTModel(); torch.save(model.state_dict(), ckpt)
    cfg.write_text("training:\n  instruments: [drums, bass, other, vocals]\n")
    declaration = separators.declaration_record("https://example.invalid/docs", "train only", "v1.0.12") if extra_declaration else None
    return separators.model_record(model_id="bsr", checkpoints=[ckpt], native_source_order=order, samplerate=44100, train_data_status=status, family="msst", model_type="bs_roformer",
                                   config_file={"path": str(cfg), "sha256": sha256_file(cfg), "url": None}, declaration=declaration, msst_commit="a" * 40, inference_overrides=overrides)


# Verbatim copies of the pinned MSST (tag v1.0.23) functions the fakes stand in for, so fake tests exercise the same arithmetic/key handling.
def msst_normalize_audio(audio):
    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def msst_denormalize_audio(audio, norm_params):
    return audio * norm_params["std"] + norm_params["mean"]


def msst_load_start_checkpoint(args, model, old_model, type_="train"):
    """utils/model_utils.py:698 load_start_checkpoint, `type_ != 'train'` and not htdemucs/apollo branch."""
    if 'state' in old_model:
        old_model = old_model['state']
    if 'state_dict' in old_model:
        old_model = old_model['state_dict']
    if 'model_state_dict' in old_model:
        old_model = old_model['model_state_dict']
    model.load_state_dict(old_model)


def official_sequence(demix, config, model, mix, device, model_type):
    """The test's own copy of MSST inference.py:104-106 and 142-145 around one demix call (bigshifts=1)."""
    if "normalize" in config.inference:
        if config.inference["normalize"] is True:
            mix, norm_params = msst_normalize_audio(mix)
    waveforms_orig = demix(config, model, mix, device, model_type=model_type, pbar=False)
    result = {}
    for instr in config.training.instruments:
        estimates = waveforms_orig[instr]
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                estimates = msst_denormalize_audio(estimates, norm_params)
        result[instr] = estimates
    return result


if HAVE_TORCH:
    class FakeMSSTModel(torch.nn.Module):
        """Per-instrument tanh(scale*x + bias): NOT scale-equivariant (missing normalization shows), elementwise (batch size cannot change the result)."""
        def __init__(self): super().__init__(); self.scale = torch.nn.Parameter(torch.tensor([1., 2., 3., 4.])); self.bias = torch.nn.Parameter(torch.tensor([.05, -.1, 0., .2]))
        def forward(self, x):  # (B, C, L) -> (B, 4, C, L)
            return torch.tanh(self.scale[None, :, None, None] * x[:, None] + self.bias[None, :, None, None])

    class FakeAPI:
        """Stands in for MSST utils: chunked, batched, native-order outputs per instrument like demix(); autocast is emulated by refusing use_amp=True."""
        def __init__(self, config=None, keys=None):
            self.config, self.keys, self.calls, self.model_calls = config or fake_config(), keys, [], []
            self.normalize_audio, self.denormalize_audio, self.load_start_checkpoint = msst_normalize_audio, msst_denormalize_audio, msst_load_start_checkpoint
        def get_model_from_config(self, model_type, config_path):
            self.calls.append(("get_model", model_type, config_path)); return FakeMSSTModel(), self.config
        def demix(self, config, model, mix, device, model_type=None, pbar=False):
            use_amp = getattr(config.training, "use_amp", True)  # utils/model_utils.py:195
            self.calls.append(("demix", mix.shape, device, model_type, pbar, use_amp))
            if use_amp: raise AssertionError("demix would run under fp16 autocast (config.training.use_amp is True)")
            chunk, batch, mix_t = config.audio.chunk_size, config.inference.batch_size, torch.tensor(mix, dtype=torch.float32)
            parts = [torch.nn.functional.pad(mix_t[:, i:i + chunk], (0, chunk - mix_t[:, i:i + chunk].shape[-1])) for i in range(0, mix_t.shape[1], chunk)]
            outs = []
            with torch.no_grad():
                for j in range(0, len(parts), batch):
                    self.model_calls.append(len(parts[j:j + batch])); outs.append(model(torch.stack(parts[j:j + batch])))
            full = torch.cat(outs, 0).permute(1, 2, 0, 3).reshape(4, 2, -1)[..., :mix.shape[-1]].numpy()
            names = self.keys or list(config.training.instruments)
            return {name: full[i].astype(np.float32) for i, name in enumerate(names)}


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class MSSTAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.base = Path(self.tmp.name).resolve()

    def sep(self, record=None, api=None, **kw):
        return separators.MSSTSeparator(registered=record or _msst_record(self.base), runtime_dir=self.base, device="cpu", api=api or FakeAPI(), **kw)

    def test_reorders_to_the_contract_source_order(self):
        record = _msst_record(self.base); api = FakeAPI(fake_config(normalize=False)); sep = self.sep(record, api)
        x = np.random.default_rng(0).normal(size=(2, 500)).astype(np.float32) * .3; out = sep(x)
        self.assertEqual(out.shape, (4, 2, 500)); native_scale = dict(zip(NATIVE, range(4)))
        model = FakeMSSTModel()
        for i, name in enumerate(SOURCES):
            k = native_scale[name]; expected = torch.tanh(model.scale[k] * torch.from_numpy(x) + model.bias[k]).detach().numpy()
            self.assertTrue(np.allclose(out[i], expected, atol=1e-6), name)
        self.assertEqual(api.calls[0], ("get_model", "bs_roformer", record["config_file"]["path"])); self.assertEqual(api.calls[1][3:5], ("bs_roformer", False))

    def test_hash_and_config_checks(self):
        record = _msst_record(self.base)
        Path(record["members"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "missing or changed"): self.sep(record)
        record = _msst_record(self.base); Path(record["config_file"]["path"]).write_text("training:\n  instruments: [vocals]\n")
        with self.assertRaisesRegex(ValueError, "config missing or changed"): self.sep(record)
        record = _msst_record(self.base)
        with self.assertRaisesRegex(ValueError, "instruments"): self.sep(record, FakeAPI(fake_config(instruments=SOURCES)))
        with self.assertRaisesRegex(ValueError, "not an MSST"): self.sep({**record, "family": "demucs"})

    def test_bad_outputs_are_refused_and_cuda_is_never_a_silent_fallback(self):
        record = _msst_record(self.base); x = np.zeros((2, 100), np.float32)
        sep = self.sep(record, FakeAPI(fake_config(normalize=False), keys=["a", "b", "c", "d"]))
        with self.assertRaisesRegex(ValueError, "contract sources"): sep(x)
        with self.assertRaises(ValueError): sep(np.zeros((1, 100), np.float32))
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is prohibited"): separators.MSSTSeparator(registered=record, runtime_dir=self.base, api=FakeAPI())

    def test_load_msst_api_from_a_pinned_clean_checkout(self):
        repo = self.base / "msst"; repo.mkdir()
        (repo / "utils.py").write_text("def get_model_from_config(model_type, config_path): return ('m', 'c')\ndef demix(*a, **k): return {}\ndef load_start_checkpoint(*a, **k): pass\n"
                                       "def normalize_audio(a): return a\ndef denormalize_audio(a, p): return a\n")
        git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q"); git("add", "."); git("commit", "-q", "-m", "x"); commit = git("rev-parse", "HEAD")
        try:
            api = separators.load_msst_api(repo, commit); self.assertEqual(api.get_model_from_config("t", "p"), ("m", "c"))
            self.assertNotIn(str(repo.resolve()), sys.path)
            with api.sys_path(): self.assertEqual(sys.path[0], str(repo.resolve()))
            self.assertNotIn(str(repo.resolve()), sys.path)
            with self.assertRaisesRegex(ValueError, "registered pin"): separators.load_msst_api(repo, "0" * 40)
            (repo / "utils.py").write_text("changed = 1\n")
            with self.assertRaisesRegex(ValueError, "local modifications"): separators.load_msst_api(repo, commit)
            with self.assertRaises(FileNotFoundError): separators.load_msst_api(self.base / "nowhere", commit)
        finally: sys.modules.pop("utils", None)

    def test_pinned_commit_is_the_v1_0_23_tag(self):
        self.assertEqual(separators.MSST_PINNED_COMMIT, "f17924ddb1817cdbae6353706708c4518dd7cd14")

    # -- item 1: FP32 -------------------------------------------------------------------------------------------------
    def test_use_amp_is_forced_off_recorded_and_demix_never_sees_autocast(self):
        api = FakeAPI(fake_config(use_amp=True)); self.assertTrue(api.config.training.use_amp)
        sep = self.sep(api=api); self.assertIs(api.config.training.use_amp, False)
        self.assertTrue(sep.attestation["config_use_amp"]); self.assertTrue(sep.attestation["use_amp_forced_false"]); self.assertFalse(sep.attestation["autocast_enabled"]); self.assertTrue(sep.attestation["fp32"])
        sep(np.random.default_rng(1).normal(size=(2, 250)).astype(np.float32)); self.assertEqual([c[5] for c in api.calls if c[0] == "demix"], [False])
        self.assertTrue(all(p.dtype == torch.float32 for p in sep.model.parameters())); self.assertFalse(sep.model.training)

    def test_the_test_fails_if_autocast_is_enabled(self):
        api = FakeAPI(fake_config(use_amp=True)); sep = self.sep(api=api)
        sep.config.training.use_amp = True  # someone re-enables mixed precision after construction
        with self.assertRaisesRegex(RuntimeError, "use_amp must be False"): sep(np.ones((2, 100), np.float32))
        with self.assertRaisesRegex(AssertionError, "autocast"): api.demix(api.config, sep.model, np.ones((2, 100), np.float32), "cpu")  # the fake demix itself trips on use_amp True

    # -- item 2: official normalization -------------------------------------------------------------------------------
    def test_adapter_matches_the_official_sequence_with_and_without_normalization(self):
        rng = np.random.default_rng(3)
        fixtures = {"dc-offset": rng.normal(.3, .2, (2, 430)), "x0.05": .05 * rng.normal(0, 1, (2, 430)), "asymmetric": np.stack([rng.normal(.2, .5, 430), rng.normal(-.1, .05, 430)])}
        for normalize in (True, False, None):
            for name, mix in fixtures.items():
                mix = mix.astype(np.float32); api = FakeAPI(fake_config(normalize=normalize, use_amp=False)); sep = self.sep(api=api)
                got = sep(mix)
                config = fake_config(normalize=normalize, use_amp=False); ref_api = FakeAPI(config); model = ref_api.get_model_from_config("bs_roformer", "p")[0]; model.load_state_dict(sep.model.state_dict())
                expected = official_sequence(ref_api.demix, config, model, mix.copy(), "cpu", "bs_roformer")
                self.assertTrue(np.array_equal(got, np.stack([expected[n] for n in SOURCES]).astype(np.float32)), (normalize, name))
                self.assertEqual(sep.last_normalization["normalize"], normalize is True)
                if normalize is True and name == "dc-offset":  # sensitivity: the un-normalized path is really different for this model
                    plain = self.sep(api=FakeAPI(fake_config(normalize=False)))(mix); self.assertGreater(float(np.abs(plain - got).max()), 1e-3)
        self.assertIs(separators.msst_normalize_decision(fake_config(normalize="true")), False)  # only the boolean True counts (`is True`)

    def test_reference_helper_equals_the_tests_own_copy_of_the_official_lines(self):
        config = fake_config(normalize=True, use_amp=False); api = FakeAPI(config); model = api.get_model_from_config("m", "p")[0]; mix = np.random.default_rng(5).normal(.2, .3, (2, 300)).astype(np.float32)
        a = separators.official_msst_reference(api, config, model, mix.copy(), "cpu", model_type="m"); b = official_sequence(api.demix, config, model, mix.copy(), "cpu", "m")
        for k in b: self.assertTrue(np.array_equal(a[k], b[k]))

    def test_silent_and_constant_input_is_centred_only_and_stays_finite(self):
        for value in (0.0, .25):
            mix = np.full((2, 200), value, np.float32); api = FakeAPI(fake_config(normalize=True, use_amp=False)); sep = self.sep(api=api)
            got = sep(mix)
            self.assertTrue(np.isfinite(got).all()); self.assertTrue(sep.last_normalization["centered_only"]); self.assertEqual(sep.last_normalization["std"], 1.0)
            model = FakeMSSTModel(); expected = torch.tanh(model.bias).detach().numpy()[:, None, None] + value  # centred input is 0 -> tanh(bias), restored with +mean
            native = dict(zip(NATIVE, expected)); self.assertTrue(np.allclose(got, np.stack([np.broadcast_to(native[n], (2, 200)) for n in SOURCES]), atol=1e-6))
            with np.errstate(all="ignore"): normalized, _ = msst_normalize_audio(mix)
            self.assertFalse(np.isfinite(normalized).all())  # the official normalize_audio itself divides by zero here
        no_normalize = self.sep(api=FakeAPI(fake_config(normalize=False, use_amp=False)))
        no_normalize(np.zeros((2, 100), np.float32)); self.assertFalse(no_normalize.last_normalization["centered_only"])
        self.assertIn("centred only", separators.MSST_STD_ZERO_POLICY)

    # -- item 3: official checkpoint loading --------------------------------------------------------------------------
    def test_every_checkpoint_key_form_loads_strictly_through_the_official_loader(self):
        reference = FakeMSSTModel(); reference.scale.data += 7; state = reference.state_dict(); api = FakeAPI()
        forms = {"raw": state, "state": {"state": state}, "state_dict": {"state_dict": state}, "model_state_dict": {"model_state_dict": state, "optimizer": {}}}
        for name, payload in forms.items():
            path = self.base / f"{name}.ckpt"; torch.save(payload, path); model = FakeMSSTModel()
            separators.load_msst_checkpoint(api, model, path, "bs_roformer")
            self.assertTrue(torch.equal(model.scale, reference.scale), name)
        torch.save({**{k: v for k, v in state.items() if k != "bias"}}, self.base / "missing.ckpt")
        with self.assertRaisesRegex(RuntimeError, "Missing key"): separators.load_msst_checkpoint(api, FakeMSSTModel(), self.base / "missing.ckpt", "bs_roformer")
        torch.save({**state, "extra": torch.zeros(1)}, self.base / "extra.ckpt")
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"): separators.load_msst_checkpoint(api, FakeMSSTModel(), self.base / "extra.ckpt", "bs_roformer")

    def test_official_loader_message_goes_to_stderr_not_stdout(self):
        def loud(args, model, ckpt, type_="train"): print("Start from checkpoint:", args.start_check_point); msst_load_start_checkpoint(args, model, ckpt, type_)
        api = FakeAPI(); api.load_start_checkpoint = loud; path = self.base / "m.ckpt"; torch.save(FakeMSSTModel().state_dict(), path)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err): separators.load_msst_checkpoint(api, FakeMSSTModel(), path, "bs_roformer")
        self.assertEqual(out.getvalue(), ""); self.assertIn("Start from checkpoint", err.getvalue())

    # -- item 6: batch size ---------------------------------------------------------------------------------------------
    def test_batch_size_override_is_recorded_and_does_not_change_results(self):
        mix = np.random.default_rng(9).normal(.1, .3, (2, 1000)).astype(np.float32); results = {}
        for label, kw, overrides in (("default", {}, None), ("one", {"batch_size": 1}, None), ("seven", {"batch_size": 7}, None), ("registered", {}, {"batch_size": 4})):
            api = FakeAPI(fake_config(batch_size=3, use_amp=False)); sep = self.sep(_msst_record(self.base, overrides=overrides), api, **kw); results[label] = (sep(mix), sep.attestation, api.model_calls)
        self.assertEqual({k: (v[1]["batch_size"], v[1]["config_batch_size"], v[1]["batch_size_overridden"]) for k, v in results.items()},
                         {"default": (3, 3, False), "one": (1, 3, True), "seven": (7, 3, True), "registered": (4, 3, True)})
        self.assertEqual(results["one"][2], [1] * 10); self.assertEqual(results["default"][2], [3, 3, 3, 1])  # batching really differs
        for label in ("one", "seven", "registered"): self.assertTrue(np.array_equal(results["default"][0], results[label][0]), label)
        with self.assertRaisesRegex(ValueError, ">= 1"): self.sep(api=FakeAPI(), batch_size=0)


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


RELEASES = {"model_bs_roformer_ep_17_sdr_9.6568.ckpt": "v1.0.12", "scnet_checkpoint_musdb18.ckpt": "v.1.0.6", "SCNet-large_starrytong_fixed.ckpt": "v1.0.9",
            "model_scnet_sdr_9.3244.ckpt": "v1.0.8", "model_scnet_ep_54_sdr_9.8051.ckpt": "v1.0.13"}
CONFIGS = {"model_bs_roformer_ep_17_sdr_9.6568.ckpt": "config_bs_roformer_384_8_2_485100.yaml", "scnet_checkpoint_musdb18.ckpt": "config_musdb18_scnet.yaml",
           "SCNet-large_starrytong_fixed.ckpt": "config_musdb18_scnet_large_starrytong.yaml", "model_scnet_sdr_9.3244.ckpt": "config_musdb18_scnet_large.yaml",
           "model_scnet_ep_54_sdr_9.8051.ckpt": "config_musdb18_scnet_xl.yaml"}


class RegistrationTableTests(unittest.TestCase):
    def test_table_lists_the_five_allowed_and_four_excluded(self):
        table = separators.load_eligibility_table()
        self.assertEqual({e["file"] for e in table["allowed"]}, set(RELEASES))
        self.assertEqual(len(table["excluded"]), 4); self.assertTrue(all(e["reason"] for e in table["excluded"]))
        self.assertEqual({e["file"]: e["release"] for e in table["allowed"]}, RELEASES)
        for e in table["allowed"]:
            self.assertRegex(e["sha256"], r"^[0-9a-f]{64}$"); self.assertRegex(e["config"]["sha256"], r"^[0-9a-f]{64}$"); self.assertEqual(e["config"]["file"], CONFIGS[e["file"]])
            self.assertEqual(e["url"], f"https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/{e['release']}/{e['file']}")
            self.assertEqual(e["config"]["url"], f"https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/{e['release']}/{e['config']['file']}")

    def entry(self, name="model_bs_roformer_ep_17_sdr_9.6568.ckpt"):
        e = {x["file"]: x for x in separators.load_eligibility_table()["allowed"]}[name]
        return dict(model_id="bsr", filenames=[name], status="documented-train-only", release=e["release"], family="msst", checkpoint_sha256={name: e["sha256"]},
                    config_name=e["config"]["file"], config_sha256=e["config"]["sha256"])

    def test_check_registration_allowed(self):
        ok = self.entry(); separators.check_registration_allowed(**ok)
        for name in RELEASES: separators.check_registration_allowed(**self.entry(name))
        for edit, message in (({"release": "v9"}, "release"), ({"filenames": ["other.ckpt"], "checkpoint_sha256": {}}, "not listed"), ({"family": "demucs"}, "msst checkpoint"),
                              ({"model_id": "mdx_extra", "filenames": ["x.th"], "status": "unknown"}, "excluded"),
                              ({"filenames": ["MelBandRoformer.ckpt"], "status": "unknown"}, "excluded"),
                              ({"filenames": ["model_bs_roformer_ep_317_sdr_12.9755.ckpt"], "status": "documented-train-only"}, "excluded"),
                              ({"checkpoint_sha256": {ok["filenames"][0]: "0" * 64}}, "sha256 .* differs"), ({"checkpoint_sha256": {}}, "sha256"),
                              ({"config_sha256": "1" * 64}, "config sha256"), ({"config_name": "other.yaml"}, "config 'other.yaml'"), ({"config_name": None, "config_sha256": None}, "config")):
            with self.assertRaisesRegex(ValueError, message): separators.check_registration_allowed(**{**ok, **edit})
        for name, release in (("scnet_checkpoint_musdb18.ckpt", "v1.0.6"), ("model_scnet_sdr_9.3244.ckpt", "v1.0.13"), ("model_scnet_ep_54_sdr_9.8051.ckpt", "v1.0.8")):
            with self.assertRaisesRegex(ValueError, "release"): separators.check_registration_allowed(**{**self.entry(name), "release": release})
        separators.check_registration_allowed(model_id="other", filenames=["unlisted.th"], status="unknown", release=None, family="demucs")

    def test_register_model_cli_for_msst_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); runtime, artifact = base / "runtime", base / "artifacts"; common = ["--runtime-dir", runtime, "--artifact-dir", artifact, "--config", ROOT / "experiments/banelp/config.json"]
            name = "model_bs_roformer_ep_17_sdr_9.6568.ckpt"; ckpt = base / name; ckpt.write_bytes(b"weights"); cfg = base / CONFIGS[name]; cfg.write_text("training: {}\n")
            table = separators.load_eligibility_table()  # the real table, with the listed hashes replaced by those of the fake files
            for e in table["allowed"]:
                if e["file"] == name: e["sha256"] = sha256_file(ckpt); e["config"]["sha256"] = sha256_file(cfg)
            real_table = separators.load_eligibility_table; separators.load_eligibility_table = lambda: table; self.addCleanup(setattr, separators, "load_eligibility_table", real_table)
            args = ["register-model", *common, "--model-id", "bsr", "--checkpoint", ckpt, "--family", "msst", "--model-type", "bs_roformer", "--msst-config", cfg, "--msst-commit", separators.MSST_PINNED_COMMIT,
                    "--native-source-order", "drums,bass,other,vocals", "--samplerate", "44100", "--train-data-status", "documented-train-only", "--release", "v1.0.12",
                    "--declaration-url", "https://example.invalid/pretrained_models.md", "--declaration-quote", "MUSDB18HQ train 100 only", "--url", "https://example.invalid/ckpt", "--msst-config-url", "https://example.invalid/cfg"]
            result = _run(*args)["model"]
            self.assertEqual((result["family"], result["model_type"], result["msst_commit"], result["candidate_eligible"], result["clean_winner_eligible"]), ("msst", "bs_roformer", separators.MSST_PINNED_COMMIT, True, False))
            self.assertEqual(result["config_file"]["sha256"], sha256_file(cfg)); self.assertEqual(result["members"][0]["sha256"], sha256_file(ckpt)); self.assertEqual(result["declaration"]["release"], "v1.0.12")
            self.assertIsNone(result["inference_overrides"])
            self.assertEqual(_run(*args, "--msst-batch-size", "2")["model"]["inference_overrides"], {"batch_size": 2})
            with self.assertRaisesRegex(ValueError, "msst-batch-size"): _run(*args, "--msst-batch-size", "0")
            self.assertTrue((artifact / "evidence/manifests/models.json").is_file())
            with self.assertRaisesRegex(ValueError, "declaration"): _run(*[a for i, a in enumerate(args) if args[i - 1] != "--declaration-quote" and a != "--declaration-quote"])
            with self.assertRaisesRegex(ValueError, "needs --model-type"): _run(*[a for i, a in enumerate(args) if args[i - 1] != "--msst-commit" and a != "--msst-commit"])
            with self.assertRaisesRegex(ValueError, "release"): _run(*[("v9" if a == "v1.0.12" else a) for a in args])
            other = base / "viperx.ckpt"; other.write_bytes(b"w")
            with self.assertRaisesRegex(ValueError, "not listed"): _run(*[(str(other) if a == ckpt else a) for a in args])
            excluded = base / "MelBandRoformer.ckpt"; excluded.write_bytes(b"w")
            with self.assertRaisesRegex(ValueError, "excluded"): _run(*[(str(excluded) if a == ckpt else a) for a in args])
            ckpt.write_bytes(b"a different checkpoint with the same name")  # listed file, wrong bytes
            with self.assertRaisesRegex(ValueError, "sha256 .* differs"): _run(*args)
            ckpt.write_bytes(b"weights"); cfg.write_text("training: {instruments: [vocals]}\n")  # listed config, wrong bytes
            with self.assertRaisesRegex(ValueError, "config sha256"): _run(*args)
            self.assertEqual(len(json.loads((runtime / "manifests/models.json").read_text())["models"]), 1)

    def test_smoke_accepts_a_model_flag(self):
        args = cli._parser().parse_args(["smoke", "--runtime-dir", "r", "--artifact-dir", "a", "--config", "c", "--device", "cuda", "--require-cuda", "--model", "bsr", "--seconds", "30"])
        self.assertEqual((args.model, args.seconds), ("bsr", 30))
        self.assertIsNone(cli._parser().parse_args(["smoke", "--runtime-dir", "r", "--artifact-dir", "a", "--config", "c"]).model)
        self.assertEqual(cli._parser().parse_args(["smoke", "--runtime-dir", "r", "--artifact-dir", "a", "--config", "c", "--model", "m", "--msst-batch-size", "2"]).msst_batch_size, 2)


def _real_runtime():
    """The runtime directory holding the pinned MSST checkout and the downloaded checkpoints, or None (the real-code tests then skip)."""
    candidates = [Path(p) for p in (os.environ.get("BANELP_RUNTIME_DIR"),) if p]
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        candidates.append((ROOT / common).resolve().parent / ".agent_runtime" / "banelp-v1")
    except (OSError, subprocess.CalledProcessError): pass
    for runtime in candidates:
        if (runtime / "src/MSST/.git").exists() and (runtime / "cache/msst").is_dir(): return runtime
    return None


REAL_RUNTIME = _real_runtime()
# per registered pair: instruments order, normalize flag and batch size read from the release configs (owner's CPU probe, msst-cpu-probe.jsonl)
REAL_PAIRS = {"model_bs_roformer_ep_17_sdr_9.6568.ckpt": ("bs_roformer", False, 2), "scnet_checkpoint_musdb18.ckpt": ("scnet", True, 8),
              "SCNet-large_starrytong_fixed.ckpt": ("scnet", True, 8), "model_scnet_sdr_9.3244.ckpt": ("scnet", False, 8), "model_scnet_ep_54_sdr_9.8051.ckpt": ("scnet", False, 4)}


def _real_record(name, model_type, runtime=REAL_RUNTIME):
    cache = runtime / "cache/msst"
    return separators.model_record(model_id=name, checkpoints=[cache / name], native_source_order=NATIVE, samplerate=44100, train_data_status="documented-train-only", family="msst", model_type=model_type,
                                   config_file={"path": str(cache / CONFIGS[name]), "sha256": sha256_file(cache / CONFIGS[name]), "url": None},
                                   declaration=separators.declaration_record("https://example.invalid", "q", RELEASES[name]), msst_commit=separators.MSST_PINNED_COMMIT)


@unittest.skipUnless(HAVE_TORCH and REAL_RUNTIME, "torch, the pinned MSST checkout ($R/src/MSST) or the downloaded checkpoints ($R/cache/msst) are absent")
class RealMSSTCpuTests(unittest.TestCase):
    """The adapter against the real MSST code (tag v1.0.23) and the real checkpoints, on CPU only."""
    @classmethod
    def setUpClass(cls):
        cls.api = separators.load_msst_api(REAL_RUNTIME / "src/MSST", separators.MSST_PINNED_COMMIT)
        cls.threads = torch.get_num_threads(); torch.set_num_threads(min(cls.threads, 4))  # a few threads: the forward is slow on a shared CPU, oversubscription makes it slower
        cls.real_sha256 = separators.sha256_file; cache = {}  # the five checkpoints are ~1.1 GB: hash each file once for the whole class
        def cached(path):
            key = (str(path), Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
            if key not in cache: cache[key] = cls.real_sha256(path)
            return cache[key]
        separators.sha256_file = cached; cls.sha = staticmethod(cached)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads); separators.sha256_file = cls.real_sha256

    def test_table_hashes_match_the_downloaded_files(self):
        for e in separators.load_eligibility_table()["allowed"]:
            ckpt, cfg = REAL_RUNTIME / "cache/msst" / e["file"], REAL_RUNTIME / "cache/msst" / e["config"]["file"]
            self.assertEqual((ckpt.stat().st_size, self.sha(cfg)), (e["bytes"], e["config"]["sha256"]), e["file"]); self.assertEqual(self.sha(ckpt), e["sha256"], e["file"])

    def test_all_five_pairs_build_load_and_map_instruments(self):
        stems = {"drums": 1., "bass": 2., "other": 3., "vocals": 4.}
        for name, (model_type, normalize, batch) in REAL_PAIRS.items():
            with self.subTest(name):
                sep = separators.MSSTSeparator(registered=_real_record(name, model_type), runtime_dir=REAL_RUNTIME, device="cpu")
                self.assertEqual(list(sep.config.training.instruments), NATIVE); self.assertEqual(sep.native_sources, tuple(NATIVE))
                self.assertIs(sep.config.training.use_amp, False); self.assertTrue(sep.attestation["config_use_amp"]); self.assertEqual(sep.attestation["normalize"], normalize)
                self.assertEqual((sep.attestation["batch_size"], sep.attestation["msst_commit"]), (batch, separators.MSST_PINNED_COMMIT)); self.assertFalse(sep.model.training)
                self.assertTrue(all(p.dtype == torch.float32 for p in sep.model.parameters()))
                real_demix = sep.api.demix  # instruments order mapping on the real config: each native instrument carries its own constant (a silent input keeps normalization neutral), SOURCES order must come out
                sep.api = types.SimpleNamespace(**{**vars(sep.api), "demix": lambda config, model, mix, device, model_type, pbar=False: {n: np.full_like(mix, stems[n]) for n in config.training.instruments}})
                out = sep(np.zeros((2, 64), np.float32)); self.assertEqual([float(out[i, 0, 0]) for i in range(4)], [stems[n] for n in SOURCES]); sep.api = types.SimpleNamespace(**{**vars(sep.api), "demix": real_demix})
                del sep

    def independent_official_run(self, mix, *, use_amp, normalize_override=None):
        """MSST's own sequence called directly (inference.py:208-216 load, :104-106 / :142-145), independent of the adapter."""
        name = "scnet_checkpoint_musdb18.ckpt"; cache = REAL_RUNTIME / "cache/msst"
        with self.api.sys_path(): model, config = self.api.get_model_from_config("scnet", str(cache / CONFIGS[name]))
        checkpoint = torch.load(str(cache / name), weights_only=False, map_location="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            self.api.load_start_checkpoint(argparse.Namespace(start_check_point=str(cache / name), model_type="scnet", lora_checkpoint_loralib=None), model, checkpoint, type_="inference")
        config.training.use_amp = use_amp
        if normalize_override is not None: config.inference["normalize"] = normalize_override
        model = model.eval()
        calls = []; original = torch.cuda.amp.autocast
        torch.cuda.amp.autocast = lambda *a, **k: (calls.append(k.get("enabled", a[0] if a else True)), original(*a, **k))[1]
        try:
            with contextlib.redirect_stderr(io.StringIO()): out = separators.official_msst_reference(self.api, config, model, mix.copy(), "cpu", model_type="scnet")
        finally: torch.cuda.amp.autocast = original
        return out, calls, config

    def test_scnet_small_adapter_equals_the_official_sequence_end_to_end(self):
        rng = np.random.default_rng(11); mix = (rng.normal(.05, .15, (2, 88200)) + .03 * np.sin(np.arange(88200) / 40)).astype(np.float32)  # 2 s, DC offset, normalize is true for this config
        expected, expected_calls, config = self.independent_official_run(mix, use_amp=False)
        self.assertTrue(config.inference["normalize"] is True); self.assertEqual(expected_calls, [False])
        sep = separators.MSSTSeparator(registered=_real_record("scnet_checkpoint_musdb18.ckpt", "scnet"), runtime_dir=REAL_RUNTIME, device="cpu")
        calls = []; original = torch.cuda.amp.autocast
        torch.cuda.amp.autocast = lambda *a, **k: (calls.append(k.get("enabled", a[0] if a else True)), original(*a, **k))[1]
        try:
            with contextlib.redirect_stderr(io.StringIO()): got = sep(mix)
        finally: torch.cuda.amp.autocast = original
        self.assertEqual(calls, [False])  # the adapter's demix ran with autocast disabled (the test fails if use_amp were left true)
        want = np.stack([expected[n] for n in SOURCES]).astype(np.float32)
        self.assertEqual(got.shape, (4, 2, 88200)); self.assertLessEqual(float(np.abs(got - want).max()), 1e-6)
        self.assertTrue(sep.last_normalization["normalize"] and not sep.last_normalization["centered_only"])

    def test_real_loader_handles_each_key_form_strictly_and_real_normalize_on_silence_is_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = FakeMSSTModel(); reference.scale.data += 3; state = reference.state_dict()
            for form, payload in {"raw": state, "state": {"state": state}, "state_dict": {"state_dict": state}, "model_state_dict": {"model_state_dict": state}}.items():
                path = Path(tmp) / f"{form}.ckpt"; torch.save(payload, path); model = FakeMSSTModel()
                separators.load_msst_checkpoint(self.api, model, path, "scnet"); self.assertTrue(torch.equal(model.scale, reference.scale), form)
            torch.save({k: v for k, v in state.items() if k != "bias"}, Path(tmp) / "bad.ckpt")
            with self.assertRaises(RuntimeError): separators.load_msst_checkpoint(self.api, FakeMSSTModel(), Path(tmp) / "bad.ckpt", "scnet")
        with np.errstate(all="ignore"): normalized, _ = self.api.normalize_audio(np.zeros((2, 50), np.float32))
        self.assertTrue(np.isnan(normalized).all())

    def test_real_demix_batch_size_does_not_change_results_and_autocast_follows_use_amp(self):
        from ml_collections import ConfigDict
        outputs, spy = {}, []
        original = torch.cuda.amp.autocast
        torch.cuda.amp.autocast = lambda *a, **k: (spy.append(k.get("enabled", a[0] if a else True)), original(*a, **k))[1]
        try:
            for batch, use_amp in ((1, False), (4, False), (4, True)):
                cfg = ConfigDict({"audio": {"chunk_size": 4000}, "training": {"instruments": NATIVE, "use_amp": use_amp, "target_instrument": None}, "inference": {"batch_size": batch, "num_overlap": 4}})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore"); out = self.api.demix(cfg, FakeMSSTModel().eval(), np.random.default_rng(2).normal(0, .3, (2, 30000)).astype(np.float32), "cpu", model_type="scnet", pbar=False)
                outputs[(batch, use_amp)] = np.stack([out[n] for n in NATIVE])
        finally: torch.cuda.amp.autocast = original
        self.assertTrue(np.array_equal(outputs[(1, False)], outputs[(4, False)])); self.assertEqual(outputs[(1, False)].shape, (4, 2, 30000))
        self.assertEqual(spy, [False, False, True])  # the pinned demix enables autocast exactly when config.training.use_amp is true (so forcing it False matters)


if __name__ == "__main__":
    unittest.main()
