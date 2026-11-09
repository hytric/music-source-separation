"""`cli rirs --gpurir` end to end with an injected fake simulator, then a wet-cache build from the new manifest."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import soundfile as sf
from experiments.banelp import cli, provenance, rir
from experiments.banelp.contracts import SOURCES
from test_banelp_rir import FakeGpuRIR

ROOT = Path(__file__).parents[1]
GEOMETRY = ROOT / "experiments/banelp/rir_geometry.json"
SAMPLES = 8192


def _run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out): code = cli.main([str(a) for a in argv])
    assert code == 0
    return json.loads(out.getvalue())


class GpuRirCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); base = Path(self.tmp.name).resolve()
        self.runtime, self.artifact = base / "runtime", base / "artifacts"
        config = json.loads((ROOT / "experiments/banelp/config.json").read_text()); config["samples"] = SAMPLES
        self.config = base / "config.json"; self.config.write_text(json.dumps(config))
        self.common = ["--runtime-dir", self.runtime, "--artifact-dir", self.artifact, "--config", self.config]
        self.record = base / "build-record.json"; self.record.write_text(json.dumps({"gpurir_commit": "0" * 40, "build_path": "fake"}))
        patcher = mock.patch.object(rir, "load_gpurir", return_value=(FakeGpuRIR(), {"fake": True})); patcher.start(); self.addCleanup(patcher.stop)

    def test_rirs_then_wet_cache_from_gpurir_manifest(self):
        rng = np.random.default_rng(5); tracks = self.runtime / "data/test"
        for i in range(50):
            for stem in SOURCES:
                path = tracks / f"track-{i:02d}" / f"{stem}.wav"; path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, rng.normal(0, .1, (SAMPLES + 500, 2)), 44100, subtype="FLOAT")
        _run("prepare", *self.common, "--track-root", tracks)
        result = _run("rirs", *self.common, "--gpurir", GEOMETRY, "--gpurir-build-record", self.record)
        self.assertEqual(set(result["rooms"].values()), {"gpurir"}); self.assertEqual(result["targets"], rir.TARGET_RT60)
        self.assertEqual(set(result["measured"]), set(rir.TARGET_RT60)); self.assertEqual(set(result["measured"]["R1"]["P1"]), {"edt", "t20", "t30"})
        manifest = json.loads((self.runtime / "manifests/rirs.json").read_text())
        self.assertEqual(manifest["gpurir"]["gpurir_commit"], "0" * 40); self.assertTrue((self.artifact / "evidence/manifests/rirs.json").is_file())
        summary = _run("build-wet", *self.common, "--limit", 1)
        self.assertEqual(summary["written"], 4)
        self.assertEqual(summary["wet_cache_key"], cli._keys(provenance.current_binding(self.runtime, self.config, require=("dataset", "rirs")), json.loads(self.config.read_text()), None)[0])

    def test_incompatible_arguments_are_refused(self):
        with self.assertRaises(ValueError): _run("rirs", *self.common, "--gpurir", GEOMETRY, "--offline-synthetic", "--gpurir-build-record", self.record)
        with self.assertRaises(ValueError): _run("rirs", *self.common, "--gpurir", GEOMETRY, "--synthetic-fallback", "--gpurir-build-record", self.record)
        with self.assertRaises(ValueError): _run("rirs", *self.common, "--gpurir", GEOMETRY)
        with self.assertRaises(ValueError): _run("rirs", *self.common, "--offline-synthetic", "--gpurir-build-record", self.record)


if __name__ == "__main__":
    unittest.main()
