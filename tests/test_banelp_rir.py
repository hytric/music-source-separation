"""CPU tests for the gpuRIR generation path (injected fake simulator) and the noise-compensated RT estimator."""
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.banelp import rir
from experiments.banelp.data import RT_METHOD, decay_times, prepare_rir

ROOT = Path(__file__).parents[1]
GEOMETRY = ROOT / "experiments/banelp/rir_geometry.json"
FS = 44100
SIMULATE_PARAMS = ["room_sz", "beta", "pos_src", "pos_rcv", "nb_img", "Tmax", "fs", "Tdiff", "spkr_pattern", "mic_pattern", "orV_src", "orV_rcv", "c"]


class FakeGpuRIR:
    """Same names and signatures as gpuRIR 1.2 (checked against its source at 06da27dd); records every call."""
    def __init__(self, output=None):
        self.calls, self.mixed, self.lut, self.output = [], None, None, output

    def activateMixedPrecision(self, activate=True): self.mixed = activate
    def activateLUT(self, activate=True): self.lut = activate

    def beta_SabineEstimation(self, room_sz, T60, abs_weights=[1.0] * 6):
        self.calls.append(("beta", list(room_sz), T60, list(abs_weights)))
        return np.full(6, 0.8, dtype=np.float32)

    def att2t_SabineEstimator(self, att_dB, T60):
        self.calls.append(("att2t", att_dB, T60)); return att_dB / 60.0 * T60

    def t2n(self, T, rooms_sz, c=343.0):
        return [int(n) for n in np.ceil(2 * T / (np.array(rooms_sz) / c))]

    def simulateRIR(self, room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=None, spkr_pattern="omni", mic_pattern="omni",
                    orV_src=None, orV_rcv=None, c=343.0):
        self.calls.append(("simulate", dict(src=np.asarray(pos_src), rcv=np.asarray(pos_rcv), Tmax=Tmax, fs=fs, Tdiff=Tdiff,
                                            spkr_pattern=spkr_pattern, mic_pattern=mic_pattern, c=c)))
        if self.output is not None: return self.output
        n = int(np.ceil(Tmax * fs)); t = np.arange(n) / fs
        rng = np.random.default_rng(int(1000 * float(np.sum(pos_src))))
        out = 0.1 * rng.standard_normal(n) * np.exp(-3 * np.log(10) * t / Tmax)  # Tmax is T60 here (att_max = 60 dB)
        delay = int(round(float(np.linalg.norm(np.asarray(pos_src) - np.asarray(pos_rcv))) / c * fs)); out[:delay] = 0; out[delay] = 1.0
        return out[None, None, :].astype(np.float32)


def _old_t30(rir_, fs=FS):
    """The estimator this replaces: whole-file Schroeder curve, first -5/-35 dB crossings."""
    decay = np.cumsum((rir_ ** 2)[::-1])[::-1]; edc = 10 * np.log10(np.maximum(decay / decay[0], 1e-30))
    i5, i35 = np.flatnonzero(edc <= -5), np.flatnonzero(edc <= -35)
    return float(2 * (i35[0] - i5[0]) / fs) if i5.size and i35.size and i35[0] > i5[0] else None


def _noisy_decay(t60, seconds, floor_db, seed=20261109):
    rng = np.random.default_rng(seed); n = int(seconds * FS); t = np.arange(n) / FS
    decay = rng.standard_normal(n) * np.exp(-3 * np.log(10) * t / t60); decay[0] = 0.0
    floor = rng.standard_normal(n) * np.sqrt(np.mean(decay[:441] ** 2)) * 10 ** (floor_db / 20)
    return decay + floor


class GpuRirGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.runtime = Path(self.tmp.name).resolve()
        self.geometry = rir.load_geometry(GEOMETRY); self.fake = FakeGpuRIR()

    def generate(self, **kw):
        return rir.acquire_rirs(self.runtime, sources=None, synthetic_fallback=False, gpurir_geometry=self.geometry, simulator=self.fake, **kw)

    def test_fake_matches_gpurir_signatures(self):
        self.assertEqual(list(inspect.signature(self.fake.simulateRIR).parameters), SIMULATE_PARAMS)
        self.assertEqual(list(inspect.signature(self.fake.beta_SabineEstimation).parameters), ["room_sz", "T60", "abs_weights"])

    def test_sixteen_rirs_manifest_and_round_trip(self):
        manifest = self.generate()
        self.assertEqual(len(list((self.runtime / "rir").glob("R*/P*.npy"))), 16)
        self.assertEqual({r["kind"] for r in manifest["rooms"].values()}, {"gpurir"})
        self.assertEqual(manifest["generator"], "gpurir"); self.assertEqual(manifest["geometry"]["sha256"], self.geometry["sha256"])
        self.assertEqual(manifest["t60_input_policy"], rir.T60_INPUT_POLICY); self.assertEqual(manifest["schema_version"], 2)
        loaded = rir.load_rirs(self.runtime, manifest)
        for room, record in manifest["rooms"].items():
            self.assertEqual(record["simulation"]["T60_target_s"], rir.TARGET_RT60[room]); self.assertEqual(record["simulation"]["att_diff_db"], 25.0)
            for position, entry in record["positions"].items():
                self.assertTrue(np.isfinite(loaded[room][position]).all()); self.assertEqual(entry["length_samples"], len(loaded[room][position]))
                self.assertEqual(entry["peak_index"], entry["direct_delay_samples"]); self.assertEqual(entry["rt_method"], RT_METHOD)
                for key in ("measured_edt_seconds", "measured_t20_seconds", "measured_t30_seconds", "noise_floor_db", "distance_m"): self.assertIn(key, entry)

    def test_simulator_receives_reference_settings(self):
        self.generate()
        self.assertIs(self.fake.mixed, False); self.assertIs(self.fake.lut, True)
        att = [c for c in self.fake.calls if c[0] == "att2t"]; self.assertEqual({c[1] for c in att}, {25.0, 60.0})
        for call in (c for c in self.fake.calls if c[0] == "beta"):
            self.assertEqual(call[1], [7.5, 6.0, 3.0]); self.assertEqual(call[3], [.9] * 5 + [.5])
        sims = [c[1] for c in self.fake.calls if c[0] == "simulate"]; self.assertEqual(len(sims), 16)
        for call in sims:
            self.assertEqual((call["src"].shape, call["rcv"].shape), ((1, 3), (1, 3))); self.assertEqual(call["mic_pattern"], "omni")
            self.assertEqual((call["spkr_pattern"], call["c"], call["fs"]), ("omni", 343.0, FS)); self.assertLess(call["Tdiff"], call["Tmax"])
        self.assertTrue(np.allclose(sims[0]["rcv"], [[3.73, .87, 1.5]])); self.assertTrue(np.allclose(sims[0]["src"], [[3.73, 2.87, 1.5]]))

    def test_measured_rt_tracks_target_on_fake_decay(self):
        manifest = self.generate()
        for room, record in manifest["rooms"].items():
            for entry in record["positions"].values():
                for key in ("t20", "t30"):
                    value = entry[f"measured_{key}_seconds"]
                    self.assertIsNotNone(value, (room, key, entry[f"{key}_failure"])); self.assertAlmostEqual(value / rir.TARGET_RT60[room], 1.0, delta=.08)

    def test_geometry_rejections(self):
        base = json.loads(GEOMETRY.read_text())
        def broken(edit):
            spec = copy.deepcopy(base); edit(spec); path = self.runtime / "g.json"; path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError): rir.load_geometry(path)
        broken(lambda s: s["t60_s"].update(R2=0.9))
        broken(lambda s: s["positions"]["P1"].update(pos_m=[3.73, 2.87, 3.0]))
        broken(lambda s: s["receiver"].update(pos_m=[0.0, .87, 1.5]))
        broken(lambda s: (s["positions"]["P1"].update(stem="drums"), s["positions"]["P2"].update(stem="vocals")))
        broken(lambda s: s["simulation"].update(att_diff_db=15.0))
        broken(lambda s: s["simulation"].update(abs_weights=[1.0] * 6))
        broken(lambda s: s.update(fs=16000))
        broken(lambda s: s["receiver"].update(count=2))

    def test_path_mixing_and_missing_build_record(self):
        with self.assertRaises(ValueError): rir.acquire_rirs(self.runtime, sources={"rooms": []}, synthetic_fallback=False, gpurir_geometry=self.geometry, simulator=self.fake)
        with self.assertRaises(ValueError): rir.acquire_rirs(self.runtime, sources=None, synthetic_fallback=True, gpurir_geometry=self.geometry, simulator=self.fake)
        with self.assertRaises(ValueError): rir.acquire_rirs(self.runtime, sources=None, synthetic_fallback=False, gpurir_geometry=self.geometry)

    def test_bad_simulator_output_is_refused(self):
        for output in (np.full((1, 1, 100), np.nan), np.zeros((1, 1, 100)), np.ones((1, 100)), np.ones((1, 1, 0))):
            self.fake = FakeGpuRIR(output=output)
            with self.assertRaises(RuntimeError): self.generate()

    def test_incomplete_api_is_refused(self):
        class Partial: simulateRIR = staticmethod(lambda *a, **k: None)
        with self.assertRaises(RuntimeError): rir.load_gpurir(Partial())


class DecayTimesTests(unittest.TestCase):
    def test_minus_50_db_floor_defeats_old_estimator_not_new(self):
        for t60, seconds in ((0.5, 3.0), (0.26, 2.0)):
            x = _noisy_decay(t60, seconds, -50.0); old = _old_t30(x); new = decay_times(x)
            self.assertGreaterEqual(old / t60, 1.5, (t60, old))
            self.assertLess(abs(new["measured_t30_seconds"] / t60 - 1), .05, (t60, new["measured_t30_seconds"]))
            self.assertLess(abs(new["measured_t20_seconds"] / t60 - 1), .05); self.assertLess(abs(new["measured_edt_seconds"] / t60 - 1), .10)
            self.assertEqual(new["compensation"], "extrapolated-tail"); self.assertEqual(new["rt_method"], RT_METHOD)

    def test_noise_free_decay(self):
        rng = np.random.default_rng(1); n = int(1.5 * 1.96 * FS); t = np.arange(n) / FS
        new = decay_times(rng.standard_normal(n) * np.exp(-3 * np.log(10) * t / 1.96))
        self.assertLess(abs(new["measured_t30_seconds"] / 1.96 - 1), .02)

    def test_degenerate_inputs_give_none_with_reason(self):
        for x in (np.r_[1., np.zeros(100)], np.zeros(5000)):
            r = decay_times(x)
            for key in ("edt", "t20", "t30"): self.assertIsNone(r[f"measured_{key}_seconds"]); self.assertEqual(r[f"{key}_failure"], "rir too short or silent")
        self.assertIsNone(prepare_rir(np.r_[1., np.zeros(100)], FS)[1]["rt"]["measured_t30_seconds"])
        with self.assertRaises(ValueError): decay_times(np.array([np.nan, 1.]))

    def test_shallow_dynamic_range_is_not_guessed(self):
        r = decay_times(_noisy_decay(0.5, 3.0, -30.0))
        self.assertIsNone(r["measured_t30_seconds"]); self.assertEqual(r["t30_failure"], "insufficient dynamic range above noise floor")
        self.assertIsNone(r["measured_t20_seconds"])


if __name__ == "__main__":
    unittest.main()
