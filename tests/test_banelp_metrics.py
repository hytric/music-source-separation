"""Independent expected values for the projection metrics, and the metric/STFT config contract (P2-3, P2-4)."""
import copy
import unittest
import numpy as np
from experiments.banelp import candidates, cli
from experiments.banelp.contracts import DEFAULT_CONFIG, STFT_CONTRACT, validate_practical_config
from experiments.banelp.metrics import project_metrics

DIM = 8


def basis(*indexes):
    """Unit vectors e_i of R^8 reshaped to (2, 4) stereo-like targets."""
    out = np.zeros((len(indexes), DIM))
    for row, i in enumerate(indexes): out[row, i] = 1.0
    return out.reshape(len(indexes), 2, 4)


def vec(**coef):
    z = np.zeros(DIM)
    for key, value in coef.items(): z[int(key[1:])] = value
    return z.reshape(2, 4)


def db(x): return float(10 * np.log10(x))


class IndependentExpectedValueTests(unittest.TestCase):
    def test_orthogonal_basis_leakage(self):
        targets = basis(0, 1, 2, 3); z = vec(e0=2, e1=3, e4=4)
        m = project_metrics(targets[0], z, targets)
        self.assertAlmostEqual(m["target_gain"], 2.0)
        self.assertAlmostEqual(m["si_sdr"]["value_db"], db(4 / 25)); self.assertAlmostEqual(m["si_sir"]["value_db"], db(4 / 9)); self.assertAlmostEqual(m["si_sar"]["value_db"], db(13 / 16))
        self.assertEqual((m["rank"], m["state"]), (4, "ok"))

    def test_signed_gain_is_reported_and_does_not_hide_a_sign_flip(self):
        targets = basis(0, 1, 2, 3); z = vec(e0=-2, e4=1)
        m = project_metrics(targets[0], z, targets)
        self.assertAlmostEqual(m["target_gain"], -2.0)
        self.assertAlmostEqual(m["si_sdr"]["value_db"], db(4 / 1))  # scale invariant: the sign is only visible through the gain
        self.assertAlmostEqual(m["sd_sdr"]["value_db"], db(4 / (9 + 1)))  # |z - s|^2 = (-3)^2 + 1

    def test_rank_deficient_targets(self):
        targets = np.concatenate([basis(0, 1), basis(0)[0:1] + basis(1)[0:1], basis(2)])  # e0, e1, e0+e1, e2: rank 3
        m = project_metrics(targets[0], vec(e0=2, e1=3, e3=4), targets)
        self.assertEqual(m["rank"], 3)
        self.assertAlmostEqual(m["si_sdr"]["value_db"], db(4 / 25)); self.assertAlmostEqual(m["si_sir"]["value_db"], db(4 / 9)); self.assertAlmostEqual(m["si_sar"]["value_db"], db(13 / 16))

    def test_perfect_estimate_is_positive_infinity_not_a_capped_number(self):
        targets = basis(0, 1, 2, 3); m = project_metrics(targets[0], targets[0], targets)
        for key in ("si_sdr", "si_sir"): self.assertEqual((m[key]["state"], m[key]["value_db"]), ("positive-infinity", None))
        self.assertEqual(project_metrics(targets[0], 0 * targets[0], targets)["si_sdr"]["state"], "undefined-zero-over-zero")


class ConfiguredMetricTests(unittest.TestCase):
    def setUp(self):
        self.targets = basis(0, 1, 2, 3); self.z = vec(e0=2, e1=3, e4=4)

    def test_normal_inputs_are_unchanged_by_the_defaults(self):
        rng = np.random.default_rng(4); targets = rng.normal(size=(4, 2, 512)); z = targets[0] * .7 + .1 * rng.normal(size=(2, 512))
        a = project_metrics(targets[0], z, targets); b = project_metrics(targets[0], z, targets, rcond=1e-10, energy_epsilon=1e-12, db_cap=120.0)
        self.assertEqual(a, b)
        for key in ("si_sdr", "si_sir", "si_sar"): self.assertEqual(a[key]["state"], "finite"); self.assertFalse(a[key]["energy_floored"])

    def test_db_cap_is_applied(self):
        z = self.targets[0] + 1e-5 * vec(e4=1.0)  # SI-SDR ~ 100 dB
        self.assertGreater(project_metrics(self.targets[0], z, self.targets)["si_sdr"]["value_db"], 90)
        capped = project_metrics(self.targets[0], z, self.targets, db_cap=60.0)["si_sdr"]
        self.assertEqual((capped["value_db"], capped["state"]), (60.0, "capped"))

    def test_relative_energy_floor_turns_a_vanishing_residual_into_infinity(self):
        z = self.targets[0] + 1e-9 * vec(e4=1.0)  # residual energy 1e-18 of the target: below the 1e-12 relative floor
        default = project_metrics(self.targets[0], z, self.targets)["si_sdr"]
        self.assertEqual((default["state"], default["value_db"]), ("positive-infinity", None))
        loose = project_metrics(self.targets[0], z, self.targets, energy_epsilon=1e-20)["si_sdr"]  # a stricter floor keeps the finite (capped) number
        self.assertEqual((loose["state"], loose["value_db"]), ("capped", 120.0))

    def test_numerator_below_floor_is_displayed_at_the_floor_and_flagged(self):
        z = vec(e4=1.0)  # orthogonal to the target: gain 0, numerator 0
        m = project_metrics(self.targets[0], z, self.targets)["si_sdr"]
        self.assertTrue(m["energy_floored"]); self.assertAlmostEqual(m["value_db"], -120.0)  # floor 1e-12 x max(|s|^2,|z|^2) over |z|^2
        self.assertEqual(project_metrics(self.targets[0], z, self.targets, db_cap=50.0)["si_sdr"]["value_db"], -50.0)

    def test_low_energy_signals_are_scale_invariant(self):
        base = project_metrics(self.targets[0], self.z, self.targets)
        for scale in (1e-3, 1e-6):
            m = project_metrics(scale * self.targets[0], scale * self.z, scale * self.targets)
            for key in ("si_sdr", "si_sir", "si_sar"): self.assertAlmostEqual(m[key]["value_db"], base[key]["value_db"], places=6)

    def test_pinv_rcond_changes_the_reported_rank_and_projection(self):
        near = np.concatenate([basis(0, 1, 2), basis(0)[0:1] + 1e-6 * basis(3)[0:1]])  # e3 direction only at 1e-6 relative strength
        z = vec(e0=2, e3=4, e5=1)
        tight = project_metrics(near[0], z, near, rcond=1e-10); loose = project_metrics(near[0], z, near, rcond=1e-3)
        self.assertEqual((tight["rank"], loose["rank"]), (4, 3))
        self.assertNotAlmostEqual(tight["si_sar"]["value_db"], loose["si_sar"]["value_db"], places=3)

    def test_stem_metrics_passes_the_configured_values_to_the_computation(self):
        estimates = np.stack([self.targets[0] + 1e-9 * vec(e4=1.0), *self.targets[1:]]).astype(np.float64)
        default = cli._stem_metrics(estimates, self.targets, cli._metric_kwargs(DEFAULT_CONFIG))
        self.assertTrue(np.isnan(default["values"]["si_sdr"][0])); self.assertEqual(default["states"]["si_sdr"][0], "positive-infinity")
        config = copy.deepcopy(DEFAULT_CONFIG); config["metrics"]["energy_epsilon"] = 1e-20; config["metrics"]["db_cap"] = 80
        tuned = cli._stem_metrics(estimates, self.targets, cli._metric_kwargs(config))
        self.assertEqual(tuned["values"]["si_sdr"][0], 80.0); self.assertEqual(tuned["states"]["si_sdr"][0], "capped")


class ConfigContractTests(unittest.TestCase):
    def test_default_validates_and_matches_the_candidate_constants(self):
        validate_practical_config(DEFAULT_CONFIG)
        self.assertEqual((candidates.NFFT, candidates.HOP), (STFT_CONTRACT["nperseg"], STFT_CONTRACT["hop"]))

    def test_stft_block_must_equal_the_implemented_constants(self):
        for edit in ({"nperseg": 2048}, {"hop": 512}, {"window": "hann-symmetric"}, {"boundary": None}, {"padded": False}, {"scaling": "psd"}):
            config = copy.deepcopy(DEFAULT_CONFIG); config["stft"].update(edit)
            with self.assertRaisesRegex(ValueError, "stft"): validate_practical_config(config)
        config = copy.deepcopy(DEFAULT_CONFIG); del config["stft"]["scaling"]
        with self.assertRaisesRegex(ValueError, "stft"): validate_practical_config(config)

    def test_metrics_block_rejections(self):
        for key, value in (("center", True), ("pinv_rcond", 0), ("pinv_rcond", 2), ("energy_epsilon", -1), ("energy_epsilon", 1.5), ("db_cap", 0), ("db_cap", float("nan")), ("db_cap", "120")):
            config = copy.deepcopy(DEFAULT_CONFIG); config["metrics"][key] = value
            with self.assertRaises(ValueError, msg=(key, value)): validate_practical_config(config)
        config = copy.deepcopy(DEFAULT_CONFIG); config["metrics"]["extra"] = 1
        with self.assertRaises(ValueError): validate_practical_config(config)
        config = copy.deepcopy(DEFAULT_CONFIG); config["metrics"].update(pinv_rcond=1e-8, energy_epsilon=1e-10, db_cap=100)
        validate_practical_config(config)  # supported numeric options are accepted (and reach the computation)


if __name__ == "__main__":
    unittest.main()
