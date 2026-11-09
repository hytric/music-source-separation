"""P1-1 sample-median statistic and cluster bootstrap, selection gates, and the policy-lock/eligibility units."""
import unittest
import numpy as np
from scipy.stats import wilcoxon
from experiments.banelp import cli, policy
from experiments.banelp.candidates import candidate_registry
from experiments.banelp.contracts import SOURCES
from experiments.banelp.selection import STATISTIC_WORDS, paired_summary, select_candidate

SPEC = {"params": {}, "additional_inference_calls": 0}
TRACKS, ROOMS_N = 10, 4


def _row(name, room_deltas, sar_delta=0.1, base_sar=2.0, **row_kwargs):
    """Candidate row from per-track-per-room deltas; `room_deltas` is (tracks, rooms) and equal across the four stems."""
    room_deltas = np.asarray(room_deltas, dtype=float).reshape(TRACKS, ROOMS_N)
    base_sdr = np.full((TRACKS, 16), 1.0); base_sar_arr = np.full((TRACKS, 16), base_sar)
    sdr = base_sdr + np.repeat(room_deltas, 4, axis=1)
    sar = base_sar_arr + (np.asarray(sar_delta, dtype=float) if np.ndim(sar_delta) else sar_delta)
    return cli._candidate_row(name, SPEC, sdr, sar, base_sdr, base_sar_arr, **row_kwargs)


def _select(*rows): return select_candidate(list(rows), sar_policy_status="confirmed", sar_floor_db=-0.5, minimum_sdr_gain_db=0.2)


class SampleMedianStatisticTests(unittest.TestCase):
    def test_project_verify_counterexample_is_not_selected(self):
        # room deltas [-1,-1,-1,+4] dB on every track, SAR +0.1 dB: sample median -1 dB; the old per-track mean gave +0.25 dB.
        row = _row("A", np.tile([-1, -1, -1, 4.0], (TRACKS, 1)))
        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(row["summary"]["sample_median_delta_db"], -1.0)
        self.assertAlmostEqual(row["sample_median_delta_db"], -1.0)
        self.assertAlmostEqual(row["summary"]["track_median_delta_db"], 0.25)  # non-gating report field only
        self.assertEqual(_select(row)["selected"], "identity")

    def test_median_passes_but_sample_cluster_ci_lower_bound_fails(self):
        values = np.full((TRACKS, ROOMS_N), 0.5); values[6:] = -1.0  # 6 tracks +0.5, 4 tracks -1: median +0.5
        row = _row("B", values)
        self.assertGreaterEqual(row["summary"]["sample_median_delta_db"], 0.2); self.assertLessEqual(row["summary"]["sample_bootstrap_95_ci"][0], 0)
        choice = _select(row); self.assertEqual(choice["selected"], "identity")
        choice = select_candidate([row, _row("C", np.full((TRACKS, ROOMS_N), 0.6))], sar_policy_status="confirmed", sar_floor_db=-0.5)
        self.assertEqual(choice["selected"], "C")
        self.assertEqual(choice["candidate"]["gates"], {"sample_median_gain": True, "sample_ci_lower_gt_0": True, "stem_sar_floor": True})

    def test_one_stem_sar_median_below_floor_blocks_selection(self):
        sar_delta = np.full(16, 0.1); sar_delta[2::4] = -0.6  # bass SAR delta -0.6 dB in every sample
        row = _row("D", np.full((TRACKS, ROOMS_N), 1.0), sar_delta=sar_delta)
        self.assertAlmostEqual(row["stem_median_delta_sar"]["bass"], -0.6); self.assertAlmostEqual(row["stem_median_delta_sar"]["vocals"], 0.1)
        self.assertEqual(_select(row)["selected"], "identity")
        ok = _row("E", np.full((TRACKS, ROOMS_N), 1.0), sar_delta=np.where(np.arange(16) % 4 == 2, -0.5, 0.1))  # exactly at the floor passes
        self.assertEqual(_select(ok)["selected"], "E")

    def test_stem_without_a_defined_sar_sample_fails_the_gate(self):
        base_sdr = np.full((TRACKS, 16), 1.0); base_sar = np.full((TRACKS, 16), 2.0); base_sar[:, 2::4] = np.nan  # bass SAR never defined
        sdr = base_sdr + 1.0; sar = base_sar + 0.1
        row = cli._candidate_row("F", SPEC, sdr, sar, base_sdr, base_sar)
        self.assertEqual(row["status"], "ok"); self.assertIsNone(row["stem_median_delta_sar"]["bass"])
        self.assertEqual(_select(row)["selected"], "identity")

    def test_sar_median_is_over_all_track_room_samples_not_track_means(self):
        # bass SAR delta: -1 in 3 of 4 rooms on every track. Median over the 40 samples is -1 although the per-track mean is +0.25.
        sar_delta = np.zeros((TRACKS, 16)); sar_delta[:, 2::4] = np.tile([-1, -1, -1, 4.0], (TRACKS, 1))
        base_sdr = np.full((TRACKS, 16), 1.0); base_sar = np.full((TRACKS, 16), 2.0)
        row = cli._candidate_row("G", SPEC, base_sdr + 1.0, base_sar + sar_delta, base_sdr, base_sar)
        self.assertAlmostEqual(row["stem_median_delta_sar"]["bass"], -1.0); self.assertEqual(_select(row)["selected"], "identity")

    def test_rank_uses_sample_median_then_calls_then_hash(self):
        a = _row("A", np.full((TRACKS, ROOMS_N), 0.5)); b = _row("B", np.full((TRACKS, ROOMS_N), 0.9)); c = _row("C", np.full((TRACKS, ROOMS_N), 0.9))
        c["additional_inference_calls"] = 1; c["config_hash"] = "0" * 64; b["config_hash"] = "f" * 64
        self.assertEqual(_select(a, b, c)["selected"], "B")  # equal gain: fewer extra calls beats a smaller hash
        c["additional_inference_calls"] = 0
        self.assertEqual(_select(a, b, c)["selected"], "C")  # then the hash decides

    def test_paired_summary_definitions(self):
        rng = np.random.default_rng(3); samples = rng.normal(0.3, 1.0, (TRACKS, ROOMS_N))
        summary = paired_summary(samples)
        self.assertAlmostEqual(summary["sample_median_delta_db"], float(np.median(samples)))
        per_track = samples.mean(axis=1)
        self.assertAlmostEqual(summary["track_median_delta_db"], float(np.median(per_track)))
        self.assertAlmostEqual(summary["wilcoxon_p"], float(wilcoxon(per_track, zero_method="wilcox", alternative="two-sided", method="approx", correction=False).pvalue))
        # the CI resamples whole tracks and recomputes the sample median (not the median of track means)
        rng = np.random.default_rng(20261108); idx = rng.integers(0, TRACKS, (10000, TRACKS))
        boots = np.median(samples[idx].reshape(10000, -1), axis=1)
        self.assertEqual(summary["sample_bootstrap_95_ci"], [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))])
        self.assertEqual(paired_summary(samples), summary)  # seeded
        self.assertEqual(paired_summary(np.zeros((TRACKS, ROOMS_N)))["wilcoxon_p"], 1.0)
        self.assertEqual(summary["samples"], 40); self.assertEqual(summary["statistic_words"], STATISTIC_WORDS)

    def test_dropped_samples_stay_inside_the_cluster_resample(self):
        samples = np.full((TRACKS, ROOMS_N), 1.0); samples[0, :3] = np.nan
        summary = paired_summary(samples)
        self.assertEqual(summary["samples"], 37); self.assertEqual(summary["sample_median_delta_db"], 1.0); self.assertEqual(summary["sample_bootstrap_95_ci"], [1.0, 1.0])
        samples[0, :] = np.nan
        with self.assertRaises(ValueError): paired_summary(samples)

    def test_statistic_words_name_estimand_and_resampling_unit(self):
        self.assertIn("median over dev track x room samples of the 4-stem mean dSI-SDR", STATISTIC_WORDS["effect_estimand"])
        self.assertIn("95% percentile track-cluster bootstrap of the same statistic, B=10000, seed=20261108", STATISTIC_WORDS["ci"])
        self.assertIn("per-stem median over dev track x room samples of dSI-SAR", STATISTIC_WORDS["sar_estimand"])

    def test_row_detail_keeps_three_metrics_improvement_and_undefined_reasons(self):
        base_sdr, base_sir, base_sar = (np.full((TRACKS, 16), v) for v in (1.0, 2.0, 3.0))
        base_sdr[0, 0] = base_sir[0, 0] = base_sar[0, 0] = np.nan
        states = lambda value, nan_state: {m: np.where(np.isnan(value), nan_state, "finite").astype(object) for m in ("si_sdr", "si_sir", "si_sar")}
        sdr, sir, sar = base_sdr + 1, base_sir - 0.5, base_sar + 0.25
        row = cli._candidate_row("H", SPEC, sdr, sar, base_sdr, base_sar, sir=sir, base_sir=base_sir,
                                 states={"si_sdr": states(sdr, "silent-target")["si_sdr"], "si_sir": states(sir, "silent-target")["si_sir"], "si_sar": states(sar, "silent-target")["si_sar"]},
                                 base_states=states(base_sdr, "silent-target"))
        detail = row["delta_detail"]
        self.assertAlmostEqual(detail["overall_sample_median"]["si_sdr"], 1.0); self.assertAlmostEqual(detail["overall_sample_median"]["si_sir"], -0.5)
        self.assertEqual(set(detail["stem_median"]), {"si_sdr", "si_sir", "si_sar"}); self.assertEqual(set(detail["stem_median"]["si_sir"]), set(SOURCES))
        self.assertEqual(detail["improvement_ratio"]["si_sdr"], {"improved": 159, "defined": 159, "ratio": 1.0}); self.assertEqual(detail["improvement_ratio"]["si_sir"]["improved"], 0)
        self.assertEqual(detail["undefined_by_reason"]["si_sdr"]["baseline"], {"silent-target": 1})


class PolicyUnitTests(unittest.TestCase):
    ALLOW = {"baseline_train_data_policy": "allow-unknown-with-disclosure"}

    def model(self, model_id, status, sha=None, evidence=("doc",), **extra):
        sha = sha or (model_id[0] * 64)
        flags = {"clean_winner_eligible": status == "verified-excluded" and bool(evidence)}
        return {"model_id": model_id, "sha256": sha, "train_data_status": status, "training_data_evidence": list(evidence) if status == "verified-excluded" else [], "members": [], **flags, **extra}

    def config(self, **selection):
        return {"selection": {"sar_policy_status": "confirmed", "sar_floor_db": -0.5, "minimum_sdr_gain_db": .2, **selection}}

    def build(self, models, **selection):
        config = self.config(**selection); registry = candidate_registry(model_candidates=config["selection"].get("model_candidates", []), baseline_id="base")
        return policy.build_lock_content(config, registry, {m["model_id"]: m for m in models}, "base")

    def test_model_candidates_need_every_depended_model_eligible(self):
        base = self.model("base", "verified-excluded"); good = self.model("good", "verified-excluded"); unknown = self.model("unk", "unknown", evidence=())
        included = self.model("inc", "verified-included"); documented = self.model("doc", "documented-train-only", declaration={"url": "u", "quote": "q", "release": "v1"})
        lock = self.build([base, good], model_candidates=["replace:good"]); self.assertIn("replace:good", [c["name"] for c in lock["candidates"]])
        self.assertEqual([m["model_id"] for m in lock["models"]], ["base", "good"])
        lock = self.build([base, good, documented], model_candidates=["average:doc+good"])
        self.assertEqual({m["model_id"]: m["route"] for m in lock["models"]}, {"base": "verified-excluded", "doc": "documented-train-only", "good": "verified-excluded"})
        label = [m for m in lock["models"] if m["model_id"] == "doc"][0]["label_ko"]; self.assertEqual(label, "문서상 train-only, 곡별 감사 미확인")
        for bad in (unknown, included):
            with self.assertRaisesRegex(ValueError, "not eligible|verified-included"): self.build([base, good, bad], model_candidates=[f"replace:{bad['model_id']}"])
        with self.assertRaisesRegex(ValueError, "not registered"): self.build([base], model_candidates=["replace:ghost"])
        undeclared = self.model("und", "documented-train-only")
        with self.assertRaisesRegex(ValueError, "not eligible"): self.build([base, undeclared], model_candidates=["replace:und"])

    def test_unverified_baseline_needs_decision_for_that_exact_model(self):
        base = self.model("base", "unknown", evidence=()); decision = {"model_id": "base", "sha256": base["sha256"], "decision": "use-with-disclosure", "decided_by": "user", "date": "2026-11-09"}
        clean = self.model("clean", "verified-excluded")
        for kwargs in ({}, {"baseline_decisions": []}, {"baseline_decisions": [{**decision, "sha256": "e" * 64}]}, {"baseline_decisions": [{**decision, "model_id": "clean"}]}):
            with self.assertRaisesRegex(ValueError, "decision|baseline_train_data_policy"): self.build([base, clean], **self.ALLOW, **kwargs)
        lock = self.build([base, clean], **self.ALLOW, baseline_decisions=[decision])
        self.assertEqual(lock["baseline"]["eligibility"]["route"], "user-decision"); self.assertEqual(lock["baseline"]["eligibility"]["decision"], decision)
        with self.assertRaisesRegex(ValueError, "baseline_train_data_policy"): self.build([base, clean], baseline_decisions=[decision])
        with self.assertRaisesRegex(ValueError, "verified-included"): self.build([self.model("base", "verified-included")], **self.ALLOW, baseline_decisions=[decision])

    def test_lock_is_independent_of_unrelated_models_but_not_of_dependencies(self):
        base = self.model("base", "verified-excluded"); other = self.model("other", "verified-excluded")
        self.assertEqual(self.build([base]), self.build([base, other]))
        self.assertNotEqual(self.build([base]), self.build([self.model("base", "verified-excluded", sha="9" * 64)]))

    def test_pipeline_dependencies_are_baseline_plus_selected_candidate_models_only(self):
        models = {m["model_id"]: m for m in (self.model("base", "verified-excluded"), self.model("a", "verified-excluded"), self.model("b", "unknown", evidence=()))}
        self.assertEqual([d["model_id"] for d in policy.pipeline_dependencies(models, "base", [], {})], ["base"])
        self.assertEqual([d["model_id"] for d in policy.pipeline_dependencies(models, "base", ["a", "base"], {})], ["a", "base"])
        with self.assertRaisesRegex(ValueError, "not eligible"): policy.pipeline_dependencies(models, "base", ["b"], {})

    def test_lock_requires_confirmed_numeric_policy(self):
        registry = candidate_registry(baseline_id="base"); models = {"base": self.model("base", "verified-excluded")}
        with self.assertRaisesRegex(ValueError, "confirmed"): policy.build_lock_content(self.config(sar_policy_status="pending-owner-confirmation"), registry, models, "base")
        with self.assertRaisesRegex(ValueError, "numeric"): policy.build_lock_content(self.config(sar_floor_db=None), registry, models, "base")


if __name__ == "__main__":
    unittest.main()
