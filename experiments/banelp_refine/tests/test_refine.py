"""Scientific contracts and the boundary that preparation must not train."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from banelp_refine.common import atomic_json, digest, file_entry, load_config, read_json, verify_entry
from banelp_refine.prepare import song_split, train_members
from banelp_refine.cache import plan_records
from banelp_refine.cache import CachedCrops
from banelp_refine.contracts import validate_execution_config
from banelp_refine.engine import preflight, state_hash, train, verify_ready
from banelp_refine.evaluate import cluster_summary
from banelp_refine.losses import ReconstructionLoss
from banelp_refine.model import JointRefiner, infer_record, istft, project_mixture, stft

CONFIG = load_config(Path(__file__).parents[1] / "configs/pilot.json")


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_song_split_is_order_independent_and_disjoint(self):
        names = [f"track-{i}" for i in range(100)]
        a = song_split(names, CONFIG["data"]["split_counts"], "test")
        self.assertEqual(a, song_split(names[::-1], CONFIG["data"]["split_counts"], "test"))
        self.assertEqual({s: list(a.values()).count(s) for s in set(a.values())}, {"train": 80, "validation": 10, "confirmation": 10})

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "bad.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("train/../escape.wav", b"not audio")
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):
                    train_members(z)

    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "x"
            p.write_bytes(b"original")
            e = file_entry(p, directory)
            p.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify_entry(directory, e)

    def test_unsupported_settings_are_not_silently_ignored(self):
        validate_execution_config(CONFIG)
        for section, key, value in [("model", "center", False), ("baseline", "shifts", 1), ("data", "dry_conditions", 2)]:
            c = copy.deepcopy(CONFIG)
            c[section][key] = value
            with self.assertRaises(ValueError):
                validate_execution_config(c)

    def test_short_song_training_crops_stay_in_its_real_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, left, right = 1323000, 8 * 44100, 21 * 44100
            y = np.zeros((4, 2, frames), dtype=np.float32)
            y[..., left:right] = 1
            files = {}
            for name, value in (("targets", y), ("estimates", y), ("mixture", y.sum(axis=0))):
                p = root / f"{name}.npy"
                np.save(p, value)
                files[name] = file_entry(p, root)
            atomic_json(root / "manifests/cache.json", {"records": [{"split": "train", "id": "short", "files": files,
                        "active_start_frame": left, "active_end_frame": right}]})
            dataset = CachedCrops(root, CONFIG, "train")
            for step in range(8):
                sample, info = dataset.load(0, step=step)
                self.assertGreaterEqual(info["crop_start_frame"], left)
                self.assertLessEqual(info["crop_start_frame"] + info["crop_frames"], right)
                self.assertEqual(float(sample["targets"].mean()), 1.0)

    def test_plan_keeps_rirs_with_their_song_split(self):
        dataset = {"tracks": [{"name": s, "split": s, "pad_left_frames": 0, "crop_frames": 1323000} for s in ("train", "validation", "confirmation")]}
        rirs = {"sets": [{"id": f"{s}-{i}", "split": s} for s in ("train", "validation", "confirmation") for i in range(4)]}
        rows = plan_records(dataset, rirs, CONFIG)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertTrue(row["rir_set"] is None or row["rir_set"].startswith(row["split"]))

    def test_stft_roundtrip_preserves_odd_length_and_channels(self):
        x = torch.randn(2, 4, 2, 12345)
        reconstructed = istft(stft(x, 2048, 512), 2048, 512, x.shape[-1])
        self.assertEqual(x.shape, reconstructed.shape)
        torch.testing.assert_close(x, reconstructed, atol=1e-6, rtol=1e-6)

    def test_model_parameter_count_and_label_free_interface(self):
        m = JointRefiner(CONFIG["model"])
        self.assertEqual(sum(p.numel() for p in m.parameters()), 325024)
        self.assertEqual(list(inspect.signature(m.forward).parameters), ["mixture", "estimates"])

    def test_zero_head_is_projection_not_raw_separator(self):
        m = JointRefiner(CONFIG["model"])
        x, z = torch.randn(1, 2, 8193), torch.randn(1, 4, 2, 8193)
        with torch.no_grad():
            out = m(x, z)
        torch.testing.assert_close(out, project_mixture(x, z), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(out.sum(dim=1), x, atol=2e-6, rtol=1e-5)

    def test_nonzero_head_keeps_mixture_sum(self):
        m = JointRefiner(CONFIG["model"])
        with torch.no_grad():
            m.head.weight.normal_(std=0.01)
            out = m(torch.randn(1, 2, 8193), torch.randn(1, 4, 2, 8193))
        self.assertTrue(torch.isfinite(out).all())
        # Recompute with retained inputs for a direct sum assertion.
        x, z = torch.randn(1, 2, 8193), torch.randn(1, 4, 2, 8193)
        with torch.no_grad():
            out = m(x, z)
        torch.testing.assert_close(out.sum(dim=1), x, atol=3e-6, rtol=1e-5)

    def test_silent_input_remains_silent_after_nonzero_head(self):
        m = JointRefiner(CONFIG["model"])
        with torch.no_grad():
            m.head.bias.copy_(torch.arange(16, dtype=torch.float32))
            out = m(torch.zeros(1, 2, 8192), torch.zeros(1, 4, 2, 8192))
        self.assertEqual(float(out.abs().max()), 0)

    def test_loss_gradients_are_finite_with_a_silent_stem(self):
        target = torch.randn(1, 4, 2, 16384)
        target[:, 3] = 0
        for variant in ("base", "extended"):
            c = {**CONFIG["loss"], "variant": variant}
            prediction = (target + 0.05 * torch.randn_like(target)).requires_grad_()
            loss, _ = ReconstructionLoss(c)(prediction, target, target.sum(dim=1))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_exact_reconstruction_and_silence_have_finite_gradients(self):
        for quiet in (False, True):
            target = torch.zeros(1, 4, 2, 16384) if quiet else torch.randn(1, 4, 2, 16384)
            prediction = target.clone().requires_grad_()
            loss, _ = ReconstructionLoss(CONFIG["loss"])(prediction, target, target.sum(dim=1))
            loss.backward()
            self.assertAlmostEqual(float(loss.detach()), 0, places=6)
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_overlap_add_preserves_partial_final_chunk(self):
        m = JointRefiner(CONFIG["model"]).eval()
        x, z = torch.randn(2, 20001), torch.randn(4, 2, 20001)
        out = infer_record(m, x, z, 8192, "cpu")
        torch.testing.assert_close(out, project_mixture(x[None], z[None])[0], atol=3e-6, rtol=1e-5)

    def test_training_requires_explicit_flag_before_touching_runtime(self):
        with self.assertRaisesRegex(ValueError, "start-training"):
            train("does-not-exist", CONFIG)

    def test_cluster_bootstrap_keeps_song_conditions_together(self):
        rows = [{"track": s, "mean_delta_db": 2.0} for s in ("a", "b", "c") for _ in range(4)]
        result = cluster_summary(rows, 100, 42)
        self.assertEqual(result["songs"], 3)
        self.assertEqual(result["song_cluster_bootstrap_95_ci_db"], [2.0, 2.0])

    def test_preflight_backpropagates_without_any_optimizer_step(self):
        class FixtureCrops:
            records = [{"rir_set": "wet-fixture"}]

            def __init__(self, *args):
                pass

            def load(self, *args, **kwargs):
                y = torch.randn(4, 2, 8192) * 0.05
                return {"mixture": y.sum(dim=0), "targets": y, "estimates": y + 0.01 * torch.randn_like(y)}, {"fixture": True, "id": "fixture", "track": "fixture", "condition": "wet"}

        def forbidden_update(*args, **kwargs):
            raise AssertionError("preparation crossed the optimizer boundary")

        binding = {"fixture": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "manifests/dataset.json", {})
            atomic_json(root / "manifests/rirs.json", {})
            atomic_json(root / "manifests/cache.json", {"binding_sha256": digest(binding)})
            with patch("banelp_refine.engine.producer_binding", return_value=(binding, {})), \
                 patch("banelp_refine.engine.validate_inventory", return_value={"fixture": True}), \
                 patch("banelp_refine.engine.CachedCrops", FixtureCrops), \
                 patch.object(torch.optim.AdamW, "step", forbidden_update):
                report = preflight(root, CONFIG, root, root, device="cpu")
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(report["weights_unchanged"])
            self.assertEqual(report["status"], "cpu-check-only")
            ckpt = torch.load(root / "initial/step-000000.pt", weights_only=True)
            self.assertEqual(ckpt["optimizer"]["state"], {})
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                verify_ready(root, CONFIG, check_files=False)


if __name__ == "__main__":
    unittest.main()
