"""Tests for study resume, artifact integrity and paired bootstrap."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep import study
from tsprep import cli
from tsprep.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.experiment import (train_experiment, load_artifact, verify_metrics, file_hash,
                                verify_artifact, write_json)
from tsprep.models.decoders import TCNDecoder
from tsprep.provenance import read_json, atomic_write, input_arrays, input_fingerprint
from tsprep.evaluation.uncertainty import paired_block_interval


class StudyIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.target = Path(self.temporary.name)/"study"
        s = generate_series(length=700, seed=2026)
        self.data = dict(raw=s.values, regime=s.regime, is_anomaly=s.is_anomaly,
                         clean_raw=s.clean_values, dates=None, feature_names=["price"])
        self.args = dict(epochs=1, seeds=(42, 43), datasets=("synthetic",),
                         models=["tcn_ae"], folds=(0,), device="cpu")

    def tearDown(self):
        self.temporary.cleanup()

    def run_study(self, **kwargs):
        with patch.object(study, "dataset", return_value=self.data), redirect_stdout(io.StringIO()):
            return study.run_study(Path(__file__).resolve().parents[1], self.target, **{**self.args, **kwargs})

    def interrupt(self):
        real = study.train_experiment
        def call(exp, *args, **kwargs):
            if exp.train.seed == 43:
                raise RuntimeError("simulated interruption")
            return real(exp, *args, **kwargs)
        with patch.object(study, "train_experiment", side_effect=call):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_study()

    def test_resume_keeps_completed_run_and_finishes_grid(self):
        self.interrupt()
        first = next(self.target.glob("*/fold_*/*/model.pt"))
        original_hash = file_hash(first)
        self.assertFalse((self.target/"complete.json").exists())
        self.run_study()
        self.assertEqual(file_hash(first), original_hash)
        self.assertEqual(len(study.validate_completed_study(self.target)), 2)

    def test_changed_values_rejected_before_any_write(self):
        self.interrupt()
        previous = file_hash(self.target/"data_quality.json")
        self.data["raw"] = self.data["raw"].copy()
        self.data["raw"][20] *= 1.01
        with self.assertRaisesRegex(ValueError, "changed inputs"):
            self.run_study()
        self.assertEqual(file_hash(self.target/"data_quality.json"), previous)
        self.assertFalse((self.target/"complete.json").exists())

    def test_changed_label_rejected(self):
        self.interrupt()
        self.data["regime"] = self.data["regime"].copy()
        self.data["regime"][20] = 99
        with self.assertRaisesRegex(ValueError, "changed inputs"):
            self.run_study()

    def test_corrupt_cached_weights_rejected_on_resume(self):
        self.interrupt()
        with next(self.target.glob("*/fold_*/*/model.pt")).open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.run_study()

    def test_corrupt_cached_metrics_rejected_on_resume(self):
        self.interrupt()
        path = next(self.target.glob("*/fold_*/*/metrics.json"))
        metrics = read_json(path)
        metrics["mse"] = -999
        write_json(path, metrics)
        with self.assertRaisesRegex(ValueError, "metrics provenance/checksum"):
            self.run_study()

    def test_corrupt_baseline_rejected(self):
        self.interrupt()
        path = next(self.target.glob("*/fold_*/baselines.json"))
        baseline = read_json(path)
        baseline["pca"]["mse"] = -999
        write_json(path, baseline)
        with self.assertRaisesRegex(ValueError, "Baseline checksum"):
            self.run_study()

    def test_final_publication_failure_does_not_publish_completion(self):
        real = study.summarize
        def call(rows, target):
            real(rows, target)
            raise RuntimeError("publication interrupted")
        with patch.object(study, "summarize", side_effect=call):
            with self.assertRaisesRegex(RuntimeError, "publication interrupted"):
                self.run_study()
        self.assertFalse((self.target/"complete.json").exists())
        self.run_study()
        self.assertEqual(len(study.validate_completed_study(self.target)), 2)

    def test_finished_resume_never_rebuilds_partial_table(self):
        self.run_study()
        original = file_hash(self.target/"comparison_runs.csv")
        with patch.object(study, "summarize", side_effect=AssertionError("must not rebuild")):
            self.run_study()
        self.assertEqual(file_hash(self.target/"comparison_runs.csv"), original)

    def test_stale_completion_and_duplicate_grid_rejected(self):
        self.run_study()
        frame = pd.read_csv(self.target/"comparison_runs.csv")
        frame.iloc[:1].to_csv(self.target/"comparison_runs.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicated"):
            study.validate_completed_study(self.target)
        with self.assertRaises(ValueError):
            study.check_grid(pd.concat([frame.iloc[:1]]*2), read_json(self.target/"protocol.json"))

    def test_atomic_file_retains_old_content_on_replace_failure(self):
        path = Path(self.temporary.name)/"table.csv"
        path.write_bytes(b"original")
        with patch("tsprep.provenance.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                atomic_write(path, b"partial replacement")
        self.assertEqual(path.read_bytes(), b"original")
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_fingerprint_includes_labels_dates_and_feature_order(self):
        arrays = input_arrays(np.ones((10, 2)), regime=np.zeros(10), dates=np.arange(10).astype(str))
        original = input_fingerprint(arrays, ["a", "b"])
        self.assertNotEqual(original, input_fingerprint(arrays, ["b", "a"]))
        arrays["regime"][0] = 1
        self.assertNotEqual(original, input_fingerprint(arrays, ["a", "b"]))

    def test_fixed_interpolation_matches_forward_and_gradient(self):
        for length in (1, 7, 48, 49):
            torch.manual_seed(12)
            a = TCNDecoder(1, 4, 3, length, 1, 3, 0, "linear")
            b = TCNDecoder(1, 4, 3, length, 1, 3, 0, "matmul")
            b.load_state_dict(a.state_dict())
            za = torch.randn(3, 3, requires_grad=True)
            zb = za.detach().clone().requires_grad_(True)
            xa, xb = a(za), b(zb)
            torch.testing.assert_close(xa, xb, atol=2e-6, rtol=2e-5)
            xa.square().sum().backward()
            xb.square().sum().backward()
            torch.testing.assert_close(za.grad, zb.grad, atol=2e-6, rtol=2e-5)

    def test_bootstrap_rejects_full_length_and_too_few_blocks(self):
        delta = np.tile(np.arange(75.), (3, 1))
        for block in (24, 48, 75, 96):
            report = paired_block_interval(delta, block, repeats=100)
            self.assertIsNone(report["lower"])
            self.assertEqual(report["status"], "insufficient_temporal_blocks")
            self.assertEqual(report["block_len"], block)

    def test_paired_interval_preserves_constant_difference(self):
        report = paired_block_interval(np.full((3, 300), -2.), 24, repeats=100)
        self.assertEqual(report["status"], "ok")
        self.assertEqual((report["lower"], report["upper"]), (-2., -2.))
        variable = paired_block_interval(np.tile(np.arange(300.), (3, 1)), 24, repeats=100)
        self.assertLess(variable["lower"], variable["upper"])

    def test_source_mismatch_requires_explicit_historical_mode(self):
        self.run_study(seeds=(42,))
        path = next(self.target.glob("*/fold_*/*/manifest.json"))
        manifest = read_json(path)
        manifest["source_hashes"]["models/decoders.py"] = "changed"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Computational source mismatch"):
            load_artifact(path.parent)
        with self.assertWarns(UserWarning):
            load_artifact(path.parent, allow_source_mismatch=True)

    def test_cli_initialization_seed_does_not_change_synthetic_data(self):
        captured = []
        def capture(exp, raw, **kwargs):
            captured.append((exp, raw.copy()))
            return dict(mse=1., forecast=dict(mae=1.))
        for seed in (42, 43):
            argv = ["tsprep", "--model", "tcn_ae", "--length", "700", "--epochs", "1",
                    "--model-seed", str(seed), "--loader-seed", "500", "--device", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(cli, "train_experiment", side_effect=capture), \
                    redirect_stdout(io.StringIO()):
                cli.main()
        np.testing.assert_array_equal(captured[0][1], captured[1][1])
        self.assertEqual([item[0].train.seed for item in captured], [42, 43])
        self.assertEqual([item[0].data.seed for item in captured], [500, 500])

    def test_v2_loader_preserves_original_interpolation(self):
        self.run_study(seeds=(42,))
        path = next(self.target.glob("*/fold_*/*/manifest.json"))
        manifest = read_json(path)
        manifest["version"] = 2
        del manifest["config"]["model"]["upsampling"]
        write_json(path, manifest)
        model, _, exp, _, _ = load_artifact(path.parent)
        self.assertEqual(exp.model.upsampling, "linear")
        self.assertEqual(model.decoder.upsampling, "linear")


if __name__ == "__main__":
    unittest.main()
