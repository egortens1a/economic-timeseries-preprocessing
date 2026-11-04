"""Checks for paired targets and saved downstream tables."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from tsprep.data_preprocessing.datasets import prepare_and_window, build_eval_loader
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer
from tsprep.experiment import train_experiment, load_artifact, evaluate_artifact, verify_metrics, write_json
from tsprep.mlcore.trainer import _run_epoch
from tsprep.provenance import read_json
from tsprep.evaluation.study_results import load_study_tables, task_summary
from tsprep.evaluation.examples import downstream_example
from tsprep.evaluation.reconstruction import collect_embeddings
from tsprep import study


class DownstreamNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_target_is_used_in_loss_and_input_stays_noisy(self):
        class Identity(torch.nn.Module):
            variational = False
            def forward(self, x):
                return x, x[:, 0], x[:, 0], None
        x = np.ones((5, 4, 1), np.float32)
        clean = x.copy()
        clean[:, -1] = 0
        cfg = TrainConfig(validation_target="clean")
        result = _run_epoch(Identity(), build_eval_loader(x, 3, targets=clean), cfg, 0)
        self.assertEqual(result["recon"], .25)
        self.assertEqual(result["endpoint_mae"], 1.)
        with self.assertRaisesRegex(ValueError, "paired"):
            _run_epoch(Identity(), build_eval_loader(x, 3), cfg, 0)

    def test_clean_target_required_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run"
            exp = ExperimentConfig(train=TrainConfig(training_target="clean"))
            with self.assertRaisesRegex(ValueError, "aligned clean_raw"):
                train_experiment(exp, np.ones((400, 1)), output_dir=path)
            self.assertFalse(path.exists())

    def test_clean_target_roundtrip_uses_observed_train_normalizer(self):
        s = generate_series(length=500, seed=9001)
        cfg = DataConfig(window_len=8, batch_size=64)
        expected = prepare_and_window(s.values, cfg).normalizer.state()
        exp = ExperimentConfig(cfg, ModelConfig(model_name="tcn_ae", hidden_dim=8, latent_dim=3, tcn_levels=1),
            TrainConfig(epochs=2, training_target="clean", validation_target="clean", deterministic=True,
                        selection_metric="endpoint_mae"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run"
            before = train_experiment(exp, s.values, clean_raw=s.clean_values, output_dir=path, verbose=False)
            model, p, restored, manifest, _ = load_artifact(path)
            self.assertEqual(p.normalizer.state(), expected)
            self.assertEqual(restored.train.training_target, "clean")
            self.assertEqual(restored.train.validation_target, "clean")
            clean = prepare_and_window(s.clean_values, cfg, normalizer=p.normalizer)
            val = collect_embeddings(model, build_eval_loader(p.val_windows, cfg.batch_size))
            actual_selection = float(np.abs(val["x_hat"][:, -1]-clean.val_windows[:, -1]).mean())
            self.assertAlmostEqual(manifest["history"]["best_score"], actual_selection, places=6)
            after = evaluate_artifact(path)
            self.assertAlmostEqual(before["clean_reconstruction"]["mse"], after["clean_reconstruction"]["mse"])
            example = downstream_example(path)
            self.assertIn("clean", example["endpoint"])
            self.assertEqual(len(example["forecast"]["target"]), len(example["forecast"]["predictions"]["embedding_ridge"]))

    def test_historical_metric_permission_never_bypasses_checksum(self):
        s = generate_series(length=400)
        exp = ExperimentConfig(DataConfig(window_len=8), ModelConfig(model_name="tcn_ae", hidden_dim=8, tcn_levels=1),
                               TrainConfig(epochs=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"run"
            train_experiment(exp, s.values, output_dir=path, verbose=False)
            with patch("tsprep.experiment.evaluator_hash", return_value="another evaluator"):
                with self.assertRaisesRegex(ValueError, "evaluator mismatch"):
                    verify_metrics(path)
                self.assertEqual(verify_metrics(path, allow_evaluator_mismatch=True)["model"], "tcn_ae")
            m = read_json(path/"metrics.json")
            m["mse"] = -1
            write_json(path/"metrics.json", m)
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_metrics(path, allow_evaluator_mismatch=True)

    def test_saved_tables_keep_baselines_unique_and_missing_tasks_absent(self):
        s = generate_series(length=700, seed=9001)
        data = dict(raw=s.values, regime=s.regime, is_anomaly=s.is_anomaly,
                    clean_raw=s.clean_values, feature_names=["price"], dates=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"study"
            with patch.object(study, "dataset", return_value=data), redirect_stdout(io.StringIO()):
                study.run_study(Path(__file__).resolve().parents[1], path, epochs=1, seeds=(42, 43),
                                datasets=("synthetic",), folds=(0,), models=["tcn_ae"])
            tables = load_study_tables(path)
            fore = tables["forecast"]
            self.assertEqual(len(fore[fore.role == "model"]), 2)
            self.assertFalse(fore[fore.role == "baseline"].duplicated(["dataset", "fold", "model"]).any())
            raw = fore[(fore.role == "baseline") & (fore.model == "raw_window_ridge")].iloc[0]
            self.assertAlmostEqual(raw.delta_mae_vs_raw, 0.)
            self.assertTrue(tables["classification"].empty)
            self.assertFalse(tables["anomaly"].empty)
            self.assertEqual(set(tables["anomaly"].unit), {"window", "endpoint"})
            self.assertEqual(len(tables["clean"][tables["clean"].model == "observed_input"]), 1)
            self.assertFalse(task_summary(fore, ["mae", "r2"]).empty)


if __name__ == "__main__":
    unittest.main()
