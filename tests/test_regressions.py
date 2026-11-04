"""Regression tests for preprocessing, training and evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.config import *
from tsprep.models.autoencoder import build_model
from tsprep.models.decoders import LSTMDecoder
from tsprep.models.layers import PositionalEncoding
from tsprep.mlcore.trainer import EarlyStopping, _run_epoch, train_model
from tsprep.data_preprocessing.datasets import *
from tsprep.data_preprocessing.preprocessing import *
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.data_preprocessing.loaders import align_labels_to_dates
from tsprep.evaluation.downstream import *
from tsprep.evaluation.latent_space import information_collapse_ratio
from tsprep.experiment import train_experiment, load_artifact, evaluate_artifact


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_existing_functions(self):
        path = Path(__file__).with_name("test_pipeline.py")
        spec = importlib.util.spec_from_file_location("legacy_tests", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(n for n in vars(module) if n.startswith("test_")):
            with self.subTest(name=name):
                getattr(module, name)()

    def test_lstm_batch_invariance(self):
        torch.manual_seed(42)
        decoder = LSTMDecoder(2, 8, 3, 9, 2, 0).eval()
        z = torch.randn(5, 3)
        with torch.no_grad():
            together = decoder(z)
            singles = torch.cat([decoder(v[None]) for v in z])
            perm = torch.tensor([4, 1, 0, 3, 2])
            torch.testing.assert_close(together, singles, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(decoder(z[perm]), together[perm])

    def test_patience(self):
        stop = EarlyStopping(3, .0001)
        self.assertEqual([stop.should_stop(v) for v in [5, 4, 3, 2, 2.1, 2.1, 2.1]],
                         [False, False, False, False, False, False, True])

    def test_patience_resets(self):
        stop = EarlyStopping(2, .01)
        self.assertEqual([stop.should_stop(v) for v in [1, 1.1, .8, .9, 1]],
                         [False, False, False, False, True])

    def test_sample_weighted_loss(self):
        class Zero(torch.nn.Module):
            variational = False
            def forward(self, x):
                return torch.zeros_like(x), None, None, None
        x = np.zeros((65, 2, 1), np.float32); x[-1] = 10
        result = _run_epoch(Zero(), build_eval_loader(x, 64), TrainConfig(), 0)
        self.assertAlmostEqual(result["loss"], 100/65)

    def test_no_dropped_small_train(self):
        x = np.arange(1., 201.)[:, None]
        cfg = DataConfig(window_len=8, batch_size=256)
        p = prepare_and_window(x, cfg)
        loader, _, _ = build_dataloaders(p, cfg)
        self.assertEqual(len(loader), 1)
        self.assertEqual(len(next(iter(loader))), len(p.train_windows))

    def test_empty_loader_raises(self):
        class Zero(torch.nn.Module):
            pass
        with self.assertRaises(ValueError):
            _run_epoch(Zero(), [], TrainConfig(), 0)

    def test_fixed_checkpoint_after_warmup(self):
        model = build_model(ModelConfig(model_name="tcn_vae", hidden_dim=8, tcn_levels=1), 1, 8)
        def epoch(m, loader, cfg, beta, optimizer=None):
            if optimizer is not None:
                return dict(loss=1., recon=1., kl=0.)
            recon, kl = next(values)
            return dict(loss=recon+beta*kl, recon=recon, kl=kl)
        values = iter([(0.01, 100), (.5, .2), (.4, .1)])
        with patch("tsprep.mlcore.trainer._run_epoch", side_effect=epoch):
            history = train_model(model, None, None, TrainConfig(epochs=3, kl_anneal_epochs=1), verbose=False)
        self.assertEqual(history.best_epoch, 3)
        self.assertEqual(history.best_beta, .01)

    def test_exact_horizon(self):
        raw = np.arange(1., 301.)[:, None]
        for stride in (1, 2, 3, 8):
            cfg = DataConfig(window_len=8, forecast_horizon=5, stride=stride, seasonal_period=0, log_transform=False)
            p = prepare_and_window(raw, cfg)
            ids, y = forecast_pairs(p, "test", cfg)
            targets = p.window_starts["test"][ids]+7+5
            np.testing.assert_array_equal(y, p.raw_normalized[targets, 0])
            self.assertTrue((targets < len(p.raw_normalized)).all())

    def test_anomaly_mask_difference(self):
        raw = np.arange(1., 301.)[:, None]
        mask = np.zeros(300, bool); mask[10] = True
        p = prepare_and_window(raw, DataConfig(window_len=8, seasonal_period=3), is_anomaly=mask)
        np.testing.assert_array_equal(np.flatnonzero(p.is_anomaly), [7, 10])

    def test_no_test_threshold_tuning(self):
        scores = np.arange(100, dtype=float)
        threshold = calibrate_threshold(np.arange(50.), quantile=.8)
        x, xhat = np.zeros((100,1,1)), np.sqrt(scores)[:,None,None]
        a = anomaly_detection_f1(x,xhat,(scores>90).astype(int),threshold=threshold)
        b = anomaly_detection_f1(x,xhat,(scores>70).astype(int),threshold=threshold)
        self.assertEqual(a["threshold"], b["threshold"])
        with self.assertRaises(TypeError):
            anomaly_detection_f1(x,xhat,(scores>70).astype(int))

    def test_single_class_not_kmeans(self):
        z = np.random.default_rng(1).normal(size=(30,3))
        result = clustering_quality(z, np.zeros(30))
        self.assertIsNone(result["silhouette"])
        self.assertEqual(result["label_source"], "true_regime")

    def test_constant_embeddings(self):
        self.assertIsNone(clustering_quality(np.ones((20,3)))["silhouette"])

    def test_single_class_classification(self):
        r = binary_report(np.zeros(20), np.zeros(20), np.zeros(20))
        self.assertIsNone(r["f1"]); self.assertIsNone(r["roc_auc"])
        self.assertEqual(r["false_positive_rate"], 0.)

    def test_no_future_label_and_order_preserved(self):
        values = np.array(["2000-01-03","2000-01-01"], dtype="datetime64[D]")
        dates = np.array(["2000-01-02"], dtype="datetime64[D]")
        result = align_labels_to_dates(values, dates, np.array([1]))
        self.assertEqual(result[0], 1); self.assertTrue(np.isnan(result[1]))

    def test_normalizer_persisted_no_refit(self):
        raw = np.arange(1., 301.)[:, None]
        cfg = DataConfig(window_len=8)
        p = prepare_and_window(raw, cfg)
        norm = TrainFittedNormalizer.from_state(p.normalizer.state())
        with patch.object(TrainFittedNormalizer, "fit", side_effect=AssertionError("unexpected fit")):
            restored = prepare_and_window(raw, cfg, normalizer=norm)
        np.testing.assert_array_equal(p.test_windows, restored.test_windows)

    def test_inverse_roundtrip(self):
        raw = np.exp(np.random.default_rng(42).normal(size=(100, 2)))
        for s in (0, 1, 5):
            prep = prepare_series(raw, seasonal_period=s)
            out = inverse_series(prep["normalized"], prep["normalizer"], prep["y0"], s, True)
            np.testing.assert_allclose(out, raw, atol=1e-10)

    def test_online_aggregation(self):
        w = np.arange(12.).reshape(3,4,1)
        x, covered = aggregate_windows(w, np.arange(3), 6, mode="last")
        np.testing.assert_array_equal(np.flatnonzero(covered), [3,4,5])
        np.testing.assert_array_equal(x[covered,0], w[:,-1,0])

    def test_config_validation(self):
        for kwargs in (dict(stride=0), dict(train_frac=-.1, val_frac=.5, test_frac=.6),
                       dict(forecast_horizon=0), dict(target_feature=2)):
            with self.assertRaises(ValueError):
                DataConfig(**kwargs)
        with self.assertRaises(ValueError):
            ModelConfig(model_name="hybrid_ae", hidden_dim=15)

    def test_synthetic_one_regime_clean_pairs(self):
        s = generate_series(length=500, n_regimes=1)
        self.assertEqual(np.unique(s.regime).tolist(), [0])
        self.assertTrue((s.clean_values>0).all())
        self.assertEqual(s.clean_values.shape, s.values.shape)

    def test_teacher_forcing_dispatch(self):
        model = build_model(ModelConfig(model_name="tcn_ae"), 1, 8)
        with self.assertRaises(ValueError):
            model(torch.randn(2,8,1), use_teacher_forcing=True)
        model = build_model(ModelConfig(model_name="lstm_ae", hidden_dim=8), 1, 8).train()
        self.assertEqual(model(torch.randn(2,8,1), use_teacher_forcing=True)[0].shape, (2,8,1))

    def test_positional_encoding_long_odd(self):
        layer = PositionalEncoding(5, max_len=8)
        self.assertEqual(layer(torch.zeros(2, 20, 5)).shape, (2,20,5))

    def test_variance_ratio_equivalence(self):
        rng = np.random.default_rng(1)
        z,x = rng.normal(size=(50,3)),rng.normal(size=(50,5,2))
        expected=np.trace(np.cov(z,rowvar=False))/np.trace(np.cov(x.reshape(50,-1),rowvar=False))
        self.assertAlmostEqual(information_collapse_ratio(z,x), expected)

    def test_artifact_roundtrip_and_tamper(self):
        s=generate_series(length=400,seed=42)
        cfg=ExperimentConfig(DataConfig(window_len=8,batch_size=64),
            ModelConfig(model_name="tcn_ae",hidden_dim=8,tcn_levels=1,latent_dim=3),
            TrainConfig(epochs=2,kl_anneal_epochs=0))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"run"
            before=train_experiment(cfg,s.values,output_dir=path,regime=s.regime,is_anomaly=s.is_anomaly,
                                    clean_raw=s.clean_values,verbose=False)
            after=evaluate_artifact(path)
            self.assertAlmostEqual(before["mse"],after["mse"],places=7)
            self.assertAlmostEqual(before["forecast"]["mae"],after["forecast"]["mae"],places=7)
            self.assertEqual(before["anomaly"]["threshold"],after["anomaly"]["threshold"])
            with (path/"model.pt").open("ab") as f: f.write(b"tampered")
            with self.assertRaises(ValueError): load_artifact(path)


if __name__ == "__main__":
    unittest.main()
