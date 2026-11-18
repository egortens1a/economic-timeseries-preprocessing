"""Проверки заполнения пропусков и повторного запуска на небольших данных."""
from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.config import ImputationProtocol
from tsprep.imputation import data as d, baselines as f, training as n
from tsprep.imputation.evaluation import bootstrap_indices, bootstrap_ratio, decide, forecast_features, forecast_predict
from tsprep.imputation_experiment import run_experiment
from tsprep.provenance import file_hash, read_json

spec = importlib.util.spec_from_file_location("nn5_components", Path(__file__).resolve().parents[1]/"scripts/check_nn5_components.py")
components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components)


def toy():
    t = np.arange(791.)
    raw = np.stack([20+i*5+.01*t+4*np.sin(2*np.pi*t/7)+np.cos(t/20+i) for i in range(4)])
    raw[:, 30] = np.nan
    raw[:, 50:52] = np.nan
    scaler, normalized = d.fit_scaler(raw)
    train = d.train_windows(normalized)
    train["truth_raw"] = np.stack([raw[s, a:a+56] for s, a in zip(train["atm"], train["start"])])
    return raw, normalized, scaler, train


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw, cls.normalized, cls.scaler, cls.train = toy()
        cls.pca = f.fit_pca(cls.train, cls.scaler, 4)
        cls.ridge = f.fit_ridge(cls.train, cls.scaler, 1.)


    def test_scaler_does_not_use_validation_or_test(self):
        modified = self.raw.copy()
        modified[:, 511:] *= 100
        scaler, _ = d.fit_scaler(modified)
        for key in self.scaler:
            np.testing.assert_array_equal(scaler[key], self.scaler[key])

    def test_mask_count_replay_and_no_mutation(self):
        original = self.train["windows"].copy()
        a = d.sample_mask(original, np.random.default_rng(1), self.scaler)
        b = d.sample_mask(original, np.random.default_rng(1), self.scaler)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a.sum(axis=1), np.ceil(.05*np.isfinite(original).sum(axis=1)))
        self.assertTrue(np.isfinite(original[a]).all())
        np.testing.assert_array_equal(original, self.train["windows"])

    def test_weekly_scoring_has_no_duplicate_cells(self):
        extra = d.additional_masks(self.normalized, self.scaler, 511, 623, [31001])
        case = d.evaluation_windows(self.raw, self.normalized, extra, 511, 623)
        self.assertFalse(case["score_mask"][:, :-7].any())
        self.assertEqual(int(case["score_mask"].sum()), int(extra.sum()))
        self.assertFalse(extra[:, :, :511].any())
        self.assertFalse(extra[:, :, 623:].any())

    def test_no_observation_after_origin_is_used(self):
        extra = d.additional_masks(self.normalized, self.scaler, 511, 623, [31001])
        a = d.evaluation_windows(self.raw, self.normalized, extra, 511, 623)
        modified = self.normalized.copy()
        modified[:, 518:] += 100
        b = d.evaluation_windows(self.raw, modified, extra, 511, 623)
        np.testing.assert_array_equal(a["windows"][0], b["windows"][0])
        self.assertEqual(int(a["start"][0])+55, 517)

    def test_linear_fills_affine_gap(self):
        values = np.tile(np.arange(56.)/10, (4, 1))
        values[:, 20:23] = np.nan
        case = dict(windows=values, atm=np.arange(4), start=np.zeros(4, int))
        prediction = f.simple("linear", case, self.scaler)
        np.testing.assert_allclose(prediction[:, 20:23], np.tile(np.array([2., 2.1, 2.2]), (4, 1)))

    def test_every_filler_preserves_observed_cells(self):
        masked = self.train["windows"][:16].copy()
        masked[:, 20:23] = np.nan
        case = dict(windows=masked, atm=self.train["atm"][:16], start=self.train["start"][:16])
        fitted = dict(pca=self.pca, ridge_gap=self.ridge)
        for method in f.METHODS:
            prediction = f.predict(method, case, self.scaler, fitted)
            self.assertTrue(np.isfinite(prediction).all())
            known = np.isfinite(masked)
            np.testing.assert_array_equal(prediction[known], masked[known])

    def test_hidden_reference_is_not_an_input(self):
        case = {key: value[:8].copy() for key, value in self.train.items()}
        case["windows"][:, 20] = np.nan
        before = f.ridge_fill(case, self.scaler, self.ridge)
        case["truth_raw"][:, 20] += 100000
        after = f.ridge_fill(case, self.scaler, self.ridge)
        np.testing.assert_array_equal(before, after)

    def test_zero_head_and_zero_shrink_equal_baseline(self):
        case = {key: value[:8].copy() for key, value in self.train.items()}
        case["windows"][:, 20] = np.nan
        base = f.simple("linear", case, self.scaler)
        model = n.build(111, "cpu")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4305)
        np.testing.assert_array_equal(n.predict(model, case, self.scaler, base, "cpu"), base)
        np.testing.assert_array_equal(n.predict(None, case, self.scaler, base, "cpu", 0.), base)

    def test_gradient_and_checkpoint_reload(self):
        model = n.build(111, "cpu")
        x = torch.as_tensor(np.random.default_rng(11).normal(size=(8, 56, 5)).astype(np.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = ((model(x)-1)**2).mean()
        loss.backward()
        optimizer.step()
        self.assertGreater(float(model.head.weight.detach().abs().sum()), 0.)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = n.build(222, "cpu")
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        np.testing.assert_array_equal(model(x).detach().numpy(), restored(x).detach().numpy())

    def test_native_known_copy_and_masked_metrics(self):
        extra = d.additional_masks(self.normalized, self.scaler, 511, 623, [31001])
        case = d.evaluation_windows(self.raw, self.normalized, extra, 511, 623)
        prediction = f.simple("weekday_train", case, self.scaler)
        physical = d.native_values(prediction, case, self.scaler)
        known = np.isfinite(case["windows"])
        np.testing.assert_array_equal(physical[known], case["truth_raw"][known])
        block = d.error_blocks(prediction, case, self.scaler)
        self.assertEqual(int(block["count"].sum()), int(extra.sum()))
        self.assertTrue(np.isfinite(block["score"]))

    def test_forecast_alignment_and_feature_dimensions(self):
        frame = d.forecast_windows(self.raw, self.normalized, 511, 623)
        self.assertTrue((frame["start"]+56 == frame["target_index"]).all())
        filled = f.simple("linear", frame, self.scaler)
        features = forecast_features(filled, frame)
        self.assertEqual(features.shape[1], 119)
        np.testing.assert_array_equal(features[:, 56:112], ~np.isfinite(frame["windows"]))
        np.testing.assert_array_equal(features[:, 112:].argmax(axis=1), frame["target_index"] % 7)

    def test_paired_bootstrap_preserves_constant_ratio(self):
        atm, weeks = bootstrap_indices(5, 8, repeats=50)
        a = np.ones((5, 8))
        np.testing.assert_allclose(bootstrap_ratio(a, a*2, atm, weeks), .5)
        np.testing.assert_array_equal(np.diff(weeks[:, :4], axis=1) % 8, 1)

    def test_component_inputs_and_initial_backbones_are_paired(self):
        case = {key: value[:8].copy() for key, value in self.train.items()}
        case["windows"][:, 20] = np.nan
        base = f.simple("weekday_window", case, self.scaler)
        original = f.network_features(case, self.scaler, base)
        disabled = components.without_phase(case, self.scaler, base)
        np.testing.assert_array_equal(original[..., :3], disabled[..., :3])
        self.assertTrue((disabled[..., 3:] == 0).all())
        models = [components.build_arm(arm, 11, "cpu")[0] for arm in components.ARMS]
        self.assertEqual(len({components.backbone_hash(model) for model in models}), 1)
        self.assertEqual([sum(v.numel() for v in model.parameters()) for model in models], [4305, 4289, 4305])
        for model in models:
            np.testing.assert_array_equal(model(torch.as_tensor(original)).detach().numpy(), np.zeros(base.shape))

    def test_forward_pass_has_no_direct_dependence_on_later_input(self):
        model, _ = components.build_arm("forward", 11, "cpu")
        with torch.no_grad():
            model.head.weight.fill_(.1)
        x = torch.as_tensor(np.random.default_rng(12).normal(size=(2, 56, 5)).astype(np.float32))
        changed = x.clone()
        changed[:, 30:] += 100
        a, b = model(x).detach().numpy(), model(changed).detach().numpy()
        np.testing.assert_array_equal(a[:, :30], b[:, :30])
        self.assertFalse(np.array_equal(a[:, 30:], b[:, 30:]))


class UniversalImputationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.p = ImputationProtocol(window=20, period=5, train_stop=80, val_stop=100,
                    epochs=2, batch_size=32, learning_rates=(.001,), initialization_seeds=(11,),
                    pca_ranks=(4, 8), validation_mask_seeds=(31,), test_mask_seeds=(41,),
                    bootstrap_repeats=30)
        t = np.arange(130.)
        rng = np.random.default_rng(19)
        self.raw = np.stack([20+5*np.sin(t*2*np.pi/5+i)+rng.normal(size=len(t)) for i in range(3)])
        self.raw[:, 30:32] = np.nan
        self.raw[0, 0] = 0.
        self.input = self.root/"panel.npz"
        np.savez(self.input, values=self.raw)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_zip_and_npz_have_same_values(self):
        path = self.root/"original.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("input.tsf", "@frequency daily\n@data\nT1:2000-01-01 00-00-00:0,?,2\nT2:2000-01-01 00-00-00:3,4,5\n")
        panel = d.load_panel(path)
        np.testing.assert_array_equal(panel["values"], [[0., np.nan, 2.], [3., 4., 5.]])
        np.testing.assert_array_equal(panel["dates"], ["2000-01-01", "2000-01-02", "2000-01-03"])
        self.assertEqual(panel["series_ids"].tolist(), ["T1", "T2"])

    def test_other_period_and_window_keep_scoring_and_forecast_alignment(self):
        self.p.validate(self.raw.shape[1])
        scaler, normalized = d.fit_scaler(self.raw, self.p)
        masks = d.additional_masks(normalized, scaler, 100, 130, [41])
        case = d.evaluation_windows(self.raw, normalized, masks, 100, 130, self.p)
        self.assertEqual(int(case["score_mask"].sum()), int(masks.sum()))
        self.assertFalse(case["score_mask"][:, :-5].any())
        frame = d.forecast_windows(self.raw, normalized, 100, 130, p=self.p)
        value = f.simple("linear", frame, scaler)
        self.assertEqual(forecast_features(value, frame).shape[1], 45)
        self.assertTrue((frame["start"]+20 == frame["target_index"]).all())

    def test_no_natural_training_gaps_use_single_point_augmentation(self):
        scaler, _ = d.fit_scaler(np.nan_to_num(self.raw), self.p)
        np.testing.assert_array_equal(scaler["gap_lengths"], [1])
        np.testing.assert_array_equal(scaler["gap_probabilities"], [1.])

    def test_bad_input_or_protocol_does_not_create_results(self):
        for p in (replace(self.p, val_stop=101), replace(self.p, window=19),
                  replace(self.p, missing_fraction=0), replace(self.p, learning_rates=(float("nan"),)),
                  replace(self.p, validation_mask_seeds=(41,))):
            with self.assertRaises(ValueError):
                run_experiment(self.input, self.root/"invalid", p)
        self.assertFalse((self.root/"invalid").exists())
        for bad in (np.array([1., 2.]), np.array([[1., np.inf]]), np.array([[-1., 2.]])):
            np.savez(self.root/"bad.npz", values=bad)
            with self.assertRaises(ValueError):
                d.load_panel(self.root/"bad.npz")

    def test_missing_training_phase_is_rejected(self):
        values = self.raw.copy()
        values[0, :80:5] = np.nan
        with self.assertRaisesRegex(ValueError, "seasonal phase"):
            d.fit_scaler(values, self.p)

    def test_undefined_uncertainty_does_not_claim_success(self):
        blocks = {name: dict(scaled=np.zeros((1, 3, 6))) for name in ("neural_11", "linear", "ridge_gap", "pca")}
        decision, _ = decide(blocks, "linear", self.p)
        self.assertFalse(decision["joint_success"])
        for comparison in decision["primary"].values():
            self.assertIsNone(comparison["ratio"])
            self.assertIsNone(comparison["corrected_one_sided_upper"])
            self.assertEqual(comparison["status"], "undefined")

    def test_two_complete_runs_reproduce_and_do_not_overwrite(self):
        before = file_hash(self.input)
        with redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
            for name in ("a", "b"):
                run_experiment(self.input, self.root/name, self.p)
        self.assertEqual(file_hash(self.input), before)
        self.assertEqual(file_hash(self.root/"a/inputs/original.npz"), before)
        required = ("imputation_summary.csv", "forecast_summary.csv", "decision.json",
                    "imputation_metrics.png", "imputation_example.png", "source.zip", "complete.json")
        for name in required:
            self.assertTrue((self.root/"a"/name).is_file(), name)
        for name in ("imputation_summary.csv", "forecast_summary.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(self.root/"a"/name), pd.read_csv(self.root/"b"/name), check_exact=True)
        receipt = read_json(self.root/"a/complete.json")
        self.assertEqual(receipt["status"], "complete")
        for name, digest in receipt["files"].items():
            self.assertEqual(file_hash(self.root/"a"/name), digest)
        protocol = read_json(self.root/"a/protocol.json")
        self.assertEqual((protocol["series"], protocol["length"], protocol["period"]), (3, 130, 5))
        with self.assertRaises(FileExistsError):
            run_experiment(self.input, self.root/"a", self.p)
        self.assertEqual(read_json(self.root/"a/complete.json"), receipt)

    def test_training_refactor_preserves_default_weights_and_history(self):
        scaler, normalized = d.fit_scaler(self.raw, self.p)
        train = d.train_windows(normalized, self.p)
        train["truth_raw"] = np.stack([self.raw[s, a:a+self.p.window] for s, a in zip(train["atm"], train["start"])])
        masks = d.additional_masks(normalized, scaler, 80, 100, [31])
        val = d.evaluation_windows(self.raw, normalized, masks, 80, 100, self.p)
        base = f.simple("weekday_window", val, scaler)
        with threadpool_limits(limits=1):
            a, receipt_a = n.fit(train, val, scaler, {}, "weekday_window", base, .001, 11, "cpu", self.p)
            b, features = components.build_arm("full", 11, "cpu")
            b, receipt_b = n._fit_model(b, features, train, val, scaler, {}, "weekday_window", base, .001, 11, "cpu", self.p)
        for key in a.state_dict():
            np.testing.assert_array_equal(a.state_dict()[key].numpy(), b.state_dict()[key].numpy())
        self.assertEqual(receipt_a["history"], receipt_b["history"])

    def test_component_runs_reproduce_preserve_known_values_and_reload(self):
        p = replace(self.p, epochs=1, ridge_alphas=(1.,), shrinkages=(0., 1.))
        with redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
            for name in ("c", "d"):
                components.run(self.input, self.root/name, p)
        pd.testing.assert_frame_equal(pd.read_csv(self.root/"c/summary.csv"), pd.read_csv(self.root/"d/summary.csv"), check_exact=True)
        complete = read_json(self.root/"c/complete.json")
        for name, digest in complete["files"].items():
            self.assertEqual(file_hash(self.root/"c"/name), digest)
        with zipfile.ZipFile(self.root/"c/source.zip") as archive:
            self.assertIn("scripts/check_nn5_components.py", archive.namelist())
        with np.load(self.root/"c/test/windows.npz") as frame, np.load(self.root/"c/test/predictions.npz") as predictions:
            known = np.isfinite(frame["windows"])
            for name in predictions.files:
                np.testing.assert_array_equal(predictions[name][known], frame["windows"][known])
            altered = dict(frame)
            altered["truth_raw"] = altered["truth_raw"]+1000
            scaler, _ = d.fit_scaler(self.raw, p)
            with (self.root/"c/models/boosting.pkl").open("rb") as stream:
                model = components.pickle.load(stream)
            shrink = read_json(self.root/"c/boosting_selection.json")["selected"]["shrinkage"]
            np.testing.assert_array_equal(components.boosting_fill(model, altered, scaler, shrink), predictions["boosting"])
        with self.assertRaises(FileExistsError):
            components.run(self.input, self.root/"c", p)
        self.assertEqual(read_json(self.root/"c/complete.json"), complete)


if __name__ == "__main__":
    unittest.main()
