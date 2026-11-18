"""Проверки модели исправления и оценки на независимых траекториях."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import contextlib
import io
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.data_preprocessing.denoising import FAMILIES, generate_denoising_series
from tsprep.data_preprocessing.preprocessing import TrainFittedNormalizer
from tsprep.denoising_experiment import (DenoisingProtocol, _decision, _fit_probe, _probe_predict,
    endpoint_metrics, relative_scores, trajectory_interval, trajectory_segments)
from tsprep.models.denoiser import ResidualTCNDenoiser
from tsprep.config import RepresentationProtocol
from tsprep.data_preprocessing.datasets import TensorWindowLoader
from tsprep.denoising.data import representation_windows
from tsprep.denoising.metrics import representation_decision
from tsprep.provenance import file_hash, read_json
from tsprep.mlcore.losses import residual_loss


class ResidualDenoisingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_initialization_and_nontrivial_causality(self):
        torch.manual_seed(1)
        model = ResidualTCNDenoiser(hidden_dim=4, levels=2)
        x = torch.randn(3, 48, 1)
        self.assertTrue(torch.equal(model(x), x))
        # Exercise causality with an active correction head, not the initial identity.
        torch.nn.init.normal_(model.error_head.weight)
        y = x.clone()
        y[:, 25:] += 50
        self.assertTrue(torch.equal(model(x)[:, :25], model(y)[:, :25]))
        model(x).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_clean_loss_can_move_model_away_from_identity(self):
        model = ResidualTCNDenoiser(hidden_dim=4, levels=1)
        x = torch.randn(8, 16, 1)
        target = x[:, -1, 0] - .5
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        initial = (model(x)[:, -1, 0]-target).square().mean().item()
        for _ in range(30):
            loss = (model(x)[:, -1, 0]-target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess((model(x)[:, -1, 0]-target).square().mean().item(), initial/5)

    def test_local_features_exclude_current_value_and_future(self):
        model=ResidualTCNDenoiser(hidden_dim=4,levels=1,local_context=4)
        x=torch.tensor([1.,2.,3.,4.,100.,6.,7.,8.]).reshape(1,-1,1)
        features=model._local_features(x)
        self.assertEqual(features.shape,(1,8,4))
        self.assertAlmostEqual(features[0,4,1].item(),2.)
        self.assertAlmostEqual(features[0,4,2].exp().item(),1.4826,places=5)
        self.assertEqual(features[0,4,3].item(),20.)
        other=x.clone();other[:,4]=1000;other[:,5:]=-1000
        changed=model._local_features(other)
        self.assertTrue(torch.equal(features[:,:4],changed[:,:4]))
        self.assertTrue(torch.equal(features[:,4,1:3],changed[:,4,1:3]))
        zeros=model._local_features(torch.zeros(2,10,1))
        self.assertTrue(torch.isfinite(zeros).all())

    def test_delayed_identity_information_boundary_and_matching_parameters(self):
        models=[]
        for mode in ('past','next'):
            torch.manual_seed(111)
            models.append(ResidualTCNDenoiser(correction_delay=1,delay_mode=mode))
        self.assertTrue(all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items()))
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),4097)
        x=torch.randn(2,64,1)
        for model in models:
            self.assertTrue(torch.equal(model(x)[:,1:],x[:,:-1]))
            torch.nn.init.normal_(model.error_head.weight,std=.1)
            later=x.clone();later[:,41:]+=50
            self.assertTrue(torch.equal(model(x)[:,:41],model(later)[:,:41]))
            model(x).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        current=x.clone();current[:,40]+=100
        self.assertTrue(torch.equal(models[0](x)[:,:41],models[0](current)[:,:41]))
        self.assertFalse(torch.equal(models[1](x)[:,40],models[1](current)[:,40]))

    def test_delay_indices_separate_target_and_availability(self):
        from tsprep.denoising.data import denoising_indices
        segment=dict(observed=np.arange(12,dtype=np.float32)[:,None],clean=np.arange(12)[:,None],
                     anomaly=np.arange(12)==5)
        p=replace(DenoisingProtocol(),window=4,correction_delay=1)
        ends,availability=denoising_indices(segment,p,2)
        np.testing.assert_array_equal(ends,[2,4,6,8,10])
        np.testing.assert_array_equal(availability,[3,5,7,9,11])
        self.assertFalse(segment['anomaly'][ends].any())
        self.assertTrue(segment['anomaly'][availability].any())
        for bad in [replace(p,correction_delay=2),replace(p,local_context=16),
                    replace(DenoisingProtocol(),delay_mode='past')]:
            with self.assertRaises(ValueError):bad.validate()

    def test_delayed_pair_filter_preserves_same_sign_jump_and_uses_next(self):
        from tsprep.denoising.baselines import delayed_pair_values
        x=np.zeros((2,64),np.float32)
        x[0,62:]=[8.,-8.]
        x[1,62:]=[8.,6.]
        result=delayed_pair_values(x,5.,[62])
        self.assertEqual(result[0,62],0.)
        self.assertEqual(result[1,62],8.)
        self.assertTrue(np.array_equal(result[:,:62],x[:,:62]))
        with self.assertRaises(ValueError):delayed_pair_values(x,5.,[63])

    def test_delayed_features_align_all_methods_with_the_same_target(self):
        from tsprep.denoising.forecast import forecast_features
        class IdentityPCA:
            def transform(self,x):return x
            def inverse_transform(self,x):return x
        p=replace(DenoisingProtocol(),window=48,forecast_features=4,correction_delay=1)
        flat=np.stack([np.arange(48),np.arange(48)+3]).astype(np.float32)
        baseline=dict(mean=0.,global_center=0.,global_scale=100.,choices={
            'causal_median':{'setting':3},'causal_hampel':{'setting':[5,3.]},
            'global_mad':{'setting':9.},'delayed_pair':{'setting':5.}})
        model=ResidualTCNDenoiser(correction_delay=1)
        features=forecast_features(flat[:,:,None],{111:model},baseline,IdentityPCA(),p)
        np.testing.assert_array_equal(features['observed_input'],flat[:,-5:-1])
        np.testing.assert_array_equal(features['residual_tcn_111'],flat[:,-5:-1])
        np.testing.assert_array_equal(features['centered_median'],flat[:,-5:-1])
        np.testing.assert_array_equal(features['delayed_pair'],flat[:,-5:-1])

    def test_delayed_pipeline_forecasts_from_availability_not_repaired_time(self):
        from tsprep.denoising_experiment import run_experiment
        from tsprep.denoising.data import window_inputs
        p=replace(DenoisingProtocol(),length=400,window=48,correction_delay=1,epochs=2,
            validation_every=1,smoke=True,development_seeds=(9001,9003),validation_seeds=(9101,9103),
            confirmation_seeds=(93001,93002),guard_scope='family',families=(FAMILIES[0],),
            initialization_seeds=(111,),learning_rates=(.001,),losses=('mse_identity',),
            identity_weights=(.1,),bootstrap_repeats=100,transfer_checks=True)
        for mode in ('past','next'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
                out=Path(directory)/'experiment';run_experiment(out,replace(p,delay_mode=mode))
                norm=TrainFittedNormalizer.from_state(read_json(out/'normalizer.json'))
                series=generate_denoising_series(FAMILIES[0],93001,p.length)
                segment=trajectory_segments(series,p,norm,development=False)['test']
                probes=read_json(out/'forecast_probes.json')
                self.assertIn('raw_available',probes['probes'])
                with np.load(out/'predictions'/f'{FAMILIES[0]}_93001.npz') as saved:
                    repaired,available=saved['endpoint_index'],saved['availability_index']
                    np.testing.assert_array_equal(repaired+1,available)
                    np.testing.assert_array_equal(saved['clean'],segment['clean'][repaired,0])
                    np.testing.assert_array_equal(saved['observed_input'],segment['observed'][repaired,0])
                    np.testing.assert_array_equal(saved['anomaly'],segment['anomaly'][repaired])
                    np.testing.assert_array_equal(saved['forecast_origin_index'],available[:-p.forecast_horizon])
                    np.testing.assert_array_equal(saved['forecast_target'],segment['clean'][available[:-p.forecast_horizon]+p.forecast_horizon,0])
                    windows=window_inputs(segment,p)
                    raw=_probe_predict(probes['probes']['raw_available'],windows[:-p.forecast_horizon,-p.forecast_features:,0])
                    np.testing.assert_array_equal(raw,saved['forecast_raw_available'])
                for name,expected in read_json(out/'complete.json')['files'].items():
                    self.assertEqual(file_hash(out/name),expected)

    def test_local_context_identity_active_causality_and_gradients(self):
        torch.manual_seed(17)
        model=ResidualTCNDenoiser(n_features=2,hidden_dim=4,levels=2,local_context=8)
        x=torch.randn(3,64,2)
        self.assertTrue(torch.equal(model(x),x))
        torch.nn.init.normal_(model.error_head.weight,std=.1)
        y=x.clone();y[:,40:]+=30
        self.assertTrue(torch.equal(model(x)[:,:40],model(y)[:,:40]))
        model(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_local_context_protocol_and_legacy_weights(self):
        with self.assertRaises(ValueError):replace(DenoisingProtocol(),local_context=16).validate()
        replace(DenoisingProtocol(),local_context=16,window=64).validate()
        with self.assertRaises(ValueError):replace(DenoisingProtocol(),local_context=-1).validate()
        torch.manual_seed(31);a=ResidualTCNDenoiser()
        torch.manual_seed(31);b=ResidualTCNDenoiser(local_context=0)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        b.load_state_dict(a.state_dict())

    def test_local_context_control_has_same_parameters_and_initial_weights(self):
        torch.manual_seed(111);a=ResidualTCNDenoiser(local_context=16)
        torch.manual_seed(111);b=ResidualTCNDenoiser(local_context=16,context_mode='repeat')
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        with self.assertRaises(ValueError):ResidualTCNDenoiser(context_mode='repeat')
        x=torch.randn(2,64,1)
        self.assertTrue(torch.equal(b(x),x))
        torch.nn.init.normal_(b.error_head.weight,std=.1)
        y=x.clone();y[:,40:]+=30
        self.assertTrue(torch.equal(b(x)[:,:40],b(y)[:,:40]))

    def test_local_context_complete_pipeline_and_reload(self):
        from tsprep.denoising_experiment import run_experiment
        from tsprep.denoising.training import build_denoiser,predict
        from tsprep.denoising.data import window_inputs
        p=replace(DenoisingProtocol(),length=400,window=64,local_context=16,epochs=2,
            validation_every=1,smoke=True,development_seeds=(9001,9003),validation_seeds=(9101,9103),
            confirmation_seeds=(93001,93002),guard_scope='family',families=(FAMILIES[0],),
            initialization_seeds=(111,),learning_rates=(.001,),losses=('mse_identity',),
            identity_weights=(.1,),bootstrap_repeats=100,transfer_checks=True)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)/'experiment';run_experiment(out,p)
            selected=read_json(out/'selection.json')['selected']
            model=build_denoiser(p,111)
            model.load_state_dict(torch.load(out/selected['path']/'model.pt',weights_only=True))
            norm=TrainFittedNormalizer.from_state(read_json(out/'normalizer.json'))
            series=generate_denoising_series(FAMILIES[0],93001,p.length)
            segment=trajectory_segments(series,p,norm,development=False)['test']
            result=predict(model,window_inputs(segment,p),p.device,p.batch_size)[:,-1,0]
            with np.load(out/'predictions'/f'{FAMILIES[0]}_93001.npz') as saved:
                np.testing.assert_array_equal(result,saved['residual_tcn_111'])
            for name,expected in read_json(out/'complete.json')['files'].items():
                self.assertEqual(file_hash(out/name),expected)

    def test_identity_loss_value_gradients_and_zero_weight(self):
        prediction = torch.tensor([3., 1.], requires_grad=True)
        clean_prediction = torch.tensor([2., 4.], requires_grad=True)
        target = torch.tensor([1., 2.])
        loss = residual_loss(prediction, target, "mse_identity", clean_prediction=clean_prediction, identity_weight=2.)
        self.assertEqual(loss.item(), 7.5)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([2., -1.]))
        torch.testing.assert_close(clean_prediction.grad, torch.tensor([2., 4.]))
        a = residual_loss(prediction, target, "mse")
        b = residual_loss(prediction, target, "mse_identity", identity_weight=0.)
        self.assertTrue(torch.equal(a, b))
        for weight in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                residual_loss(prediction, target, "mse_identity", identity_weight=weight)
        with self.assertRaises(ValueError):
            residual_loss(prediction, target, "mse_identity", identity_weight=1.)
        with self.assertRaises(ValueError):
            residual_loss(prediction, target, "typo")

    def test_volatility_scale_preserves_observation_error_stream(self):
        base = generate_denoising_series(FAMILIES[3], 9002, 500)
        scaled = generate_denoising_series(FAMILIES[3], 9002, 500, volatility_scale=2.)
        np.testing.assert_allclose(np.log(scaled.clean_values/100), 2*np.log(base.clean_values/100), atol=1e-14)
        np.testing.assert_allclose(np.log(scaled.values/scaled.clean_values),np.log(base.values/base.clean_values),atol=1e-14)
        np.testing.assert_array_equal(scaled.is_anomaly,base.is_anomaly)
        for value in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                generate_denoising_series(FAMILIES[0],9002,500,volatility_scale=value)

    def test_independent_validation_never_fits_normalizer(self):
        from tsprep.denoising.data import prepare_development
        p = replace(DenoisingProtocol(),length=400,development_seeds=(9001,9003),
                    validation_seeds=(9101,9103),volatility_scales=(.5,2.),families=(FAMILIES[0],),smoke=True)
        p.validate()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            development,norm,_=prepare_development(p,out)
            train=[]
            for i,item in enumerate(development):
                with np.load(out/'inputs'/f'development_{FAMILIES[0]}_{item["seed"]}.npz') as a:
                    train.append(np.diff(np.log(a['observed']),axis=0)[:279])
                with np.load(out/'inputs'/f'validation_{FAMILIES[0]}_{p.validation_seeds[i]}.npz') as a:
                    expected=norm.transform(np.diff(np.log(a['observed']),axis=0)[279:]).astype(np.float32)
                np.testing.assert_array_equal(expected,item['val']['observed'])
                self.assertEqual(item['validation_seed'],p.validation_seeds[i])
            np.testing.assert_array_equal(norm.mean_,np.concatenate(train).mean(axis=0))
            np.testing.assert_array_equal(norm.std_,np.concatenate(train).std(axis=0))
        for bad in [replace(p,validation_seeds=(9001,9103)),replace(p,validation_seeds=(9101,)),
                    replace(p,volatility_scales=(.5,1.,2.)),
                    replace(p,guard_scope='family',validation_upper_quantile=.99)]:
            with self.assertRaises(ValueError):bad.validate()

    def test_validation_confidence_and_volatility_strata(self):
        from tsprep.denoising.metrics import validation_guard
        p=replace(DenoisingProtocol(),families=(FAMILIES[0],),development_seeds=(1,2,3,4),
                  volatility_scales=(.5,2.),guard_scope='family',validation_bootstrap_repeats=200)
        panel=[dict(family=FAMILIES[0],volatility_scale=s) for s in (.5,2.,.5,2.)]
        values=np.array([[.5,.7],[.5,.9],[.5,.7],[.5,1.15]])
        self.assertAlmostEqual(validation_guard(values,panel,p),1.025)
        conservative=replace(p,validation_upper_quantile=.99)
        self.assertGreater(validation_guard(values,panel,conservative),p.guard_limit)
        self.assertEqual(validation_guard(values,panel,conservative),validation_guard(values,panel,conservative))
        # Low-volatility safety cannot hide high-volatility damage.
        values[:,1]=[.1,1.2,.1,1.2]
        self.assertAlmostEqual(validation_guard(values,panel,p),1.2)
        with self.assertRaises(ValueError):validation_guard(values[:2],panel,p)

    def test_zero_identity_weight_reproduces_training(self):
        from tsprep.denoising.data import prepare_development
        from tsprep.denoising.training import fit_candidate
        p = replace(DenoisingProtocol(), length=400, epochs=2, validation_every=1,
                    development_seeds=(9001,), families=(FAMILIES[0],), smoke=True)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)
            development, _, _ = prepare_development(p, out)
            a, ra = fit_candidate(p, development, "mse", .001, 111, out/"mse")
            b, rb = fit_candidate(p, development, "mse_identity", .001, 111, out/"identity", identity_weight=0.)
            self.assertEqual(ra["history"], rb["history"])
            self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    def test_identity_penalty_reduces_unnecessary_correction(self):
        # A learnable constant correction cannot distinguish signal and errors.
        # The penalty must move its optimum from 1 to 1/(1+lambda).
        corrections = []
        for weight in (0., 4.):
            correction = torch.nn.Parameter(torch.tensor(0.))
            optimizer = torch.optim.SGD([correction], lr=.05)
            for _ in range(100):
                loss = residual_loss(torch.ones(8)-correction, torch.zeros(8), "mse_identity",
                                     clean_prediction=torch.zeros(8)-correction, identity_weight=weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            corrections.append(correction.item())
        self.assertAlmostEqual(corrections[0], 1., places=4)
        self.assertAlmostEqual(corrections[1], .2, places=4)

    def test_transfer_preserves_dynamics_and_marks_persistent_errors(self):
        from tsprep.denoising.confirmation import TRANSFER_CONDITIONS
        base = generate_denoising_series(FAMILIES[0], 9002, 500)
        for settings in TRANSFER_CONDITIONS.values():
            series = generate_denoising_series(FAMILIES[0], 9002, 500, **settings)
            np.testing.assert_array_equal(series.clean_values, base.clean_values)
        isolated = generate_denoising_series(FAMILIES[0], 9002, 500, measurement_std=0)
        clustered = generate_denoising_series(FAMILIES[0], 9002, 500, measurement_std=0, impulse_duration=4)
        expected = np.convolve(isolated.is_anomaly.astype(int), np.ones(4), mode="full")[:500] > 0
        np.testing.assert_array_equal(clustered.is_anomaly, expected)
        impulse = np.log(isolated.values[:, 0]/isolated.clean_values[:, 0])
        actual = np.log(clustered.values[:, 0]/clustered.clean_values[:, 0])
        np.testing.assert_allclose(actual, np.convolve(impulse, np.ones(4), mode="full")[:500], atol=1e-14)
        with self.assertRaises(ValueError):
            generate_denoising_series(FAMILIES[0], 9002, 500, impulse_probability=2.)

    def test_family_guard_rejects_damage_hidden_by_average(self):
        p = replace(DenoisingProtocol(), families=FAMILIES[:2], confirmation_seeds=(1, 2), bootstrap_repeats=100)
        rows = []
        for family, ratio in zip(p.families, (.8, 1.2)):
            for seed in p.confirmation_seeds:
                rows.append(dict(family=family, data_seed=seed, model="observed_input", all_mae=1., normal_mse=1.))
                rows.append(dict(family=family, data_seed=seed, model="residual_tcn", all_mae=.8, normal_mse=ratio))
        frame = pd.DataFrame(rows)
        self.assertTrue(_decision(frame, p, "observed_input")["methods"]["residual_tcn"]["joint_success"])
        strict = _decision(frame, replace(p, guard_scope="family"), "observed_input")["methods"]["residual_tcn"]
        self.assertFalse(strict["joint_success"])
        self.assertAlmostEqual(strict["family_normal_mse"][p.families[1]]["mean"], 1.2)

    def test_large_clean_moves_do_not_depend_on_prediction(self):
        a = endpoint_metrics([0., 0., 0.], [1., 5., -6.], [False, False, True], large_threshold=4.)
        self.assertEqual(a["large_count"], 2)
        self.assertEqual(a["large_normal_count"], 1)
        self.assertEqual(a["large_normal_mse"], 25.)

    def test_preservation_pipeline_and_frozen_transfer(self):
        from tsprep.denoising_experiment import run_experiment
        from tsprep.denoising.confirmation import TRANSFER_CONDITIONS
        p = replace(DenoisingProtocol(), protocol_id="preservation_smoke", length=400, epochs=2,
                    validation_every=1, smoke=True, development_seeds=(9001,), confirmation_seeds=(93001, 93002),
                    initialization_seeds=(111,), families=(FAMILIES[0],), learning_rates=(.001,),
                    losses=("mse_identity",), identity_weights=(.1, 1.), guard_scope="family",
                    transfer_checks=True, bootstrap_repeats=100)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/"experiment"
            decision = run_experiment(out, p)
            self.assertFalse(decision["methods"]["residual_tcn"]["joint_success"])
            selection = read_json(out/"selection.json")
            self.assertEqual({r["identity_weight"] for r in selection["candidates"]}, {.1, 1.})
            source = np.load(out/"inputs"/f"development_{FAMILIES[0]}_9001.npz")
            norm = TrainFittedNormalizer.from_state(read_json(out/"normalizer.json"))
            clean = norm.transform(np.diff(np.log(source["clean"]), axis=0)[:279]).astype(np.float32)
            expected = np.quantile(np.abs(clean[:, 0]), p.large_move_quantile)
            self.assertEqual(selection["baseline"]["large_move_thresholds"][FAMILIES[0]], expected)
            source.close()
            frame = pd.read_csv(out/"transfer_metrics.csv")
            self.assertEqual(set(frame.condition), set(TRANSFER_CONDITIONS))
            raw = frame[(frame.condition == "clean") & (frame.model == "observed_input")]
            self.assertTrue((raw.all_mse == 0).all())
            self.assertTrue((raw.contaminated_count == 0).all())
            for name, expected in read_json(out/"complete.json")["files"].items():
                self.assertEqual(file_hash(out/name), expected)

    def test_independent_validation_pipeline_freezes_stratified_choice(self):
        from tsprep.denoising_experiment import run_experiment
        p=replace(DenoisingProtocol(),length=400,epochs=2,validation_every=1,smoke=True,
            development_seeds=(9001,9003,9004,9005),validation_seeds=(9101,9103,9104,9105),
            confirmation_seeds=(93001,93002),volatility_scales=(.5,2.),validation_upper_quantile=.99,
            validation_bootstrap_repeats=100,guard_scope='family',families=(FAMILIES[0],),
            initialization_seeds=(111,),learning_rates=(.001,),losses=('mse_identity',),
            identity_weights=(.1,),bootstrap_repeats=100,transfer_checks=True)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)/'experiment'
            run_experiment(out,p)
            panel=read_json(out/'validation_panel.json')
            self.assertEqual({x['validation_seed'] for x in panel},set(p.validation_seeds))
            selected=read_json(out/'selection.json')['selected']
            self.assertEqual(selected['validation_upper_quantile'],.99)
            self.assertEqual(len(selected['history'][-1]['validation_trajectories']),4)
            frame=pd.read_csv(out/'transfer_metrics.csv')
            self.assertIn('volatility_2',set(frame.condition))
            self.assertIn('clean_volatility_0.5',set(frame.condition))
            for name,expected in read_json(out/'complete.json')['files'].items():
                self.assertEqual(file_hash(out/name),expected)

    def test_family_reproducibility_and_error_free_control(self):
        for family in FAMILIES:
            a = generate_denoising_series(family, 9002, 500)
            b = generate_denoising_series(family, 9002, 500)
            control = generate_denoising_series(family, 9002, 500, contaminated=False)
            np.testing.assert_array_equal(a.values, b.values)
            np.testing.assert_array_equal(a.clean_values, control.values)
            self.assertFalse(control.is_anomaly.any())
            self.assertTrue((a.values > 0).all())
            self.assertFalse(np.array_equal(a.values, a.clean_values))

    def test_difference_masks_and_segments_do_not_cross_split(self):
        p = DenoisingProtocol(length=500)
        s = generate_denoising_series(FAMILIES[0], 9002, 500)
        # A single price error creates two contaminated returns.
        s.is_anomaly[:] = False
        s.is_anomaly[200] = True
        norm = TrainFittedNormalizer().fit(np.diff(np.log(s.values), axis=0)[:349])
        segments = trajectory_segments(s, p, norm, development=True)
        self.assertEqual(np.flatnonzero(segments["train"]["anomaly"]).tolist(), [199, 200])
        self.assertEqual(segments["val"]["start"], 349)
        self.assertEqual(len(segments["train"]["observed"]), 349)
        self.assertEqual(len(segments["val"]["observed"]), 150)
        expected = norm.transform(np.diff(np.log(s.clean_values), axis=0))[349:]
        np.testing.assert_allclose(segments["val"]["clean"], expected, rtol=1e-6)

    def test_endpoint_counts_and_undefined_ratios(self):
        m = endpoint_metrics([0., 2., 1.], [0., 0., 0.], [False, True, False])
        self.assertEqual(m["all_count"], 3)
        self.assertEqual(m["normal_mse"], .5)
        self.assertEqual(m["contaminated_mse"], 4.)
        self.assertEqual(relative_scores(m, m), [1., 1.])
        with self.assertRaises(ValueError):
            relative_scores(m, dict(all_mae=0., normal_mse=0.))

    def test_trajectory_bootstrap_and_joint_guard(self):
        interval = trajectory_interval(np.full((4, 3), .8), 200)
        self.assertAlmostEqual(interval["upper"], .8)
        with self.assertRaises(ValueError):
            trajectory_interval(np.ones((4, 1)), 200)
        p = replace(DenoisingProtocol(), confirmation_seeds=(1, 2), bootstrap_repeats=200)
        rows = []
        for family in p.families:
            for seed in p.confirmation_seeds:
                rows.append(dict(family=family, data_seed=seed, model="observed_input", all_mae=1., normal_mse=1.))
                # Multiple initializations are averaged within a trajectory, not independent units.
                for _ in range(3):
                    rows.append(dict(family=family, data_seed=seed, model="residual_tcn", all_mae=.8, normal_mse=1.2))
        decision = _decision(pd.DataFrame(rows), p, "observed_input")
        self.assertFalse(decision["methods"]["residual_tcn"]["joint_success"])
        self.assertEqual(decision["methods"]["residual_tcn"]["comparisons"]["observed_input"]["all_mae"]["trajectories_per_family"], 2)

    def test_probe_uses_frozen_training_scaler(self):
        train = np.arange(80, dtype=float).reshape(40, 2)
        val = np.arange(80, 120, dtype=float).reshape(20, 2)
        probe = _fit_probe(train, train[:, 0]*.3, val, val[:, 0]*.3)
        np.testing.assert_allclose(probe["mean"], train.mean(axis=0))
        self.assertLess(np.abs(_probe_predict(probe, val)-val[:, 0]*.3).mean(), .1)

    def test_protocol_rejects_overlap_and_short_context(self):
        with self.assertRaises(ValueError):
            replace(DenoisingProtocol(), confirmation_seeds=(12001,)).validate()
        with self.assertRaises(ValueError):
            replace(DenoisingProtocol(), window=24).validate()

    def test_tensor_batches_keep_targets_and_paired_shuffle(self):
        x = np.arange(36, dtype=np.float32).reshape(9, 4, 1)
        loaders = [TensorWindowLoader(x, 4, targets=-x, shuffle=True, seed=317) for _ in range(2)]
        a, b = list(loaders[0]), list(loaders[1])
        self.assertEqual([len(v[0]) for v in a], [4, 4, 1])
        for (ax, ay), (bx, by) in zip(a, b):
            self.assertTrue(torch.equal(ax, bx))
            self.assertTrue(torch.equal(ay, -ax))
            self.assertTrue(torch.equal(ay, by))
        self.assertEqual(sorted(torch.cat([v[0][:, 0, 0] for v in a]).tolist()), x[:, 0, 0].tolist())

    def test_representation_horizon_and_stress_keep_targets_aligned(self):
        p = replace(RepresentationProtocol(), window=4, forecast_horizon=2)
        clean = np.arange(10, dtype=np.float32)[:, None]
        segment = dict(clean=clean, observed=clean+3)
        x, ends, y = representation_windows(segment, p)
        np.testing.assert_array_equal(ends, np.arange(3, 8))
        np.testing.assert_array_equal(y, clean[ends+2, 0])
        np.testing.assert_array_equal(x[:, -1, 0], clean[ends, 0]+3)
        for severity in (0, 2):
            windows, other_ends, other_y = representation_windows(segment, p, severity=severity)
            np.testing.assert_array_equal(other_y, y)
            np.testing.assert_array_equal(other_ends, ends)
            np.testing.assert_array_equal(windows[:, -1, 0], clean[ends, 0]+3*severity)

    def test_representation_joint_comparisons_and_missing_grid(self):
        p = replace(RepresentationProtocol(), confirmation_seeds=(1, 2), bootstrap_repeats=200)
        rows = []
        for family in p.families:
            for seed in p.confirmation_seeds:
                for severity in (0, 1, 2):
                    for method in ("raw", "pca", "mad_pca", "clean_oracle", "mean", "median", "ordinary", "denoising"):
                        for init in p.initialization_seeds if method in ("ordinary", "denoising") else (-1,):
                            rows.append(dict(family=family, data_seed=seed, severity=severity, model=method,
                                initialization_seed=init, mae=.8 if method == "denoising" else 1., mse=1., r2=0., count=100))
        frame = pd.DataFrame(rows)
        decision = representation_decision(frame, p)
        self.assertTrue(decision["joint_success"])
        self.assertAlmostEqual(decision["upper_quantile"], 1-.05/3)
        frame.loc[frame.model == "mad_pca", "mae"] = .7
        self.assertFalse(representation_decision(frame, p)["joint_success"])
        with self.assertRaisesRegex(ValueError, "grid"):
            representation_decision(frame.iloc[:-1], p)
        with self.assertRaises(ValueError):
            trajectory_interval(np.ones((4, 3)), upper_quantile=1.)

    def test_representation_protocol_rejects_seed_leak_and_short_segments(self):
        with self.assertRaises(ValueError):
            replace(RepresentationProtocol(), confirmation_seeds=(16001, 16002)).validate()
        with self.assertRaises(ValueError):
            replace(RepresentationProtocol(), length=100).validate()

    def test_representation_roundtrip_uses_z_and_frozen_receipts(self):
        from tsprep.representation_experiment import run_experiment
        p = replace(RepresentationProtocol(), length=400, epochs=1, smoke=True,
            development_seeds=(9001,), confirmation_seeds=(97001, 97002),
            initialization_seeds=(111,), learning_rates=(.001,), mad_thresholds=(5.,),
            bootstrap_repeats=100, families=(FAMILIES[0],))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/"experiment"
            decision = run_experiment(out, p)
            self.assertFalse(decision["joint_success"])
            receipt = read_json(out/"complete.json")
            self.assertEqual(receipt["status"], "smoke_complete")
            for name, expected in receipt["files"].items():
                self.assertEqual(file_hash(out/name), expected)
            selected = read_json(out/"selection.json")["final_models"]
            self.assertEqual(selected["ordinary_seed111"]["initial_state_sha256"], selected["denoising_seed111"]["initial_state_sha256"])
            with np.load(next((out/"predictions").glob('*severity1.npz'))) as a:
                z = a["features_denoising_seed111"]
                self.assertEqual(z.shape, (p.length-p.window-p.forecast_horizon, p.latent_dim))
                probe = selected["denoising_seed111"]["probe"]
                pred = ((z-np.asarray(probe["mean"]))/np.asarray(probe["std"])) @ probe["coef"] + probe["intercept"]
                np.testing.assert_array_equal(pred, a["prediction_denoising_seed111"])


if __name__ == "__main__":
    unittest.main()
