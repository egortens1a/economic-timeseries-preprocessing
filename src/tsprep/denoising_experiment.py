"""Эксперимент по исправлению ошибок наблюдения: подготовка, обучение и оценка.

CLI: python -m tsprep.denoising_experiment --out results/residual_denoising
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

import torch
from threadpoolctl import threadpool_limits

from tsprep.denoising.baselines import tune_baselines
from tsprep.denoising.confirmation import evaluate_confirmation, transfer_conditions
from tsprep.denoising.data import prepare_development, trajectory_segments
from tsprep.denoising.forecast import fit_forecasts
from tsprep.denoising.protocol import DenoisingProtocol
from tsprep.denoising.training import overfit_check, save_ordinary_control, select_models
from tsprep.mlcore.trainer import seed_everything
from tsprep.provenance import (file_hash, runtime_environment, source_hashes,
    source_snapshot, value_hash, write_json)

# Эти имена используются в существующих скриптах анализа.
from tsprep.denoising.forecast import fit_probe as _fit_probe, predict_probe as _probe_predict
from tsprep.denoising.metrics import (confirmation_decision as _decision, endpoint_metrics,
    relative_scores, trajectory_interval)


def run_experiment(out, p):
    p.validate()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(2)
    seed_everything(111, deterministic=True)
    hashes = source_hashes()
    write_json(out/"protocol.json", dict(**asdict(p),
        scope="observation errors on synthetic log differences; no claims about real markets",
        checkpoint=f"lowest validation MAE/input with normal MSE/input <= guard_limit ({p.guard_scope}); identity fallback",
        search="loss x identity weight (mse_identity only) x learning rate, seed 111; full budget, no early stopping",
        identity_objective="endpoint MSE(f(observed), clean) + lambda * endpoint MSE(f(clean), clean)",
        timing=("at availability s repair s-1; forecast clean return at s+H, raw_available includes s"
                if p.correction_delay else "repair current endpoint; forecast endpoint+H"),
        delay_baselines=("centered median3; pair filter past16 median/MAD, thresholds3/5/7/9, opposite signs, cancellation<=3 scales"
                         if p.correction_delay else None),
        model_inputs=(f"past stream x[s-1] and {'current x[s]' if p.delay_mode == 'next' else 'a copy of past x[s-1]'}"
                      if p.correction_delay else
                      "four copies of the observed input (equal-parameter control)" if p.local_context and p.context_mode == "repeat" else
                      "observed, past lower-median, log past MAD scale, clipped local deviation; "
                      "scale=max(1.4826*MAD, .001), deviation clipped to [-20,20]; past excludes current point"
                      if p.local_context else "observed normalized log returns"),
        transfer_conditions=transfer_conditions(p) if p.transfer_checks else {},
        validation_sampling=f"independent seeds, final {1-p.train_fraction:.0%} segment" if p.validation_seeds else "development trajectory tail",
        validation_guard="worst family/volatility stratum; upper bootstrap bound is a selection heuristic, not a simultaneous guarantee over checkpoints",
        contamination="Gaussian log measurement noise std .001 and independent 1% signed log impulses U(.04,.08)",
        normal_mask="neither endpoint of the price difference has an impulse",
        uncertainty="stratified paired trajectory bootstrap; conditional on trained models",
        source_hashes=hashes))
 
    source_snapshot(out/"source.zip")
    write_json(out/"environment.json", runtime_environment(p.device))
    overfit_check(p, out)
    development, normalizer, fingerprints = prepare_development(p, out)
   
    if p.validation_seeds or p.volatility_scales != (1.,):
        write_json(out/"validation_panel.json", [dict(family=t["family"], train_seed=t["seed"],
            validation_seed=t["validation_seed"], volatility_scale=t["volatility_scale"],
            train_length=len(t["train"]["observed"]), validation_length=len(t["val"]["observed"])) for t in development])
  
    save_ordinary_control(p, development, out)
    pca, baseline = tune_baselines(development, p, out)
   
    if p.transfer_checks or p.guard_scope == "family":
        baseline["large_move_thresholds"] = {
            family: float(np.quantile(np.abs(np.concatenate([t["train"]["clean"][:, 0]
                          for t in development if t["family"] == family])), p.large_move_quantile))
            for family in p.families}
  
        baseline["large_move_definition"] = "absolute standardized clean return; family train quantile"
        write_json(out/"baseline_selection.json", baseline)
   
    models, selection = select_models(p, development, out)
    forecasts = fit_forecasts(development, models, baseline, pca, p)
    write_json(out/"forecast_probes.json", forecasts)
    selection.update(
        baseline=baseline, fingerprints=fingerprints,
        protocol_sha256=file_hash(out/"protocol.json"),
        normalizer_sha256=file_hash(out/"normalizer.json"),
        forecast_sha256=file_hash(out/"forecast_probes.json"),
        pca_sha256=file_hash(out/"pca.npz"),
        frozen_before_confirmation=True)
 
    write_json(out/"selection.json", selection)
    selection_hash = file_hash(out/"selection.json")
    selected = selection["selected"]
    print(f"SELECTION FROZEN: {selected['loss']} lr={selected['lr']}, baseline={baseline['reference_baseline']}", flush=True)
 
    if hashes != source_hashes():
        raise RuntimeError("Sources changed during development; confirmation cancelled")
  
    decision = evaluate_confirmation(
        p, normalizer, models, baseline, pca, forecasts, out, fingerprints)
   
    if hashes != source_hashes() or selection_hash != file_hash(out/"selection.json"):
        raise RuntimeError("Sources or frozen selection changed; completion refused")
  
    files = {f.relative_to(out).as_posix(): file_hash(f) for f in sorted(out.rglob("*")) if f.is_file()}
    write_json(out/"complete.json", dict(status="smoke_complete" if p.smoke else "complete",
        seconds=time.perf_counter()-begin, selection_sha256=selection_hash, files=files,
        protocol_hash=value_hash(asdict(p))))
   
    print(f"COMPLETE {out}: residual joint success={decision['methods']['residual_tcn']['joint_success']}", flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Короткий запуск для проверки кода")
    parser.add_argument("--loss", choices=("mae", "mse", "mse_identity"), help="Search only this objective")
    parser.add_argument("--identity-weights", type=float, nargs="+", default=(.1, 1., 10.))
    parser.add_argument("--learning-rates", type=float, nargs="+", help="Override the validation search grid")
    parser.add_argument("--family-guard", action="store_true", help="Require preservation in every family")
    parser.add_argument("--transfer-checks", action="store_true", help="Frozen-model clean/error-shift diagnostics")
    parser.add_argument("--development-seeds", type=int, nargs="+")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+")
    parser.add_argument("--validation-seeds", type=int, nargs="+")
    parser.add_argument("--volatility-scales", type=float, nargs="+", default=(1.,))
    parser.add_argument("--validation-upper-quantile", type=float)
    parser.add_argument("--local-context", type=int, default=0, help="Add past robust location/scale features")
    parser.add_argument("--context-mode", choices=("local", "repeat"), default="local")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--correction-delay", type=int, choices=(0, 1), default=0)
    parser.add_argument("--delay-mode", choices=("past", "next"), default="next")
    args = parser.parse_args()
    p = DenoisingProtocol(device=args.device)
    if args.smoke:
        p.length, p.epochs, p.validation_every, p.smoke = 500, 2, 1, True
        p.development_seeds = (9001,)
        p.confirmation_seeds = (93001, 93002)
        p.initialization_seeds = (111,)
        p.learning_rates, p.losses = (.001,), ("mae",)
        p.bootstrap_repeats = 200
    if args.loss:
        p.losses = (args.loss,)
    p.identity_weights = tuple(args.identity_weights)
    if args.learning_rates:
        p.learning_rates = tuple(args.learning_rates)
    p.guard_scope = "family" if args.family_guard else "average"
    p.transfer_checks = args.transfer_checks
    if args.development_seeds:
        p.development_seeds = tuple(args.development_seeds)
    if args.confirmation_seeds:
        p.confirmation_seeds = tuple(args.confirmation_seeds)
    if args.validation_seeds:
        p.validation_seeds = tuple(args.validation_seeds)
    p.volatility_scales = tuple(args.volatility_scales)
    p.validation_upper_quantile = args.validation_upper_quantile
    p.local_context, p.window = args.local_context, args.window
    p.context_mode = args.context_mode
    p.correction_delay, p.delay_mode = args.correction_delay, args.delay_mode
    if args.loss == "mse_identity" or args.family_guard or args.transfer_checks:
        p.protocol_id = "residual_preservation_v1"
    if args.validation_seeds or p.volatility_scales != (1.,) or args.validation_upper_quantile is not None:
        p.protocol_id = "residual_validation_v1"
    if p.local_context:
        p.protocol_id = "residual_local_context_v1"
    if p.correction_delay:
        p.protocol_id = "residual_delay_v1"
    with threadpool_limits(limits=2):
        run_experiment(args.out, p)


if __name__ == "__main__":
    main()
