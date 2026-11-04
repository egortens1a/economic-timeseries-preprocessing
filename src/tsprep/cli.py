"""Command line entry point for immutable experiments and evaluation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import torch
from tsprep.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig, ALL_MODEL_NAMES, ABLATION_MODEL_NAMES
from tsprep.experiment import train_experiment, evaluate_artifact, write_json
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.data_preprocessing.loaders import load_csv_series_with_dates
from tsprep.data_preprocessing.quality import csv_quality_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ALL_MODEL_NAMES+ABLATION_MODEL_NAMES, default="hybrid_vae")
    parser.add_argument("--data", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--csv-path")
    parser.add_argument("--columns", nargs="+", help="Explicit positive price columns, e.g. 'Adj Close'")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--window-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", "--model-seed", dest="seed", type=int, default=42,
                        help="Model initialization seed (--seed is a compatibility alias)")
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--loader-seed", type=int, help="Defaults to model seed")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-source-mismatch", action="store_true",
                        help="Explicit historical reevaluation; does not enable study resume")
    parser.add_argument("--beta", type=float, default=.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--training-target", choices=["observed", "clean"], default="observed")
    parser.add_argument("--validation-target", choices=["observed", "clean"], default="observed")
    parser.add_argument("--selection-metric", choices=["recon", "fixed_beta", "endpoint_mae"], default="recon")
    parser.add_argument("--log-transform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", help="New experiment directory, not a JSON filename")
    parser.add_argument("--evaluate", help="Evaluate an existing v2 artifact without refitting preprocessing")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.evaluate:
        result = evaluate_artifact(args.evaluate, args.device, allow_source_mismatch=args.allow_source_mismatch)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.data == "synthetic":
        s = generate_series(length=args.length, seed=args.data_seed)
        raw, regime, anomaly, clean = s.values, s.regime, s.is_anomaly, s.clean_values
        dates, names, quality = None, ["synthetic_price"], None
    else:
        if not args.csv_path or not args.columns:
            parser.error("CSV experiments require --csv-path and --columns")
        quality = csv_quality_report(args.csv_path)
        if quality["needs_review"]:
            parser.error("CSV quality checks failed; inspect tsprep.data_preprocessing.quality.csv_quality_report")
        dates, raw, names = load_csv_series_with_dates(args.csv_path, value_cols=args.columns)
        regime = anomaly = clean = None
    exp = ExperimentConfig(DataConfig(window_len=args.window_len, stride=args.stride, n_features=raw.shape[1],
        forecast_horizon=args.horizon, log_transform=args.log_transform,
        seed=args.seed if args.loader_seed is None else args.loader_seed),
        ModelConfig(model_name=args.model, hidden_dim=32, d_model=32, tcn_levels=3),
        TrainConfig(epochs=args.epochs, beta_target=args.beta, kl_anneal_epochs=args.warmup,
                    seed=args.seed, device=args.device, deterministic=args.deterministic,
                    training_target=args.training_target, validation_target=args.validation_target,
                    selection_metric=args.selection_metric))
    out = args.out or str(Path("results")/datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ"))
    result = train_experiment(exp, raw, output_dir=out, regime=regime, is_anomaly=anomaly, dates=dates,
                             feature_names=names, clean_raw=clean,
                             metadata={"quality": quality, "data_seed": args.data_seed if args.data == "synthetic" else None})
    print(f"Saved {out}: MSE={result['mse']:.6f}, forecast MAE={result['forecast']['mae']:.6f}")


if __name__ == "__main__":
    main()
