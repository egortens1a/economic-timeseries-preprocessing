"""
CLI: полный прогон одного эксперимента "от данных до метрик".

Пример:
    python scripts/run_experiment.py --model hybrid_vae --epochs 40
    python scripts/run_experiment.py --model tcn_ae --data csv --csv-path data/raw/spx.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tsprep.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from tsprep.data_preprocessing.datasets import chronological_split_indices, sliding_windows
from tsprep.data_preprocessing.datasets import build_dataloaders, build_eval_loader, prepare_and_window
from tsprep.data_preprocessing.loaders import load_csv_series
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.evaluation.downstream import (anomaly_detection_f1, clustering_quality,
                                           forecast_via_linear_probe, window_anomaly_labels)
from tsprep.evaluation.latent_space import information_collapse_ratio
from tsprep.evaluation.reconstruction import collect_embeddings, reconstruction_report
from tsprep.models.autoencoder import build_model
from tsprep.mlcore.trainer import train_model


def run(exp: ExperimentConfig, raw: np.ndarray, regime=None, is_anomaly=None, verbose=True) -> dict:
    torch.manual_seed(exp.train.seed)
    np.random.seed(exp.train.seed)

    prepared = prepare_and_window(raw, exp.data, regime=regime, is_anomaly=is_anomaly)
    train_loader, val_loader, test_loader = build_dataloaders(prepared, exp.data)

    model = build_model(exp.model, n_features=exp.data.n_features, window_len=exp.data.window_len)
    history = train_model(model, train_loader, val_loader, exp.train, verbose=verbose)

    test_out = collect_embeddings(model, test_loader, device=exp.train.device)
    recon = reconstruction_report(test_out["x"], test_out["x_hat"])
    eta = information_collapse_ratio(test_out["z"], test_out["x"])

    result = {"model": exp.model.model_name, **recon, "eta_collapse": eta,
              "n_train_windows": prepared.train_windows.shape[0],
              "n_test_windows": prepared.test_windows.shape[0],
              "epochs_ran": len(history.train_loss)}

    # downstream: forecasting (target = среднее значение по признакам на шаге t+H в норм. шкале)
    h = exp.data.forecast_horizon
    
    train_eval_loader = build_eval_loader(prepared.train_windows, exp.data.batch_size)
    train_out = collect_embeddings(model, train_eval_loader, device=exp.train.device)

    step = max(1, h // max(1, exp.data.stride))
    n_train = train_out["z"].shape[0] - step
    n_test = test_out["z"].shape[0] - step
    if n_train > 10 and n_test > 10:
        z_tr, y_tr = train_out["z"][:n_train], train_out["x"][step:step + n_train, -1, 0]
        z_te, y_te = test_out["z"][:n_test], test_out["x"][step:step + n_test, -1, 0]
        result["forecast"] = forecast_via_linear_probe(z_tr, y_tr, z_te, y_te)

    if prepared.regime is not None:
        n = prepared.raw_normalized.shape[0] # type: ignore
        train_end, val_end = chronological_split_indices(n, exp.data.train_frac, exp.data.val_frac, exp.data.test_frac)
        regime_windows = sliding_windows(prepared.regime[val_end:].reshape(-1, 1), exp.data.window_len, exp.data.stride)
        regime_label = regime_windows[:, -1, 0].astype(int)
        result["clustering"] = clustering_quality(test_out["z"], labels=regime_label)
    else:
        result["clustering"] = clustering_quality(test_out["z"], labels=None)

    if prepared.is_anomaly is not None:
        n = prepared.raw_normalized.shape[0] # type: ignore
        train_end, val_end = chronological_split_indices(n, exp.data.train_frac, exp.data.val_frac, exp.data.test_frac)
        n_test_windows = test_out["x"].shape[0]
        anomaly_labels = window_anomaly_labels(prepared.is_anomaly[val_end:], exp.data.window_len,
                                                exp.data.stride, n_test_windows)
        result["anomaly"] = anomaly_detection_f1(test_out["x"], test_out["x_hat"], anomaly_labels)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hybrid_vae",
                         choices=["tcn_ae", "tcn_vae", "lstm_ae", "lstm_vae", "transformer_ae", "hybrid_vae"])
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "csv"])
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--log-transform", action="store_true",
                         help="Логарифмировать перед дифференцированием (нужно для многолетних "
                              "абсолютных ценовых рядов - см. tsprep.data.preprocessing.prepare_series)")
    parser.add_argument("--length", type=int, default=4000, help="length of synthetic series")
    parser.add_argument("--out", default="results/single_run.json")
    args = parser.parse_args()

    data_cfg = DataConfig(window_len=args.window_len, log_transform=args.log_transform)
    model_cfg = ModelConfig(model_name=args.model)
    train_cfg = TrainConfig(epochs=args.epochs)
    exp = ExperimentConfig(data=data_cfg, model=model_cfg, train=train_cfg)

    if args.data == "synthetic":
        series = generate_series(length=args.length, n_features=data_cfg.n_features, seed=data_cfg.seed)
        raw, regime, is_anomaly = series.values, series.regime, series.is_anomaly
    else:
        assert args.csv_path, "--csv-path is required when --data csv"
        raw = load_csv_series(args.csv_path)
        data_cfg.n_features = raw.shape[1]
        regime, is_anomaly = None, None

    result = run(exp, raw, regime=regime, is_anomaly=is_anomaly)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()