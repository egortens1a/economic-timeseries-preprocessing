"""Compare repeated training runs on CPU and CUDA."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tsprep.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from tsprep.data_preprocessing.synthetic import generate_series
from tsprep.experiment import train_experiment, load_artifact
from tsprep.provenance import write_json, source_hashes, runtime_environment, read_json


def check(target, summary, epochs=8):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    data = generate_series(length=800, seed=9001)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for device in devices:
        for name in ("tcn_ae", "tcn_vae", "hybrid_ae", "hybrid_vae"):
            weights, predictions, histories = [], [], []
            for repeat in range(2):
                exp = ExperimentConfig(DataConfig(window_len=24, stride=2, batch_size=64, seed=317),
                    ModelConfig(model_name=name, hidden_dim=16, d_model=16, latent_dim=8,
                                tcn_levels=2, n_heads=2, n_attn_layers=1),
                    TrainConfig(epochs=epochs, kl_anneal_epochs=2, beta_target=.001,
                                seed=317, device=device, deterministic=True))
                path = target/f"{device}_{name}_repeat{repeat}"
                train_experiment(exp, data.values, output_dir=path, regime=data.regime,
                    is_anomaly=data.is_anomaly, clean_raw=data.clean_values, evaluate=False, verbose=False)
                model, prepared, _, manifest, _ = load_artifact(path, device)
                weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                with torch.no_grad():
                    predictions.append(model(torch.tensor(prepared.test_windows[:8], dtype=torch.float32, device=device))[0].cpu())
                histories.append(manifest["history"]["val_recon"])
            difference = max(float((weights[0][k]-weights[1][k]).abs().max()) for k in weights[0])
            row = dict(device=device, model=name, epochs=epochs, weights_bitwise_equal=all(
                torch.equal(weights[0][k], weights[1][k]) for k in weights[0]),
                maximum_weight_difference=difference,
                maximum_prediction_difference=float((predictions[0]-predictions[1]).abs().max()),
                validation_history_equal=histories[0] == histories[1])
            rows.append(row)
            print(row, flush=True)
    result = dict(source_hashes=source_hashes(), debug_data_seed=9001, model_seed=317, loader_seed=317,
        runs=rows, environment=runtime_environment(devices[-1]),
        all_equal=all(r["weights_bitwise_equal"] and r["maximum_prediction_difference"] == 0 and
                      r["validation_history_equal"] for r in rows),
        scope="same process/backend/environment only; no cross-device bitwise promise; no quality conclusions")
    write_json(target/"summary.json", result)
    write_json(summary, result)
    if not result["all_equal"]:
        raise RuntimeError("Paired strict training controls did not reproduce")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        check(args.out, args.summary, args.epochs)
