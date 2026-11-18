"""Сравнение эмбеддингов обычного AE и AE с удалением шума на новых траекториях."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from threadpoolctl import threadpool_limits

from tsprep.config import RepresentationProtocol
from tsprep.denoising.baselines import fit_representation_baselines
from tsprep.denoising.confirmation import evaluate_embeddings
from tsprep.denoising.data import prepare_development
from tsprep.denoising.training import check_embedding_learning, select_embedding_models
from tsprep.mlcore.trainer import seed_everything
from tsprep.provenance import (file_hash, runtime_environment, source_hashes,
    source_snapshot, value_hash, write_json)


def run_experiment(out, p):
    p.validate()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(2)
    seed_everything(111, deterministic=True)
    hashes = source_hashes()
    write_json(out/"protocol.json", dict(**asdict(p), source_hashes=hashes,
        scope="Synthetic observation errors; compact frozen representations with a linear probe",
        model="TCN-AE, kernel3, dropout0, matmul upsampling; paired initialization and shuffle",
        train_loss="window MSE, observed or clean target", checkpoint="minimum validation clean window MSE",
        selection="common LR minimizes equal-arm/trajectory/family validation MAE/raw Ridge",
        target="one clean standardized log return exactly at endpoint+H, not cumulative return",
        primary_comparators=["ordinary", "pca", "mad_pca"], upper_quantile=1-.05/3,
        stress="same clean dynamics; clean + s*(observed-clean) in transformed units; s=0,1,2"))
 
    source_snapshot(out/"source.zip")
    write_json(out/"environment.json", runtime_environment(p.device))
    check_embedding_learning(p, out)
    development, normalizer, fingerprints = prepare_development(p, out)
    baseline = fit_representation_baselines(development, p, out)
    models, selection = select_embedding_models(p, development, baseline, out)
    frozen_files = ["protocol.json", "normalizer.json", "representation_baselines.json", "representation_pca.npz"]
    for receipt in selection["final_models"].values():
        frozen_files.extend([receipt["path"]+"/model.pt", receipt["path"]+"/training.json",
                             receipt["path"]+"/embeddings.npz"])
 
    selection.update(frozen_before_confirmation=True, input_fingerprints=dict(fingerprints),
                     files={name: file_hash(out/name) for name in frozen_files})
  
    write_json(out/"selection.json", selection)
    selection_hash = file_hash(out/"selection.json")
    print(f"SELECTION FROZEN: common LR={selection['learning_rate']}", flush=True)
    if hashes != source_hashes():
        raise RuntimeError("Sources changed during development; confirmation cancelled")
  
    decision = evaluate_embeddings(p, normalizer, models, baseline, selection, out, fingerprints)
    if hashes != source_hashes() or file_hash(out/"selection.json") != selection_hash:
        raise RuntimeError("Sources or selection changed; completion refused")
 
    if any(file_hash(out/name) != expected for name, expected in selection["files"].items()):
        raise RuntimeError("Frozen model/preprocessing receipt mismatch")
  
    files = {f.relative_to(out).as_posix(): file_hash(f) for f in sorted(out.rglob("*")) if f.is_file()}
    write_json(out/"complete.json", dict(status="smoke_complete" if p.smoke else "complete",
        seconds=time.perf_counter()-begin, protocol_hash=value_hash(asdict(p)),
        selection_sha256=selection_hash, files=files))
 
    print(f"COMPLETE {out}: joint_success={decision['joint_success']}", flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Короткий запуск для проверки кода")
    args = parser.parse_args()
    p = RepresentationProtocol(device=args.device)
    if args.smoke:
        p.length, p.epochs, p.smoke = 500, 2, True
        p.development_seeds, p.confirmation_seeds = (9001,), (97001, 97002)
        p.initialization_seeds, p.learning_rates = (111,), (.001,)
        p.bootstrap_repeats = 200
    with threadpool_limits(limits=2):
        run_experiment(args.out, p)


if __name__ == "__main__":
    main()
