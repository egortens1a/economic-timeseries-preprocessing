"""Atomic persistence, input identity and recorded execution environment."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile

import numpy as np
import pandas as pd
import sklearn
import torch


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def value_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, data):
    """Publish a whole file on the same filesystem; retain old content on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_arrays(raw, *, regime=None, is_anomaly=None, clean_raw=None, dates=None):
    arrays = {"raw": np.asarray(raw)}
    for name, data in (("regime", regime), ("is_anomaly", is_anomaly), ("clean_raw", clean_raw)):
        if data is not None:
            arrays[name] = np.asarray(data)
    if dates is not None:
        arrays["dates"] = np.asarray(dates).astype(str)
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Object arrays are not supported in input snapshots")
    return arrays


def input_fingerprint(arrays, features):
    entries = {}
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values)
        entries[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                             sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return value_hash(dict(arrays=entries, features=list(features)))


def save_npz(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write(path, buffer.getvalue())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def source_hashes():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}


def computational_hashes(hashes=None):
    hashes = source_hashes() if hashes is None else hashes
    return {key: value for key, value in hashes.items() if key != "evaluation/visualize.py"}


def evaluator_hash():
    return value_hash(computational_hashes())


def source_snapshot(path):
    root = Path(__file__).resolve().parents[2]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in (root / "src/tsprep", root / "scripts"):
            for file in sorted(folder.rglob("*.py")):
                archive.writestr(file.relative_to(root).as_posix(), file.read_bytes())
        for name in ("pyproject.toml", "reqs.txt"):
            archive.writestr(name, (root / name).read_bytes())
    atomic_write(path, buffer.getvalue())


def runtime_environment(device="cpu"):
    cuda = str(device).startswith("cuda")
    return dict(python=platform.python_version(), platform=platform.platform(),
        torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        device=str(device), cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(torch.device(device)) if cuda else None,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        environment={key: os.environ.get(key) for key in
                     ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")})


@contextmanager
def study_lock(target):
    """Exclusive cooperative writer; an interrupted process leaves a visible lock."""
    path = Path(target) / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Study writer lock exists: {path}. Check the recorded PID before removing a stale lock.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(dict(pid=os.getpid(), host=platform.node())))
        yield
    finally:
        path.unlink()
