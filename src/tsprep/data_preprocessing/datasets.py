"""
Формирование обучающих срезов методом скользящего окна и хронологический
сплит train/val/test
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsprep.config import DataConfig
from tsprep.data_preprocessing.preprocessing import prepare_series


def sliding_windows(x: np.ndarray, window_len: int, stride: int = 1) -> np.ndarray:
    """
    X^(i) = [x'_i, x'_{i+1}, ..., x'_{i+T-1}], i = 0, ..., N-T (гл. 3.1).

    x: (length, F) -> windows: (N, T, F)
    """
    length = x.shape[0]
    n_windows = (length - window_len) // stride + 1
    if n_windows <= 0:
        raise ValueError(f"Series too short ({length}) for window_len={window_len}")
    f = x.shape[1] if x.ndim > 1 else 1
    x2 = x.reshape(length, f)
    out = np.empty((n_windows, window_len, f), dtype=x2.dtype)
    for k, start in enumerate(range(0, n_windows * stride, stride)):
        out[k] = x2[start:start + window_len]
    return out


def chronological_split_indices(n: int, train_frac: float, val_frac: float, test_frac: float):
    """Возвращает (train_end, val_end) - границы индексов по времени."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return train_end, val_end


class WindowDataset(Dataset):
    """Простая обёртка над тензором окон (N, T, F) для автоэнкодера."""

    def __init__(self, windows: np.ndarray):
        self.x = torch.from_numpy(windows).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


@dataclass
class PreparedData:
    train_windows: np.ndarray
    val_windows: np.ndarray
    test_windows: np.ndarray
    normalizer: object
    regime: np.ndarray | None = None
    is_anomaly: np.ndarray | None = None
    raw_normalized: np.ndarray | None = None


def prepare_and_window(
    raw: np.ndarray,
    cfg: DataConfig,
    regime: np.ndarray | None = None,
    is_anomaly: np.ndarray | None = None,
) -> PreparedData:
    """
    Полный конвейер: детрендирование + нормализация (train-fit) ->
    хронологический сплит по времени -> скользящее окно на каждом
    под-диапазоне отдельно (чтобы окна одного среза не заглядывали в
    данные другого).
    """
    prep = prepare_series(raw, seasonal_period=cfg.seasonal_period, train_frac=cfg.train_frac,
                           log_transform=cfg.log_transform)
    normalized = prep["normalized"]
    n = normalized.shape[0]
    train_end, val_end = chronological_split_indices(n, cfg.train_frac, cfg.val_frac, cfg.test_frac)

    train_part = normalized[:train_end]
    val_part = normalized[train_end:val_end]
    test_part = normalized[val_end:]

    train_windows = sliding_windows(train_part, cfg.window_len, cfg.stride)
    val_windows = sliding_windows(val_part, cfg.window_len, cfg.stride)
    test_windows = sliding_windows(test_part, cfg.window_len, cfg.stride)

    return PreparedData(
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
        normalizer=prep["normalizer"],
        regime=regime[cfg.seasonal_period:] if regime is not None else None,
        is_anomaly=is_anomaly[cfg.seasonal_period:] if is_anomaly is not None else None,
        raw_normalized=normalized,
    )


def build_dataloaders(prepared: PreparedData, cfg: DataConfig):
    train_ds = WindowDataset(prepared.train_windows)
    val_ds = WindowDataset(prepared.val_windows)
    test_ds = WindowDataset(prepared.test_windows)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_eval_loader(windows: np.ndarray, batch_size: int) -> DataLoader:
    """
    Лоадер БЕЗ shuffle и БЕЗ drop_last - для извлечения эмбеддингов
    (``collect_embeddings``), а не для обучения.

    ВАЖНО: обучающий ``train_loader`` из ``build_dataloaders`` специально
    сделан с ``shuffle=True, drop_last=True`` - это правильно для
    градиентного спуска, но НЕЛЬЗЯ использовать его же для сбора
    эмбеддингов, если потом планируется сопоставлять их с
    внешними метками/таргетами по индексу окна (downstream-классификация,
    прогнозирование): shuffle ломает соответствие порядка, а drop_last
    уменьшает количество примеров - оба эффекта дают либо ValueError о
    несовпадении размеров, либо (что хуже) молча дают неправильные пары
    (z, label). Используйте:

        train_eval_loader = build_eval_loader(prepared.train_windows, cfg.batch_size)
        train_out = collect_embeddings(model, train_eval_loader)

    вместо ``collect_embeddings(model, train_loader)`` везде, где эмбеддинги
    train-выборки сопоставляются с метками/таргетами по порядку окон.
    """
    ds = WindowDataset(windows)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)