"""Загрузка CSV с возрастающими датами и выравнивание внешних меток."""
import numpy as np
import pandas as pd


def load_csv_series(
    path: str,
    date_col: str = "Date",
    value_cols: list[str] | None = None,
) -> np.ndarray:
    """Обратно совместимая версия: возвращает только значения (без дат)."""
    dates, values, _ = load_csv_series_with_dates(path, date_col, value_cols)
    return values


def load_csv_series_with_dates(
    path: str,
    date_col: str = "Date",
    value_cols: list[str] | None = None,
) -> tuple:
    """То же самое, что ``load_csv_series``, но дополнительно возвращает
    массив дат - нужен для выравнивания внешней разметки (см. ниже)."""
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {path}. Columns: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col])
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(
            f"Column '{date_col}' is not sorted ascending - sort the CSV by date "
            "before loading (chronological order is required for this pipeline)."
        )
    if value_cols is None:
        value_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not value_cols:
            raise ValueError("No numeric value columns found; pass value_cols explicitly.")
    if not value_cols or len(set(value_cols)) != len(value_cols):
        raise ValueError("Select a nonempty list of unique feature columns")
    values = df[value_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        n_nan = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"CSV contains {n_nan} NaN values in {value_cols}. "
            "Fill or drop them before loading (e.g. df.ffill())."
        )
    return dates.to_numpy(), values, value_cols


def load_label_series(
    path: str,
    date_col: str = "observation_date",
    label_col: str = "USREC",
) -> tuple:
    """Возвращает даты и значения внешней разметки, отсортированные по дате."""
    df = pd.read_csv(path)
    if date_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{date_col}' and '{label_col}' in {path}. "
                          f"Got: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col])
    labels = df[label_col].to_numpy()
    if dates.isna().any() or dates.duplicated().any() or not np.isfinite(labels.astype(float)).all():
        raise ValueError("Invalid or duplicate label dates / missing labels")
    order = np.argsort(dates.to_numpy())
    return dates.to_numpy()[order], labels[order]


def align_labels_to_dates(
    value_dates: np.ndarray,
    label_dates: np.ndarray,
    label_values: np.ndarray,
) -> np.ndarray:
    """Сопоставляет последнюю метку с датой не позже даты наблюдения.

    До первой метки возвращает NaN. Даты USREC обозначают периоды,
    а не публикации; метки используются ретроспективно.
    """
    label_df = pd.DataFrame({"date": pd.to_datetime(label_dates), "label": label_values}).sort_values("date")
    if len(label_dates) != len(label_values) or not len(label_dates):
        raise ValueError("Invalid label arrays")
    if label_df["date"].duplicated().any() or label_df["date"].isna().any():
        raise ValueError("Invalid label dates")
    value_df = pd.DataFrame({"date": pd.to_datetime(value_dates), "position": np.arange(len(value_dates))}).sort_values("date")
    merged = pd.merge_asof(value_df, label_df, on="date", direction="backward")
    return merged.sort_values("position")["label"].to_numpy()


def load_series_with_recession_labels(
    price_path: str,
    label_path: str,
    price_date_col: str = "Date",
    value_cols: list[str] | None = None,
    label_date_col: str = "observation_date",
    label_col: str = "USREC",
) -> dict:
    """Загружает ценовой ряд и выравнивает метки рецессии по его датам."""
    dates, values, cols = load_csv_series_with_dates(price_path, price_date_col, value_cols)
    label_dates, label_values = load_label_series(label_path, label_date_col, label_col)
    aligned_labels = align_labels_to_dates(dates, label_dates, label_values)
    return {"dates": dates, "values": values, "value_cols": cols, "labels": aligned_labels}
