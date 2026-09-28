"""
Загрузка реальных экономических данных из CSV.

Синтетика (``synthetic.py``) нужна для отладки пайплайна без
интернета. Для реальных экспериментов положите файл с историческими
котировками/курсами (например, выгруженный локально через
``yfinance``, Investing.com export, ЦБ РФ и т.п. - в этом окружении
сетевой доступ к финансовым API закрыт) в ``data/raw/`` и укажите
путь и колонки здесь.

Ожидаемый формат CSV: колонка с датой (по умолчанию 'Date') и одна
или несколько числовых колонок-признаков (например 'Close', 'Volume').
Строки должны быть УЖЕ отсортированы по возрастанию даты - функция
это проверяет и явно падает, если нет (тихая пересортировка была бы
опаснее: лучше пользователю увидеть ошибку и разобраться с источником).

Дополнительно: загрузка внешней бинарной/категориальной разметки
(например, FRED ``USREC`` - индикатор рецессии NBER) и её выравнивание
по датам ценового ряда - для downstream-задачи КЛАССИФИКАЦИИ эмбеддингов
(гл. 3.3 отчёта, "Уровень 2: полезность эмбеддингов"). Метка не участвует
в обучении автоэнкодера - только в оценке качества латентного пространства.
"""
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
    if not dates.is_monotonic_increasing:
        raise ValueError(
            f"Column '{date_col}' is not sorted ascending - sort the CSV by date "
            "before loading (chronological order is required for this pipeline)."
        )
    if value_cols is None:
        value_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not value_cols:
            raise ValueError("No numeric value columns found; pass value_cols explicitly.")
    values = df[value_cols].to_numpy(dtype=float)
    if np.isnan(values).any():
        n_nan = int(np.isnan(values).sum())
        raise ValueError(
            f"CSV contains {n_nan} NaN values in {value_cols}. "
            "Fill or drop them before loading (e.g. df.ffill())."
        )
    return dates.to_numpy(), values, value_cols


def load_label_series(
    path: str,
    date_col: str = "DATE",
    label_col: str = "USREC",
) -> tuple:
    """
    Загружает внешнюю разметку с собственной (обычно более редкой)
    частотой дискретизации - например, ежемесячный индикатор рецессии
    NBER из FRED (https://fred.stlouisfed.org/series/USREC): колонки
    ``DATE`` и ``USREC`` (0/1).

    Возвращает (dates, labels) - БЕЗ выравнивания по ценовому ряду,
    это делает ``align_labels_to_dates``.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{date_col}' and '{label_col}' in {path}. "
                          f"Got: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col])
    labels = df[label_col].to_numpy()
    order = np.argsort(dates.to_numpy())
    return dates.to_numpy()[order], labels[order]


def align_labels_to_dates(
    value_dates: np.ndarray,
    label_dates: np.ndarray,
    label_values: np.ndarray,
) -> np.ndarray:
    """
    Выравнивает более редкую разметку (например, ежемесячный USREC) на
    даты ценового ряда (например, ежедневного) методом "последнее
    известное значение" (forward-fill): для каждой даты ценового ряда
    берётся последнее значение метки, известное не позже этой даты -
    то есть без заглядывания в будущее (важно для честной оценки).

    Даты ценового ряда РАНЬШЕ первой даты разметки получают значение
    первой доступной метки (нет альтернативы без будущего knowledge).
    """
    label_df = pd.DataFrame({"date": pd.to_datetime(label_dates), "label": label_values}).sort_values("date")
    value_df = pd.DataFrame({"date": pd.to_datetime(value_dates)}).sort_values("date")
    merged = pd.merge_asof(value_df, label_df, on="date", direction="backward")
    if merged["label"].isna().any():
        # значения ценового ряда раньше первой даты метки - заполняем первой известной меткой
        merged["label"] = merged["label"].bfill()
    return merged["label"].to_numpy()


def load_series_with_recession_labels(
    price_path: str,
    label_path: str,
    price_date_col: str = "Date",
    value_cols: list[str] | None = None,
    label_date_col: str = "DATE",
    label_col: str = "USREC",
) -> dict:
    """
    Готовый конвейер для сценария "S&P 500 (или любой ценовой ряд) +
    метка рецессии NBER/FRED USREC": загружает оба CSV, выравнивает
    метку на даты ценового ряда, возвращает всё, что нужно для
    prepare_and_window(..., regime=...) и downstream-классификации
    """
    dates, values, cols = load_csv_series_with_dates(price_path, price_date_col, value_cols)
    label_dates, label_values = load_label_series(label_path, label_date_col, label_col)
    aligned_labels = align_labels_to_dates(dates, label_dates, label_values).astype(int)
    return {"dates": dates, "values": values, "value_cols": cols, "labels": aligned_labels}