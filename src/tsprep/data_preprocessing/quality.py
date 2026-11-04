"""Read-only economic CSV quality diagnostics. Never silently repair quotes."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def csv_quality_report(path):
    path = Path(path)
    df = pd.read_csv(path)
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    numeric = df.select_dtypes("number")
    report = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(df),
        columns=df.columns.tolist(), missing_values=int(df.isna().sum().sum()),
        nonfinite_numeric=int((~np.isfinite(numeric)).sum().sum()), duplicate_dates=int(dates.duplicated().sum()),
        invalid_dates=int(dates.isna().sum()), sorted_dates=bool(dates.is_monotonic_increasing),
        constant_features=[c for c in numeric if numeric[c].nunique() <= 1],
        nonpositive_counts={c: int((numeric[c] <= 0).sum()) for c in numeric})
    if {"Open", "High", "Low", "Close"} <= set(df.columns):
        bad = (df.Low>df.High) | (df.Close<df.Low) | (df.Close>df.High) | (df.Open<df.Low) | (df.Open>df.High)
        report["ohlc_inconsistent_rows"] = int(bad.sum())
        report["ohlc_examples"] = df.loc[bad].head(10).to_dict("records")
    report["needs_review"] = bool(report.get("ohlc_inconsistent_rows", 0) or report["nonfinite_numeric"]
                                  or report["duplicate_dates"] or report["invalid_dates"] or not report["sorted_dates"])
    return report
