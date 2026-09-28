"""
Скачивание сырых данных для проекта: ценовые ряды через yfinance
(Yahoo Finance) и макро-разметку (например, индикатор рецессии NBER USREC) напрямую с FRED
"""
import argparse
from pathlib import Path

import pandas as pd


def download_price_csv(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
    out_path: str | Path = "data/raw/prices.csv",
    interval: str = "1d",
) -> Path:
    """
    Скачивает OHLCV через yfinance и сохраняет в формате:
    колонка Date + числовые колонки (Open, High, Low, Close, Adj Close, Volume),
    строго отсортировано по возрастанию даты, без дублей дат.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval=interval,
                      progress=False, auto_adjust=False)
    assert isinstance(df, pd.DataFrame), f"yfinance returned unexpected type {type(df)}"
    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for ticker='{ticker}' "
            f"(start={start}, end={end}). Check the ticker symbol and date range, "
            f"and make sure this machine has internet access to Yahoo Finance."
        )

    # В некоторых версиях yfinance колонки приходят как MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()  # DatetimeIndex -> колонка Date
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date")
    if df["Date"].duplicated().any():
        df = df.drop_duplicates(subset="Date", keep="last")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[download_price_csv] {ticker}: {len(df)} rows, "
          f"{pd.to_datetime(df['Date']).min().date()} .. {pd.to_datetime(df['Date']).max().date()} -> {out_path}")
    return out_path


def download_fred_series(
    series_id: str = "USREC",
    out_path: str | Path = "data/raw/USREC.csv",
) -> Path:
    """
    Скачивает временной ряд напрямую с FRED (без API-ключа) через
    публичный CSV-эндпоинт графика. Работает для любой серии FRED, не
    только USREC - например, DFF (ставка ФРС), CPIAUCSL (CPI)
    """
    import urllib.request

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download FRED series '{series_id}' from {url}: {e}. "
            f"Check internet access, or download the CSV manually from "
            f"https://fred.stlouisfed.org/series/{series_id} (кнопка Download)."
        ) from e

    df = pd.read_csv(out_path)
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "observation_date"})
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date")
    df.to_csv(out_path, index=False)
    print(f"[download_fred_series] {series_id}: {len(df)} rows, "
          f"{df['observation_date'].min().date()} .. {df['observation_date'].max().date()} -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default=None, help="Yahoo Finance тикер, напр. AAPL, ^GSPC, BTC-USD")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None, help="по умолчанию - до сегодня")
    parser.add_argument("--interval", default="1d", help="1d, 1wk, 1mo, ...")
    parser.add_argument("--out", default="data/raw/prices.csv")
    parser.add_argument("--fred", default=None, help="ID серии FRED для скачивания, напр. USREC")
    parser.add_argument("--fred-out", default="data/raw/USREC.csv")
    args = parser.parse_args()

    if not args.ticker and not args.fred:
        parser.error("укажите хотя бы --ticker или --fred")

    if args.ticker:
        download_price_csv(args.ticker, args.start, args.end, args.out, args.interval)
    if args.fred:
        download_fred_series(args.fred, args.fred_out)


if __name__ == "__main__":
    main()