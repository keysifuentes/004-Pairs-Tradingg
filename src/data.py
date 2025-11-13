# src/data.py
import os
from pathlib import Path
import pandas as pd
import yfinance as yf


def _date_tag(start: str, end: str | None) -> str:
    s = start.replace("-", "")
    e = (end or "").replace("-", "") or "today"
    return f"{s}-{e}"


def load_prices(tickers, start, end=None, save=True, data_dir: str = "data"):
    """
    Descarga precios con yfinance y devuelve un DataFrame con UNA columna por ticker (Adj Close o Close).
    Si save=True, guarda:
      - un CSV combinado con todas las series
      - un CSV por ticker
    """
    # --- Descarga ---
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # Normaliza a UNA serie por ticker (Adj Close si existe, sino Close)
    if isinstance(df.columns, pd.MultiIndex):
        fields = set(df.columns.get_level_values(0))
        if "Adj Close" in fields:
            px = df["Adj Close"].copy()
        elif "Close" in fields:
            px = df["Close"].copy()
        else:
            raise ValueError("No hay columnas 'Adj Close' ni 'Close' en los datos de yfinance.")
    else:
        # Un solo ticker → deja 1 columna con el nombre del ticker
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].rename(
                columns={"Adj Close": tickers[0] if isinstance(tickers, (list, tuple)) else tickers}
            )
        elif "Close" in df.columns:
            px = df[["Close"]].rename(
                columns={"Close": tickers[0] if isinstance(tickers, (list, tuple)) else tickers}
            )
        else:
            raise ValueError("No hay columnas 'Adj Close' ni 'Close' en los datos descargados.")

    # Limpieza y frecuencia de hábiles
    px = px.ffill().dropna(how="all")
    try:
        px = px.asfreq("B").ffill()
    except Exception:
        pass

    # --- Guardado en /data ---
    if save:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        tag = _date_tag(start, end)
        ncols = px.shape[1]

        # Archivo combinado (una columna por ticker)
        combined_path = Path(data_dir) / f"historical_prices_{tag}_{ncols}tickers.csv"
        px.to_csv(combined_path, index_label="Date")

        # Un archivo por ticker
        for col in px.columns:
            one = px[[col]].rename(columns={col: "Price"})
            one.to_csv(Path(data_dir) / f"{col}_prices_{tag}.csv", index_label="Date")

    return px


def train_test_val_split(df, f_train=0.6, f_test=0.2, f_val=0.2):
    """
    Split temporal 60/20/20 sin look-ahead.
    """
    assert abs(f_train + f_test + f_val - 1.0) < 1e-8
    n = len(df)
    i1 = int(n * f_train)
    i2 = int(n * (f_train + f_test))
    train = df.iloc[:i1].copy()
    test = df.iloc[i1:i2].copy()
    val = df.iloc[i2:].copy()
    return train, test, val
