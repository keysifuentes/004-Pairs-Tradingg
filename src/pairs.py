# src/pairs.py
import itertools
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ---------------------------
# Screening y tests básicos
# ---------------------------
def screen_pairs(prices: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str]]:
    """
    Devuelve pares (x, y) con correlación >= threshold medido en TODO el período entregado.
    """
    corr = prices.corr()
    cols = prices.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) >= threshold:
                pairs.append((cols[i], cols[j]))
    return pairs


def engle_granger_test(df_xy: pd.DataFrame) -> Dict[str, float]:
    """
    Engle-Granger simple:
      y = b1*x + b0  -> resid = y - (b1*x + b0)
      ADF sobre residuales.
    Retorna p-valor del ADF (p<0.05 sugiere cointegración).
    """
    x = df_xy.iloc[:, 0].astype(float)
    y = df_xy.iloc[:, 1].astype(float)
    beta = np.polyfit(x.values, y.values, 1)  # y ≈ b1*x + b0
    resid = y - (beta[0] * x + beta[1])
    resid = resid.dropna()
    pval = adfuller(resid)[1]
    return {"adf_pvalue": float(pval)}


def johansen_vecm_fit(df_xy: pd.DataFrame) -> Dict[str, float]:
    """
    Johansen: devuelve primer estadístico trace y primer eigenvector.
    """
    x = df_xy.iloc[:, 0].astype(float)
    y = df_xy.iloc[:, 1].astype(float)
    mat = np.column_stack([x.values, y.values])
    joh = coint_johansen(mat, det_order=0, k_ar_diff=1)
    trace_stat = float(joh.lr1[0])  # estadístico trace para r=0
    eigvec = joh.evec[:, 0].copy()  # primer eigenvector (sin normalizar)
    return {"trace_stat": trace_stat, "eigvec": eigvec}


# ----------------------------------------
# Salidas para el reporte (rolling & tabla)
# ----------------------------------------
def save_rolling_corr(prices: pd.DataFrame, window: int = 252, out_csv: str = "reports/rolling_corr_last.csv"):
    """
    Guarda la matriz de correlaciones usando la ÚLTIMA ventana rolling (por defecto ~1 año hábil).
    """
    rc = prices.rolling(window).corr().dropna()
    # tomamos la última fecha disponible
    last_date = rc.index.get_level_values(0).max()
    last = rc.loc[last_date]
    last.to_csv(out_csv)


def save_pairs_table(prices: pd.DataFrame, out_csv: str = "reports/pair_stats.csv"):
    """
    Guarda una tabla con Corr, EG-ADF p-value y Johansen trace para TODOS los pares.
    Ordenada por p-valor ascendente y trace descendente.
    """
    rows = []
    cols = prices.columns.tolist()
    for x, y in itertools.combinations(cols, 2):
        s1, s2 = prices[x].dropna(), prices[y].dropna()
        common = s1.index.intersection(s2.index)
        s1, s2 = s1.loc[common], s2.loc[common]
        if len(common) < 50:
            continue

        corr = float(np.corrcoef(s1, s2)[0, 1])

        beta = np.polyfit(s1.values, s2.values, 1)  # s2 ≈ b1*s1 + b0
        resid = (s2 - (beta[0] * s1 + beta[1])).dropna()
        try:
            adf_p = adfuller(resid)[1]
        except Exception:
            adf_p = np.nan

        try:
            joh = coint_johansen(np.column_stack([s1.values, s2.values]), det_order=0, k_ar_diff=1)
            trace = float(joh.lr1[0])
        except Exception:
            trace = np.nan

        rows.append([x, y, corr, adf_p, trace])

    df = pd.DataFrame(rows, columns=["X", "Y", "Corr", "ADF_p", "Johansen_trace"])
    df.sort_values(["ADF_p", "Johansen_trace"], ascending=[True, False]).to_csv(out_csv, index=False)


# ------------------------------------------------------
# Eigenvector de Johansen en ventana móvil (para gráfico)
# ------------------------------------------------------
def rolling_johansen_eigenvector(
    prices: pd.DataFrame, x: str, y: str, window: int = 252, step: int = 5
) -> pd.Series:
    """
    Serie temporal de la 1ª componente del eigenvector de Johansen en ventana móvil.
    Normalizamos tomando y=1; devolvemos el coeficiente relativo de x (beta_x).
    """
    idx = prices.index
    vals = []
    times = []
    for start in range(0, len(idx) - window + 1, step):
        end = start + window
        win = prices.iloc[start:end][[x, y]].dropna()
        if len(win) < int(window * 0.8):
            continue
        mat = win.values
        try:
            joh = coint_johansen(mat, det_order=0, k_ar_diff=1)
            v = joh.evec[:, 0]
            beta_x = v[0] / v[1] if abs(v[1]) > 1e-12 else np.nan
        except Exception:
            beta_x = np.nan
        vals.append(beta_x)
        times.append(win.index[-1])
    return pd.Series(vals, index=pd.to_datetime(times), name=f"eigvec_x|{x}/{y}")
