# src/performance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


# =========================
# Helpers de métricas
# =========================
def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return float((r.mean() * 252 - rf) / (r.std() * np.sqrt(252)))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    downside = r[r < 0].std()
    if len(r) == 0 or downside == 0 or np.isnan(downside):
        return np.nan
    return float((r.mean() * 252 - rf) / (downside * np.sqrt(252)))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return np.nan
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1.0
    mdd = max_drawdown(equity)
    if mdd is None or mdd >= 0:
        return np.nan
    return float(cagr / abs(mdd))


# =========================
# Resumen y plots estándar
# =========================
def summarize_performance(train_res: dict, test_res: dict, val_res: dict, save_csv: bool = False):
    sections = {
        "TRAIN": train_res,
        "TEST": test_res,
        "VALIDATION": val_res
    }
    rows = []
    for name, res in sections.items():
        eq = res["equity"]["portfolio_value"]
        r = _returns(eq)
        rows.append([
            name,
            float(eq.iloc[0]),
            float(eq.iloc[-1]),
            sharpe(r),
            sortino(r),
            calmar(eq),
            max_drawdown(eq),
            res["costs"]["commissions_total"],
            res["costs"]["borrow_total"],
            len(res["trades"])
        ])

    headers = ["Phase", "Start", "End", "Sharpe", "Sortino", "Calmar", "MaxDD",
               "Commissions", "Borrow", "N_Trades"]

    print(tabulate(rows, headers=headers, floatfmt=".4f"))

    if save_csv:
        out = pd.DataFrame(rows, columns=headers)
        out.to_csv("performance_summary.csv", index=False)


def trade_stats(trades_df: pd.DataFrame):
    if trades_df is None or len(trades_df) == 0:
        return {"n": 0, "win_rate": np.nan, "avg_win": np.nan,
                "avg_loss": np.nan, "profit_factor": np.nan}

    rets = trades_df["return"].astype(float) if "return" in trades_df.columns \
        else trades_df["PnL"].astype(float)

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / len(rets) if len(rets) > 0 else np.nan
    avg_win = wins.mean() if len(wins) > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf

    return {
        "n": int(len(rets)),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
        "avg_win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "profit_factor": float(pf) if np.isfinite(pf) else np.inf
    }


def plot_all(train_res: dict, test_res: dict, val_res: dict):
    eq_all = pd.concat([
        train_res["equity"]["portfolio_value"],
        test_res["equity"]["portfolio_value"],
        val_res["equity"]["portfolio_value"]
    ])

    plt.figure(figsize=(11, 5), dpi=130)
    eq_all.plot()
    plt.title("Equity Curve (Train/Test/Validation)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    all_trades = pd.concat(
        [train_res["trades"], test_res["trades"], val_res["trades"]],
        ignore_index=True
    )

    plt.figure(figsize=(9, 4), dpi=130)

    col = None
    if len(all_trades) > 0:
        if "return" in all_trades.columns:
            col = "return"
        elif "PnL" in all_trades.columns:
            col = "PnL"

    if col is not None:
        vals = pd.to_numeric(all_trades[col], errors="coerce").dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=20)
        else:
            plt.hist([], bins=20)
    else:
        plt.hist([], bins=20)

    plt.title("Trade Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("trade_distribution.png")
    plt.close()


# =========================
# Gráficos EXTRA MEJORADOS
# =========================

def plot_hedge_beta(beta_series: pd.Series, path: str = "hedge_beta.png"):
    if beta_series is None or len(beta_series) == 0:
        print("plot_hedge_beta: beta_series vacío; no se genera gráfico.")
        return

    plt.figure(figsize=(11, 5), dpi=150)
    beta_series.plot(color="#1f77b4")
    plt.title("Hedge Ratio (β_t) Over Time", fontsize=13)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


# ⭐⭐⭐ GRÁFICA DE SPREAD — **VERSIÓN MEJORADA Y MÁS CLARA**
def plot_spread_evolution(spread_series: pd.Series, path: str = "spread_evolution.png"):
    """
    Grafica la evolución del spread con escalado ×1000 y media móvil suavizada.
    Mucho más visible y profesional.
    """
    if spread_series is None or len(spread_series) == 0:
        print("plot_spread_evolution: spread vacío; no se genera gráfico.")
        return

    # Escalamos el spread para que se vea
    scaled = spread_series * 1000

    # MA de 1 año
    ma = scaled.rolling(252).mean()

    plt.figure(figsize=(12, 6), dpi=150)
    scaled.plot(label="Spread × 1000", color="#1f77b4", linewidth=1.2)
    ma.plot(label="MA(252)", color="#ff7f0e", linewidth=2)

    plt.title("Spread Evolution (Scaled ×1000)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Scaled Spread", fontsize=12)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


def plot_johansen_eigenvector(eigvec_series: pd.Series | None, path: str = "johansen_eigenvector.png"):
    if eigvec_series is None or len(eigvec_series) == 0:
        print("plot_johansen_eigenvector: no hay eigenvector; se omite.")
        return

    plt.figure(figsize=(11, 5), dpi=150)
    eigvec_series.plot(color="#2ca02c")
    plt.title("First Johansen Eigenvector Over Time", fontsize=13)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


def _concat_series_if_present(res_train, res_test, res_val, key: str):
    parts = []
    for res in (res_train, res_test, res_val):
        if key in res and res[key] is not None and len(res[key]) > 0:
            parts.append(res[key].copy())
    if len(parts) == 0:
        return None
    return pd.concat(parts).sort_index()


def plot_zscore_and_signals(res_train, res_test, res_val,
                            entry_z, exit_z, stop_z,
                            path="zscore_signals.png"):

    z = _concat_series_if_present(res_train, res_test, res_val, "z")
    sig = _concat_series_if_present(res_train, res_test, res_val, "signals")

    if z is None or sig is None:
        print("plot_zscore_and_signals: no hay datos; se omite.")
        return

    z = z.astype(float)
    sig = sig.reindex(z.index).fillna(0.0).astype(float)

    sig_shift = sig.shift(1).fillna(0.0)
    entries = (sig != 0) & (sig_shift == 0)
    long_entries = entries & (sig > 0)
    short_entries = entries & (sig < 0)

    plt.figure(figsize=(12, 6), dpi=150)

    z.plot(label="Z-score", color="#1f77b4")
    plt.axhline(entry_z, linestyle="--", color="green", alpha=0.6)
    plt.axhline(-entry_z, linestyle="--", color="green", alpha=0.6)
    plt.axhline(exit_z, linestyle=":", color="orange")
    plt.axhline(-exit_z, linestyle=":", color="orange")
    plt.axhline(stop_z, linestyle="-.", color="red", alpha=0.6)
    plt.axhline(-stop_z, linestyle="-.", color="red", alpha=0.6)

    plt.scatter(z.index[long_entries], z[long_entries],
                marker="^", s=40, color="green", label="Long Entry")
    plt.scatter(z.index[short_entries], z[short_entries],
                marker="v", s=40, color="red", label="Short Entry")

    plt.title("Z-score & Signals", fontsize=14)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


def plot_equity_by_phase(res_train, res_test, res_val,
                         paths=("equity_train.png", "equity_test.png", "equity_val.png")):

    names = ("TRAIN", "TEST", "VALIDATION")
    results = (res_train, res_test, res_val)

    for name, res, path in zip(names, results, paths):
        if "equity" not in res:
            continue

        plt.figure(figsize=(11, 4), dpi=140)
        res["equity"]["portfolio_value"].plot(color="#1f77b4")
        plt.title(f"Equity Curve - {name}", fontsize=13)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Guardado: {path}")
