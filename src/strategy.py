# src/strategy.py
import pandas as pd
from dataclasses import dataclass
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class VECMZScoreStrategy:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    lookback: int = 252   # ~1 año de datos hábiles

    def _ect_from_johansen(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """
        Calcula el término de corrección de error (ECT) usando Johansen.
        Se construye una combinación lineal beta_x * x + beta_y * y.
        """
        common = x.index.intersection(y.index)
        x = x.loc[common].astype(float)
        y = y.loc[common].astype(float)

        mat = pd.concat([x, y], axis=1).dropna().values
        joh = coint_johansen(mat, det_order=0, k_ar_diff=1)
        v = joh.evec[:, 0]  # primer eigenvector

        if abs(v[1]) < 1e-12:
            beta_x = 1.0
            beta_y = -1.0
        else:
            beta_x = v[0] / v[1]
            beta_y = 1.0

        ect = beta_x * x + beta_y * y
        ect.name = "ECT"
        return ect

    def signals(self, spread_input: pd.Series | pd.DataFrame):
        """
        Si recibe DataFrame[x, y] => calcula ECT (VECM) y usa su z-score.
        Si recibe Series => asume que ya es 'spread' y usa su z-score.
        Devuelve (signal_series, z_series).
        """
        if isinstance(spread_input, pd.DataFrame):
            x = spread_input.iloc[:, 0]
            y = spread_input.iloc[:, 1]
            base = self._ect_from_johansen(x, y)
        elif isinstance(spread_input, pd.Series):
            base = spread_input.astype(float)
        else:
            raise ValueError("spread_input debe ser Series (spread) o DataFrame con columnas del par (x, y).")

        mean = base.rolling(self.lookback).mean()
        std = base.rolling(self.lookback).std(ddof=0)
        z = (base - mean) / std

        sig = pd.Series(0.0, index=base.index, dtype=float)
        z_valid = z.dropna()
        sig.loc[z_valid.index[z_valid < -self.entry_z]] = 1.0
        sig.loc[z_valid.index[z_valid > self.entry_z]] = -1.0
        sig.loc[z_valid.index[z_valid.abs() < self.exit_z]] = 0.0

        return sig, z
