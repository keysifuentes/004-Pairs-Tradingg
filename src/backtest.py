# src/backtest.py
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Backtester:
    cash: float = 1_000_000.0
    commission_rate: float = 0.00125     # 0.125% por trade
    borrow_rate_annual: float = 0.0025   # 0.25% anual
    rebalance_daily: bool = True

    def _daily_borrow_rate(self):
        return (1.0 + self.borrow_rate_annual) ** (1 / 252) - 1.0

    def run(self, data: pd.DataFrame, hedge_model, signal_model, strategy,
            phase: str = "train", pair=None):
        """
        data: DataFrame con columnas [x, y] (precios)
        hedge_model: modelo de hedge (Kalman) para beta_t
        signal_model: (no lo usamos mucho aquí, pero lo dejamos por compatibilidad)
        strategy: VECMZScoreStrategy (devuelve sig, z a partir de DataFrame [x,y])
        """
        if pair is None:
            x, y = data.columns[0], data.columns[1]
        else:
            x, y = pair

        prices = data[[x, y]].dropna().copy()
        idx = prices.index

        # Señales y z-score (VECM/ECT sobre [x,y])
        sig_series, z_series = strategy.signals(prices)

        # Posición target: -1, 0, +1 (short spread, flat, long spread)
        pos = sig_series.reindex(idx).fillna(0.0)

        # Sizing: 40% del cash en cada pata (80% total)
        leg_notional = self.cash * 0.40
        daily_borrow = self._daily_borrow_rate()

        # Trackers
        cash = self.cash
        qty_x = 0.0
        qty_y = 0.0
        portfolio_values = []
        commissions_acc = 0.0
        borrow_acc = 0.0

        trades = []
        prev_pos = 0.0

        # Para calcular PnL/return por trade
        trade_open_value = None
        trade_side = None

        for t in range(len(idx)):
            date = idx[t]
            px_x = float(prices.loc[date, x])
            px_y = float(prices.loc[date, y])
            cur_pos = float(pos.loc[date])  # -1, 0, +1

            # =======================
            #   OBTENER beta_t
            # =======================
            beta_t = None
            for attr in ["beta_", "beta_series", "beta"]:
                if hasattr(hedge_model, attr):
                    beta_attr = getattr(hedge_model, attr)
                    if isinstance(beta_attr, pd.Series):
                        if date in beta_attr.index:
                            beta_t = float(beta_attr.loc[date])
                    else:
                        try:
                            beta_t = float(beta_attr)
                        except Exception:
                            beta_t = None
                    break

            if beta_t is None:
                # fallback: OLS si no hay beta_t expuesto
                b = np.polyfit(prices[x].values, prices[y].values, 1)
                beta_t = float(b[0])

            # =======================
            #   TARGET DE CANTIDADES
            # =======================
            target_qty_x = 0.0
            target_qty_y = 0.0

            if cur_pos != 0:
                qty_leg_x = leg_notional / px_x
                qty_leg_y = leg_notional / px_y

                if cur_pos > 0:     # long spread: long y, short x
                    target_qty_y = +qty_leg_y
                    target_qty_x = -qty_leg_x
                else:               # short spread: short y, long x
                    target_qty_y = -qty_leg_y
                    target_qty_x = +qty_leg_x

            # Ejecutar rebalance: trades = target - current
            trade_x = target_qty_x - qty_x
            trade_y = target_qty_y - qty_y

            # =======================
            #   COMISIONES
            # =======================
            trade_value_x = abs(trade_x) * px_x
            trade_value_y = abs(trade_y) * px_y
            commission = self.commission_rate * (trade_value_x + trade_value_y)
            commissions_acc += commission

            # Pagar compras/ventas + comisión
            cash -= (trade_x * px_x + trade_y * px_y) + commission

            # =======================
            #   COSTO DE BORROW
            # =======================
            short_notional = 0.0
            if qty_x < 0:
                short_notional += abs(qty_x) * px_x
            if qty_y < 0:
                short_notional += abs(qty_y) * px_y
            borrow_today = short_notional * daily_borrow
            borrow_acc += borrow_today
            cash -= borrow_today

            # Actualizar holdings
            qty_x += trade_x
            qty_y += trade_y

            # Valor de cartera
            port_val = cash + qty_x * px_x + qty_y * px_y
            portfolio_values.append((date, port_val))

            # =======================
            #   GESTIÓN DE TRADES
            # =======================
            # Apertura
            if prev_pos == 0 and cur_pos != 0:
                trade_open_value = port_val
                trade_side = "LONG_SPREAD" if cur_pos > 0 else "SHORT_SPREAD"

            # Cierre
            if prev_pos != 0 and cur_pos == 0 and trade_open_value is not None:
                trade_return = (port_val - trade_open_value) / trade_open_value
                trades.append({
                    "date": date,
                    "side": trade_side,
                    "return": float(trade_return)
                })
                trade_open_value = None
                trade_side = None

            prev_pos = cur_pos

        equity_df = pd.DataFrame(
            portfolio_values,
            columns=["Date", "portfolio_value"]
        ).set_index("Date")

        trades_df = pd.DataFrame(trades)

        result = {
            "equity": equity_df,
            "trades": trades_df,
            "costs": {
                "commissions_total": commissions_acc,
                "borrow_total": borrow_acc
            },
            "signals": pos,   # Serie -1/0/+1
            "z": z_series     # Serie Z-score
        }
        return result
