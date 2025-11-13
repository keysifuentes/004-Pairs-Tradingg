# src/kalman.py
import numpy as np
import pandas as pd


class KalmanFilter:
    """
    Filtro de Kalman genérico (baja dimensión):
      x_t = F x_{t-1} + w_t,   w ~ N(0, Q)
      y_t = H x_t     + v_t,   v ~ N(0, R)
    """
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.asarray(F)
        self.H = np.asarray(H)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.x = np.asarray(x0)
        self.P = np.asarray(P0)
        self.history = []

    def step(self, y):
        # Predicción
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Actualización
        y_pred = self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        innov = y - y_pred

        self.x = x_pred + K @ innov
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ P_pred

        self.history.append((self.x.copy(), self.P.copy(), K.copy(), innov, S))
        return self.x, self.P, K, innov, S

    def run(self, y_series):
        estimates = []
        for y in y_series:
            obs = np.array([y]) if np.isscalar(y) else np.asarray(y)
            x, P, K, innov, S = self.step(obs)
            estimates.append(x.copy())
        return np.array(estimates)


class HedgeKalman:
    """
    Cobertura dinámica: y_t = alpha_t + beta_t * x_t + ruido
    Estado: [alpha_t, beta_t]'; Observación: y_t

    NOTA: usamos Q > R para que el filtro se adapte más a los movimientos del precio.
    """
    def __init__(self, q_alpha=0.10, q_beta=0.10, r_obs=0.05):
        self.q_alpha = q_alpha
        self.q_beta = q_beta
        self.r_obs = r_obs
        self.filter = None
        self.df_est = None
        self.alpha_series = None
        self.beta_series = None

    def fit(self, x_series, y_series):
        x_series = np.asarray(x_series, float)
        y_series = np.asarray(y_series, float)
        n = len(x_series)

        F = np.eye(2)                      # random walk
        Q = np.diag([self.q_alpha, self.q_beta])
        R = np.array([[self.r_obs]])
        x0 = np.array([0.0, 1.0])          # init: alpha=0, beta=1
        P0 = np.eye(2)

        estimates = []
        self.filter = None
        for t in range(n):
            H = np.array([[1.0, x_series[t]]])  # y = [1 x_t] [alpha beta]'
            if self.filter is None:
                self.filter = KalmanFilter(F, H, Q, R, x0, P0)
            else:
                self.filter.H = H
            x, P, K, innov, S = self.filter.step(np.array([y_series[t]]))
            estimates.append([x[0], x[1]])

        self.df_est = pd.DataFrame(estimates, columns=["alpha", "beta"])
        self.alpha_series = self.df_est["alpha"]
        self.beta_series = self.df_est["beta"]
        return self

    def predict_params(self):
        if self.filter is None:
            return 0.0, 1.0
        return float(self.filter.x[0]), float(self.filter.x[1])


class SignalKalman:
    """
    Kalman para suavizar el spread:
      Estado: s_t (random walk) ; Observación: spread observado
    """
    def __init__(self, q=1e-4, r=1e-3):
        self.kf = KalmanFilter(
            F=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[q]]),
            R=np.array([[r]]),
            x0=np.array([0.0]),
            P0=np.array([[1.0]])
        )
        self.smoothed = None

    def fit(self, spread_series):
        est = self.kf.run(spread_series.values)
        self.smoothed = pd.Series(est[:, 0], index=spread_series.index, name="spread_smoothed")
        return self

    def latest(self):
        return float(self.kf.x[0])
