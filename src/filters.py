import numpy as np
from dataclasses import dataclass

from quaternion_utils import (
    propagate,
    multiply,
    normalize,
    small_angle_quat,
    to_dcm,
)

# Convenient cross product matrix
def cross_matrix(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


@dataclass
class FilterState:
    q: np.ndarray  # Quaternion (4,)
    mu: np.ndarray  # Gyro bias (3,)
    P: np.ndarray  # Covariance (n,n)


class AdditiveEKF:
    """Generic additive EKF on full quaternion state (7x7 covariance)."""

    def __init__(self, Q_gyro: float, Q_bias: float, R_star: float):
        self.Q_gyro = Q_gyro
        self.Q_bias = Q_bias
        self.R_star = R_star

    def init_state(self, q0, mu0, P0):
        return FilterState(q0.copy(), mu0.copy(), P0.copy())

    # --- Process model helpers ---

    def f(self, state: FilterState, gyro_measure: np.ndarray, dt: float):
        """Process model propagation for state."""
        # angular rate estimate
        omega_est = gyro_measure - state.mu
        q_new = propagate(state.q, omega_est, dt)
        mu_new = state.mu  # bias assumed constant during dt (random walk model handled via Q)
        return q_new, mu_new

    def F_jacobian(self, state: FilterState, gyro_measure: np.ndarray, dt: float):
        """Numerical Jacobian of f w.r.t state (7x7)."""
        x0 = np.concatenate((state.q, state.mu))
        f0_q, f0_mu = self.f(state, gyro_measure, dt)
        x0_prop = np.concatenate((f0_q, f0_mu))
        n = len(x0)
        F = np.zeros((n, n))
        eps = 1e-6
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            # Positive perturbation
            qp = normalize(state.q + (dx[:4] if i < 4 else 0))
            mup = state.mu.copy()
            if i >= 4:
                mup[i - 4] += dx[i]
            state_p = FilterState(qp, mup, state.P)
            f_qp, f_mup = self.f(state_p, gyro_measure, dt)
            xp = np.concatenate((f_qp, f_mup))
            F[:, i] = (xp - x0_prop) / eps
        return F

    def propagate(self, state: FilterState, gyro_measure: np.ndarray, dt: float):
        q_new, mu_new = self.f(state, gyro_measure, dt)
        # Covariance propagation
        F = self.F_jacobian(state, gyro_measure, dt)
        # Process noise matrix Qd (discrete)
        Qd = np.zeros((7, 7))
        Qd[:4, :4] = np.eye(4) * self.Q_gyro * dt  # crude mapping
        Qd[4:, 4:] = np.eye(3) * self.Q_bias * dt
        P_new = F @ state.P @ F.T + Qd
        return FilterState(q_new, mu_new, P_new)

    # --- Measurement update (star tracker) ---

    def h(self, q: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
        return to_dcm(q) @ r_vec

    def H_jacobian(self, q: np.ndarray, r_vec: np.ndarray):
        # Analytical Jacobian: derivative w.r.t quaternion via small-angle (approx).
        # Use numerical for robustness
        eps = 1e-6
        H = np.zeros((3, 7))
        base = self.h(q, r_vec)
        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            q_pert = normalize(q + dq)
            diff = (self.h(q_pert, r_vec) - base) / eps
            H[:, i] = diff
        # Derivative w.r.t bias is zero for this measurement
        return H

    def update(self, state: FilterState, b_meas: np.ndarray, r_vec: np.ndarray):
        H = self.H_jacobian(state.q, r_vec)
        S = H @ state.P @ H.T + self.R_star * np.eye(3)
        K = state.P @ H.T @ np.linalg.inv(S)
        y = b_meas - self.h(state.q, r_vec)
        dx = K @ y  # 7x1
        # Update quaternion and bias
        dq = dx[:4]
        dmu = dx[4:]
        q_update = normalize(state.q + dq)  # Additive update
        mu_update = state.mu + dmu
        P_update = (np.eye(7) - K @ H) @ state.P
        return FilterState(q_update, mu_update, P_update)

    # Utility to match MEKF API for state extraction
    def get_state_tuple(self, state: FilterState):
        """Return (q, mu, P) tuple from FilterState."""
        return state.q.copy(), state.mu.copy(), state.P.copy()


class AEKF1(AdditiveEKF):
    pass  # Inherits behavior directly


class AEKF2(AdditiveEKF):
    def update(self, state: FilterState, b_meas: np.ndarray, r_vec: np.ndarray):
        state_up = super().update(state, b_meas, r_vec)
        # Additional quaternion normalization
        state_up.q = normalize(state_up.q)
        return state_up


class AEKF3(AdditiveEKF):
    def __init__(self, Q_gyro: float, Q_bias: float, R_star: float, sigma_nor: float = 1e-6):
        super().__init__(Q_gyro, Q_bias, R_star)
        self.sigma_nor = sigma_nor

    def update(self, state: FilterState, b_meas: np.ndarray, r_vec: np.ndarray):
        # First apply normal measurement update
        state_up = super().update(state, b_meas, r_vec)
        # Then pseudo-measurement enforcing norm=1
        q = state_up.q
        H_pseudo = np.zeros((1, 7))
        H_pseudo[0, :4] = 2 * q  # derivative of q^T q w.r.t q
        z_pred = np.dot(q, q)
        z = np.array([1.0])
        S = H_pseudo @ state_up.P @ H_pseudo.T + self.sigma_nor**2
        K = state_up.P @ H_pseudo.T / S  # 7x1
        y = z - np.array([z_pred])
        dx = (K.flatten() * y[0])
        dq = dx[:4]
        dmu = dx[4:]
        q_new = normalize(state_up.q + dq)
        mu_new = state_up.mu + dmu
        P_new = (np.eye(7) - K @ H_pseudo) @ state_up.P
        return FilterState(q_new, mu_new, P_new)


class MEKF:
    """Multiplicative EKF with 3-state error for quaternion and 3 for bias."""

    def __init__(self, Q_gyro: float, Q_bias: float, R_star: float):
        self.Q_gyro = Q_gyro
        self.Q_bias = Q_bias
        self.R_star = R_star

    def init_state(self, q0, mu0, P0_error):
        # P0_error is 6x6 covariance for [theta_error, bias_error]
        self.P = P0_error.copy()
        self.q = q0.copy()
        self.mu = mu0.copy()

    def propagate(self, gyro_measure: np.ndarray, dt: float):
        # Propagate quaternion with gyro measurement minus estimated bias
        omega_est = gyro_measure - self.mu
        self.q = propagate(self.q, omega_est, dt)
        # Bias remains but subject to random walk
        # Covariance propagation
        F = np.zeros((6, 6))
        # For small error dynamics: dtheta_dot = -[omega_est]_x dtheta - I dmu
        F[:3, :3] = -cross_matrix(omega_est)
        F[:3, 3:] = -np.eye(3)
        # Bias random walk: dmu_dot = 0
        # Discrete propagation
        Phi = np.eye(6) + F * dt
        Qd = np.zeros((6, 6))
        Qd[:3, :3] = np.eye(3) * self.Q_gyro * dt
        Qd[3:, 3:] = np.eye(3) * self.Q_bias * dt
        self.P = Phi @ self.P @ Phi.T + Qd

    def update(self, b_meas: np.ndarray, r_vec: np.ndarray):
        # Measurement sensitivity
        bo_pred = to_dcm(self.q) @ r_vec
        H = np.zeros((3, 6))
        H[:, :3] = cross_matrix(bo_pred)  # derivative w.r.t small angle error
        # derivative w.r.t bias is zero
        S = H @ self.P @ H.T + self.R_star * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        y = b_meas - bo_pred
        dx = K @ y
        dtheta = dx[:3]
        dmu = dx[3:]
        # Update quaternion
        dq = small_angle_quat(dtheta)
        self.q = normalize(multiply(dq, self.q))
        # Update bias
        self.mu = self.mu + dmu
        # Covariance update
        I_KH = np.eye(6) - K @ H
        Rm = self.R_star * np.eye(3)
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T  # Joseph-stabilized covariance update

    def get_state(self):
        return self.q.copy(), self.mu.copy(), self.P.copy() 