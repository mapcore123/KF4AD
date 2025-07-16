import numpy as np
from matrices import *

# ---------------------------------------------------------------------------
#  AEKF‑3  (Additive EKF + norm pseudo‑measurement)
# ---------------------------------------------------------------------------

class AEKF3:
    def __init__(self, dt: float,
                 sig_eps: float, sig_n: float,      # process PSDs
                 sig_b: float,                        # star‑tracker noise
                 sig_nor: float,                      # pseudo‑sensor noise
                 x0: np.ndarray | None = None,
                 P0: np.ndarray | None = None) -> None:

        self.dt = dt
        self.Q = np.diag([sig_eps**2] * 3 + [sig_n**2] * 3)   # (6×6)
        # Measurement covariances
        self.R_b   = (sig_b**2) * np.eye(3)           # (3×3)
        self.R_nor = np.array([[sig_nor**2]])                    # (1×1)
        # State & covariance
        self.x = (x0 if x0 is not None
                  else np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)).reshape(7, 1)
        self.P = (P0 if P0 is not None else np.eye(7)).astype(float)

    # -----------------------------------------------------------------------
    #  Prediction
    # -----------------------------------------------------------------------
    def predict(self, omega_m: np.ndarray) -> None:
        '''
        Predict step. omega_m is the measured angular velocity vector [w1,w2,w3]
        '''
        omega_m = omega_m.reshape(3,1)
        q = self.x[:4].reshape(4,1) # dims (4,1) 
        mu = self.x[4:] # dims (3,1)
        dq = 0.5 * omega_matrix(omega_m - mu) @ q # dims (4,1)
        f = np.vstack([dq, np.zeros((3,1))]) # dims (7,1)
        
        # State prediction step x_k+1|k
        self.x = self.x + self.dt * f # dims (7,1)
        
        # Jacobian matrices F and G
        F = F_matrix(self.x, omega_m) # dims (7,7)
        G = Gt_matrix(self.x[:4]) # dims (7,6)
        
        # Linear perturbation matrices Phi and Gamma
        Phi = np.eye(7) + F * self.dt # dims (7,7)
        Gamma = G * self.dt # dims (7,6)
        
        # Covariance prediction step P_k+1|k
        self.P = Phi @ self.P @ Phi.T + additive_Q(self.Q, Gamma, self.dt) # dims (7,7)

    # -----------------------------------------------------------------------
    #  Update: star‑tracker + quaternion‑norm pseudo‑sensor
    # -----------------------------------------------------------------------
    def update(self, z: np.ndarray, r: np.ndarray) -> None:

        z = z.reshape(3,1)
        r = r.reshape(3,1)
        q = self.x[:4].reshape(4,1) # dims (4,1)
        # Innovation vector y
        y = z - attitude_matrix(q) @ r # dims (3,1)

        # Jacobian matrix H
        Hq = H_q(q, r) # dims (3,4)
        H = np.hstack([Hq, np.zeros((3, 3))]) # dims (3,7)

        # Innovation covariance S
        S = H @ self.P @ H.T + self.R_b # dims (3,3)
        
        # Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(S) # dims (7,3)
        
        # State update step x_k+1|k+1
        self.x = self.x + K @ y # dims (7,1)
        I7 = np.eye(7) # dims (7,7)
        
        # Covariance update step P_k+1|k+1
        self.P = (I7 - K @ H) @ self.P @ (I7 - K @ H).T + K @ self.R_b @ K.T # dims (7,7)

        # ---------- 2) Norm pseudo‑measurement (1×) ----------
        y_nor = np.array([[1.0 - float(q.T @ q)]])   # scalar innovation

        H_nor = np.hstack([2.0 * q.T, np.zeros((1, 3))])  # (1×7)

        S_nor = H_nor @ self.P @ H_nor.T + self.R_nor      # (1×1)
        K_nor = self.P @ H_nor.T @ np.linalg.inv(S_nor)           # (7×1)

        self.x = self.x + K_nor * y_nor
        self.P  = (I7 - K_nor @ H_nor) @ self.P @ (I7 - K_nor @ H_nor).T + K_nor @ self.R_nor @ K_nor.T # dims (7,7)

    # Convenience accessors ---------------------------------------------------
    @property
    def quaternion(self) -> np.ndarray:
        return self.x[:4].flatten()

    @property
    def bias(self) -> np.ndarray:
        return self.x[4:].flatten()


# ---------------------------------------------------------------------------
#  One‑cycle smoke test
# ---------------------------------------------------------------------------

def test_aekf3_single_iteration() -> None:

    dt = 0.1
    dt = 0.1
    sig_eps = 0.001  # rad/s /√Hz
    sig_n = 1e-5     # rad/s² /√Hz
    sig_b = 0.001    # pixel or rad
    sig_nor = 1e-6

    kf = AEKF3(dt, sig_eps, sig_n,sig_b,sig_nor)

    omega_meas = np.array([0.01, -0.02, 0.005])
    kf.predict(omega_meas)

    r_vector  = np.array([1., 0., 0.])
    true_q = np.array([0., 0., 0., 1.])
    z_body = attitude_matrix(true_q) @ r_vector
    z_body += np.random.normal(scale=sig_b, size=3)

    kf.update(z_body, r_vector)

    # Print results
    print("\n=== AEKF‑3 single iteration test ===")
    print("State estimate x:", kf.x)
    print("Quaternion ‖q‖:", np.linalg.norm(kf.quaternion))
    print("Bias estimate μ:", kf.bias)
    print("Covariance diag:", np.diag(kf.P))


if __name__ == "__main__":
    # np.set_printoptions(precision=5, suppress=True)
    test_aekf3_single_iteration()
