import numpy as np
from matrices import *

# ----------------------------------------------------------------------------
# AEKF-2 class 
# ----------------------------------------------------------------------------

class AEKF2:
    def __init__(self, dt: float, sigma_eps: float, sigma_n: float, sigma_b: float,
                 x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        self.dt = dt        
        self.Q = np.diag([sigma_eps**2] * 3 + [sigma_n**2] * 3)
        self.R = (sigma_b**2) * np.eye(3)
        self.x = (x0 if x0 is not None else np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)).reshape(7,1)
        self.P = (P0 if P0 is not None else np.eye(7)).reshape(7,7)

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


    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        '''
        Update step. z is the measured sight-line vector [x,y,z]
        '''
        
        z = z.reshape(3,1)
        r = r.reshape(3,1)
        
        q = self.x[:4].reshape(4,1) # dims (4,1)
        # Innovation vector y
        y = z - attitude_matrix(q) @ r # dims (3,1)
        
        # Jacobian matrix H
        Hq = H_q(q, r) # dims (3,4)
        H = np.hstack([Hq, np.zeros((3, 3))]) # dims (3,7)

        # Innovation covariance S
        S = H @ self.P @ H.T + self.R # dims (3,3)
        
        # Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(S) # dims (7,3)
        
        # State update step x_k+1|k+1
        self.x = self.x + K @ y # dims (7,1)
        I7 = np.eye(7) # dims (7,7)
        
        # brute-force normalization
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        # Covariance update step P_k+1|k+1
        self.P = (I7 - K @ H) @ self.P @ (I7 - K @ H).T + K @ self.R @ K.T # dims (7,7)

    @property
    def quaternion(self) -> np.ndarray:
        return self.x[:4]

    @property
    def bias(self) -> np.ndarray:
        return self.x[4:]

# ----------------------------------------------------------------------------
# Simple 1‑iteration test helper
# ----------------------------------------------------------------------------

def test_aekf1_single_iteration() -> None:
    """Run one predict–update cycle with synthetic data and print results."""
    # Filter parameters
    dt = 0.1
    sig_eps = 0.001  # rad/s /√Hz
    sig_n = 1e-5     # rad/s² /√Hz
    sig_b = 0.001    # pixel or rad

    # Instantiate filter (identity quaternion, zero bias by default)
    kf = AEKF2(dt, sig_eps, sig_n, sig_b)

    # --- Predict step with a small gyro reading ---
    omega_meas = np.array([0.01, -0.02, 0.005])  # rad/s
    kf.predict(omega_meas)

    # --- Measurement update ---
    r_vector = np.array([1.0, 0.0, 0.0])         # inertial star direction
    true_q = np.array([0.0, 0.0, 0.0, 1.0])      # assume truth is still identity
    z_body = attitude_matrix(true_q) @ r_vector  # expected sight‑line → equals r
    z_body += np.random.normal(scale=sig_b, size=3)  # add simulated tracker noise

    kf.update(z_body, r_vector)

    # Print results
    print("\n=== AEKF‑1 single iteration test ===")
    print("State estimate x:", kf.x)
    print("Quaternion ‖q‖:", np.linalg.norm(kf.quaternion))
    print("Bias estimate μ:", kf.bias)
    print("Covariance diag:", np.diag(kf.P))

# Only run the test when this file is executed stand‑alone
if __name__ == "__main__":
    # np.set_printoptions(precision=4, suppress=True)
    test_aekf1_single_iteration()
