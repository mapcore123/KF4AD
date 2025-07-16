import numpy as np
from ofer.matrices import *

# ----------------------------------------------------------------------------
# MEKF class 
# ----------------------------------------------------------------------------
'''
need to verify the order of hamilton product dqxq or qxdq.
need to verify if H is -Aq[r]_x or [Aqr]x
'''
class MEKF:
    def __init__(self, dt: float, sigma_eps: float, sigma_n: float, sigma_b: float,
                 q0: np.ndarray | None = None, mu0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        self.dt = dt
        
        # process and measurement covariances
        self.Q = np.diag([sigma_eps**2]*3 + [sigma_n**2]*3)   # 6×6 cont.
        self.R  = (sigma_b**2) * np.eye(3)                     # 3×3
        # nominal state (NOT inside the KF state)
        self.q  = (q0  if q0  is not None else np.array([0,0,0,1.])).reshape(4,1) # shape (4,1)
        self.mu = (mu0 if mu0 is not None else np.zeros((3,1))) # shape (3,1)
        # Kalman error-state covariance 6×6
        self.P  = (P0  if P0  is not None else np.eye(6))

    def predict(self, omega_m: np.ndarray) -> None:
        '''
        Predict step. omega_m is the measured angular velocity vector [w1,w2,w3]
        '''
        
        # propagate nominal state
        omega_m = omega_m.reshape(3,1)
        omega_tilde = omega_m - self.mu # dims (3,1)
        dq = 0.5 * omega_matrix(omega_tilde) @ self.q # dims (4,1)
        self.q = self.q + dq * self.dt # dims (4,1)
        self.q = self.q / np.linalg.norm(self.q) # dims (4,1) - should renormalize?
        
        # propagate nominal drifts
        # nothing to do for constant-bias model
        
        # propagate covariance
        # Jacobian matrices F and G
        F = mekf_F_matrix(omega_tilde) # dims (6,6)
        G = mekf_Gt_matrix() # dims (6,6)
        
        # Linear perturbation matrices Phi and Gamma
        Phi = np.eye(6) + F * self.dt # dims (6,6)
        Gamma = G * self.dt # dims (6,6)
        
        # Covariance prediction step P_k+1|k
        self.P = Phi @ self.P @ Phi.T + additive_Q(self.Q, Gamma, self.dt) # dims (6,6)


    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        '''
        Update step. z is the measured sight-line vector [x,y,z]
        '''
        
        z = z.reshape(3,1)
        r = r.reshape(3,1)
        
        # nominal measurement
        b_hat = attitude_matrix(self.q) @ r # dims (3,1) 
        
        # Innovation vector y
        y = z - b_hat # dims (3,1)
        
        # Jacobian matrix H
        H = mekf_H(b_hat) # dims (3,6)

        # Innovation covariance S
        S = H @ self.P @ H.T + self.R # dims (3,3)
        
        # Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(S) # dims (6,3)
        
        # State update step x_k+1|k+1
        delta_x = K @ y # dims (6,1)
        
        # split correction
        delta_e  = delta_x[:3]                     # 3×1
        delta_mu = delta_x[3:]                     # 3×1

        # ----- 3-d  multiplicative quaternion correction ----------
        dq = np.vstack((0.5*delta_e, np.array([[1.]])))   # 4×1
        self.q = quaternion_multiply(dq, self.q)
        self.q = self.q / np.linalg.norm(self.q)
        
        # ----- 3-e  drift correction ------------------------------
        self.mu += delta_mu

        # ----- 3-f  covariance update & reset ---------------------
        I6 = np.eye(6)
        self.P = (I6 - K @ H) @ self.P  @ (I6 - K @ H).T + K @ self.R @ K.T
        
        # reset mean error to zero (implicit because we don't store it)
        

    @property
    def quaternion(self):
        return self.q.copy()

    @property
    def bias(self):
        return self.mu.copy()

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
    kf = MEKF(dt, sig_eps, sig_n, sig_b)

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
    # print("State estimate x:", kf.x)
    print("Quaternion ‖q‖:", np.linalg.norm(kf.quaternion))
    print("Bias estimate μ:", kf.bias)
    print("Covariance diag:", np.diag(kf.P))

# Only run the test when this file is executed stand‑alone
if __name__ == "__main__":
    # np.set_printoptions(precision=4, suppress=True)
    test_aekf1_single_iteration()
