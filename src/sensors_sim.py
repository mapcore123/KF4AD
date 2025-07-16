import numpy as np
from dataclasses import dataclass

from quaternion_utils import propagate, normalize, to_dcm

DEG2RAD = np.pi / 180.0
ARCSEC2RAD = DEG2RAD / 3600.0

@dataclass
class SimulationParams:
    dt: float = 0.1  # time step [s]
    duration: float = 6000.0  # total simulation time [s]
    star_tracker_period: float = 2.0  # seconds between star‐tracker samples

    # Sensor noise parameters
    sigma_gyro: float = np.sqrt(1e-13)  # rad/sqrt(s)
    sigma_gyro_bias: float = np.sqrt(1e-11)  # rad/sqrt(s^3)
    sigma_star: float = 100.0 * ARCSEC2RAD  # rad

    # Initial conditions
    q0: np.ndarray = np.array([0.3780, -0.3780, 0.7560, 0.3780])
    mu0: np.ndarray = 3e-5 * np.array([1.0, 1.0, 1.0])  # rad/s

    def reference_vectors(self):
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, 1.0, 0.0])
        return r1, r2


def true_angular_velocity(t: float) -> np.ndarray:
    """Return true angular velocity in rad/s at time t [s]."""
    return np.sin(2 * np.pi * t / 150.0) * np.array([1.0, 1.0, 1.0]) * DEG2RAD


def simulate(params: SimulationParams):
    """Generator that yields time, true state, gyro measurement, star measurement when available."""
    dt = params.dt
    steps = int(params.duration / dt)
    q = normalize(params.q0.copy())
    mu = params.mu0.copy()

    # Precompute noise covariances for discrete time
    Q_mu = params.sigma_gyro_bias ** 2 * dt  # variance for bias random walk per axis

    r1, r2 = params.reference_vectors()
    star_period_steps = int(params.star_tracker_period / dt)

    for k in range(steps):
        t = k * dt
        omega_true = true_angular_velocity(t)

        # Update true bias mu (random walk)
        mu += np.random.randn(3) * np.sqrt(Q_mu)

        # Gyro measurement
        gyro_measure = omega_true + mu + np.random.randn(3) * params.sigma_gyro

        # Propagate true quaternion
        q = propagate(q, omega_true, dt)

        # Star tracker measurement if time
        star_meas = None
        if k % star_period_steps == 0:
            # Choose star alternately
            r = r1 if (k // star_period_steps) % 2 == 0 else r2
            bo_true = to_dcm(q) @ r
            star_meas = bo_true + np.random.randn(3) * params.sigma_star
        yield t, q.copy(), mu.copy(), gyro_measure, star_meas, r1, r2 