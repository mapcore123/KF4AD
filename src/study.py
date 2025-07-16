#!/usr/bin/env python3
"""Comprehensive study script covering all simulation tasks required in the project.

Tasks implemented (see assignment Part 3):
1. Compare AEKF1/2/3 & MEKF with nominal initial conditions.
2. Compare the same four filters with large initial error.
3. Monte-Carlo (N=50) with best filter (MEKF) and nominal initial conditions.
4. Sensitivity to initial covariance P₀ (0.001 I, I, 100 I).
5. Sensitivity to star-tracker sampling period (1, 5, 25 s).

Output figures are placed under results/ with descriptive filenames.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# multiprocessing removed for compatibility

from sensors_sim import SimulationParams, simulate
from quaternion_utils import delta_angle
from filters import AEKF1, AEKF2, AEKF3, MEKF, FilterState

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILTER_SPECS = {
    "AEKF1": AEKF1,
    "AEKF2": AEKF2,
    "AEKF3": AEKF3,
    "MEKF": MEKF,
}

# Default tuning – same for all filters (can be refined manually)
QG = 1e-13
QB = 1e-11

def _create_filter(name: str, R_star: float, sigma_nor: float = 1e-6):
    cls = FILTER_SPECS[name]
    if name == "AEKF3":
        return cls(QG, QB, R_star, sigma_nor=sigma_nor)
    else:
        return cls(QG, QB, R_star)


def run_single(filter_name: str, params: SimulationParams, P0_scale: float = 5.0, seed: int | None = None):
    """Run one simulation and return time, δφ, filter σ arrays."""
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    R_star = params.sigma_star ** 2
    filt = _create_filter(filter_name, R_star)

    if filter_name.startswith("AEKF"):
        P0 = np.eye(7) * P0_scale
        state = filt.init_state(params.q0, params.mu0, P0)
    else:
        P0 = np.eye(6) * P0_scale
        filt.init_state(params.q0, params.mu0, P0)

    times, err, sig = [], [], []
    for t, q_true, mu_true, gyro_meas, star_meas, r1, r2 in simulate(params):
        if filter_name.startswith("AEKF"):
            state = filt.propagate(state, gyro_meas, params.dt)
            if star_meas is not None:
                r_vec = r1 if (int(t / params.star_tracker_period) % 2 == 0) else r2
                state = filt.update(state, star_meas, r_vec)
            q_est = state.q
            P = state.P
            val = float(np.mean(np.diag(P)[:4]))
            sigma = np.sqrt(val if val > 0.0 else 0.0)
        else:
            filt.propagate(gyro_meas, params.dt)
            if star_meas is not None:
                r_vec = r1 if (int(t / params.star_tracker_period) % 2 == 0) else r2
                filt.update(star_meas, r_vec)
            q_est, _, P = filt.get_state()
            val2 = float(np.mean(np.diag(P)[:3]))
            sigma = np.sqrt(val2 if val2 > 0.0 else 0.0)

        times.append(t)
        err.append(delta_angle(q_true, q_est))
        sig.append(sigma)

    if seed is not None:
        np.random.set_state(rng_state)  # restore
    return np.asarray(times), np.asarray(err), np.asarray(sig)


# -----------------------------------------------------------------------------
# 1 & 2 — Multi-filter comparisons (two sets of initial conditions)
# -----------------------------------------------------------------------------

def compare_filters(initial_case: int):
    if initial_case == 1:
        params = SimulationParams()
        tag = "init1"
    else:
        params = SimulationParams()
        # Set large initial error quaternion & bias (from assignment)
        params.q0 = np.array([0.3780, 0.7560, 0.3780, -0.3780])
        params.mu0 = 200 * np.pi / 180.0 / 3600.0 * np.array([1.0, 1.0, 1.0])  # 200 deg/hr → rad/s
        tag = "init2"

    for name in FILTER_SPECS.keys():
        t, e, s = run_single(name, params)
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax[0].plot(t, np.degrees(e))
        ax[0].set_ylabel("δφ [deg]")
        ax[0].set_title(f"Angular error – {name} ({tag})")
        ax[1].plot(t, s)
        ax[1].set_ylabel("√P")
        ax[1].set_xlabel("Time [s]")
        for a in ax:
            a.grid(True)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"{name.lower()}_{tag}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")


# -----------------------------------------------------------------------------
# 3 — Monte Carlo study with best filter (MEKF assumed best)
# -----------------------------------------------------------------------------

def monte_carlo(best_filter: str = "MEKF", N: int = 50):
    params = SimulationParams()
    R_star = params.sigma_star ** 2
    times = None
    errs = []
    sigs = []

    # Sequential execution to avoid multiprocessing pickling issues
    for seed in range(N):
        t, e, s = run_single(best_filter, params, seed=seed)
        if times is None:
            times = t
        errs.append(e)
        sigs.append(s)

    errs = np.vstack(errs)
    sigs = np.vstack(sigs)

    # MC statistics
    mean_err = np.mean(errs, axis=0)
    std_err = np.std(errs, axis=0, ddof=0)
    mean_sig = np.mean(sigs, axis=0)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(times, np.degrees(mean_err))
    ax[0].set_ylabel("MC mean δφ [deg]")
    ax[0].set_title(f"Monte-Carlo mean – {best_filter} (N={N})")

    ax[1].plot(times, std_err, label="MC σ(δφ)")
    ax[1].plot(times, mean_sig, label="Filter σ")
    ax[1].set_ylabel("Std dev [rad]")
    ax[1].set_xlabel("Time [s]")
    ax[1].legend()

    for a in ax:
        a.grid(True)

    fig.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"mc_{best_filter.lower()}.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"Saved {fname}")


# -----------------------------------------------------------------------------
# 4 — Initial covariance sensitivity
# -----------------------------------------------------------------------------

def covariance_sensitivity(best_filter: str = "MEKF"):
    params = SimulationParams()
    scales = [0.001, 1.0, 100.0]
    for scale in scales:
        t, e, s = run_single(best_filter, params, P0_scale=scale)
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax[0].plot(t, np.degrees(e))
        ax[0].set_ylabel("δφ [deg]")
        ax[0].set_title(f"Initial P0 = {scale} I – {best_filter}")
        ax[1].plot(t, s)
        ax[1].set_ylabel("√P")
        ax[1].set_xlabel("Time [s]")
        for a in ax:
            a.grid(True)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"sensP_{best_filter.lower()}_{scale}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")


# -----------------------------------------------------------------------------
# 5 — Star-tracker period sensitivity
# -----------------------------------------------------------------------------

def startracker_sensitivity(best_filter: str = "MEKF"):
    periods = [1.0, 5.0, 25.0]
    for T in periods:
        params = SimulationParams(star_tracker_period=T)
        t, e, s = run_single(best_filter, params)
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax[0].plot(t, np.degrees(e))
        ax[0].set_ylabel("δφ [deg]")
        ax[0].set_title(f"Star-tracker Δt = {T}s – {best_filter}")
        ax[1].plot(t, s)
        ax[1].set_ylabel("√P")
        ax[1].set_xlabel("Time [s]")
        for a in ax:
            a.grid(True)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"sensSTR_{best_filter.lower()}_{int(T)}s.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running full study – results will be saved to 'results/' folder…")
    compare_filters(1)
    compare_filters(2)
    monte_carlo()
    covariance_sensitivity()
    startracker_sensitivity()
    print("Study complete.") 