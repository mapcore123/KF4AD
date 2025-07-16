#!/usr/bin/env python3
"""Run attitude estimation simulations with several Extended Kalman Filters.

Usage:
    python run_simulation.py  # runs default single simulation of 6000 s

Graphs will be saved into the results/ directory.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sensors_sim import SimulationParams, simulate
from quaternion_utils import delta_angle
from filters import AEKF1, AEKF2, AEKF3, MEKF

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_once(filter_name: str):
    params = SimulationParams()

    # Noise parameters for filter (tuning)
    Qg = 1e-13
    Qb = 1e-11
    R_star = (params.sigma_star) ** 2

    if filter_name == "AEKF1":
        filt = AEKF1(Qg, Qb, R_star)
        state = filt.init_state(params.q0, params.mu0, np.eye(7) * 5.0)
    elif filter_name == "AEKF2":
        filt = AEKF2(Qg, Qb, R_star)
        state = filt.init_state(params.q0, params.mu0, np.eye(7) * 5.0)
    elif filter_name == "AEKF3":
        filt = AEKF3(Qg, Qb, R_star, sigma_nor=1e-6)
        state = filt.init_state(params.q0, params.mu0, np.eye(7) * 5.0)
    elif filter_name == "MEKF":
        filt = MEKF(Qg, Qb, R_star)
        filt.init_state(params.q0, params.mu0, np.eye(6) * 5.0)
    else:
        raise ValueError(filter_name)

    times = []
    ang_errors = []
    std_devs = []

    for t, q_true, mu_true, gyro_meas, star_meas, r1, r2 in simulate(params):
        if filter_name.startswith("AEKF"):
            # Propagate
            state = filt.propagate(state, gyro_meas, params.dt)
            # Update when star measurement available
            if star_meas is not None:
                r_vec = r1 if (int(t / params.star_tracker_period) % 2 == 0) else r2
                state = filt.update(state, star_meas, r_vec)
            q_est = state.q
            P = state.P
            # Quaternion variance approximate: take mean diag of orientation part
            sigma_q = np.sqrt(np.mean(np.diag(P)[:4]))
        else:
            filt.propagate(gyro_meas, params.dt)
            if star_meas is not None:
                r_vec = r1 if (int(t / params.star_tracker_period) % 2 == 0) else r2
                filt.update(star_meas, r_vec)
            q_est, _, P = filt.get_state()
            sigma_q = np.sqrt(np.mean(np.diag(P)[:3]))

        times.append(t)
        ang_errors.append(delta_angle(q_true, q_est))
        std_devs.append(sigma_q)

    # Plot
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(times, np.array(ang_errors) * 180 / np.pi)
    ax[0].set_ylabel("δφ [deg]")
    ax[0].set_title(f"Angular estimation error – {filter_name}")

    ax[1].plot(times, np.array(std_devs))
    ax[1].set_ylabel("√P (quaternion diag)")
    ax[1].set_xlabel("Time [s]")

    for a in ax:
        a.grid(True)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{filter_name.lower()}_run.png")
    fig.savefig(out_path)
    print(f"Saved graph to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    for fname in ["AEKF1", "AEKF2", "AEKF3", "MEKF"]:
        run_once(fname) 