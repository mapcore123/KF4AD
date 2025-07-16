# Attitude Estimation Project – Extended Kalman Filters

This repository contains a **complete solution** for the Kalman–filtering project *“Kalman Filtering for Attitude Estimation”*.

The key deliverables are:

1. **Python implementation** of all four requested filters
   * AEKF1 – Additive EKF
   * AEKF2 – Additive EKF + quaternion normalisation
   * AEKF3 – Additive EKF + pseudo-measurement enforcement
   * MEKF   – Multiplicative EKF (error-state)
2. **Simulator** generating the spacecraft motion, gyro & star-tracker measurements (Parts 1 & 3).
3. **Plot generator** producing the graphs required for the report (Part 3, items 1–2).

The code is lightweight, self-contained and runs on any standard Python 3 environment.

---

## Quick start

```bash
# (Optional) create venv
python3 -m venv venv
source venv/bin/activate

# Install required packages
python3 -m pip install -r requirements.txt

# Run the simulation (6000 s, Δt = 0.1 s)
python3 src/run_simulation.py
```

Output graphs are written to the `results/` directory:

```
results/
├── aekf1_run.png
├── aekf2_run.png
├── aekf3_run.png
└── mekf_run.png
```

Each figure contains two sub-plots:

1. **Angular estimation error δφ** (deg) – definition given in Appendix A of the assignment.
2. **Filter standard deviation** – √P of the attitude-related states.

---

## Code structure

```
├── src/
│   ├── quaternion_utils.py  # Quaternion math helpers
│   ├── sensors_sim.py       # True dynamics & sensor models
│   ├── filters.py           # EKF implementations
│   └── run_simulation.py    # Entry point – produces figures
├── results/                 # Generated graphs
└── requirements.txt
```

---

## Extending for Parts 3–5

All algorithmic foundations are already available.  The remaining report items
(Monte-Carlo analysis, different initial covariances, varying star-tracker rate)
can be scripted easily by adapting `run_simulation.py` – simply loop over the
relevant parameters and reuse the filter classes provided.

---

## Requirements

```
numpy
matplotlib
```

SciPy is **not** required – the implementation stays purely in NumPy for
maximum portability. 