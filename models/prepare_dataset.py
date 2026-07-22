"""
Phase 3a: Turn the sparse bus_reports.csv into a per-edge, per-timestep
feature matrix suitable for sequence modeling.

Since buses only cover a subset of edges/times, we:
  1. Pivot bus reports into an (edge x time) speed table.
  2. Forward-fill short gaps (a bus reported recently -> still informative).
  3. Leave genuinely unobserved (edge, time) cells as NaN and mask them out
     of the loss -- the model should learn to predict them from graph
     structure/time, not have them hallucinated by naive fill.
  4. Ground truth for supervision comes from edge_ground_truth.csv (dense).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

SEQ_LEN = 6       # look back 6 intervals (=3 min at 30s interval)
PRED_HORIZON = 1  # predict 1 interval ahead


def build():
    bus = pd.read_csv(os.path.join(DATA, "bus_reports.csv"))
    truth = pd.read_csv(os.path.join(DATA, "edge_ground_truth.csv"))

    times = sorted(truth["time"].unique())
    edges = sorted(truth["edge_id"].unique())
    t_index = {t: i for i, t in enumerate(times)}
    e_index = {e: i for i, e in enumerate(edges)}

    n_t, n_e = len(times), len(edges)

    # Sparse observed matrix (what buses actually reported), NaN elsewhere
    observed = np.full((n_t, n_e), np.nan, dtype=np.float32)
    for _, row in bus.iterrows():
        ti = t_index.get(row["time"])
        ei = e_index.get(row["edge_id"])
        if ti is not None and ei is not None:
            # multiple buses can report same edge/time -> average
            if np.isnan(observed[ti, ei]):
                observed[ti, ei] = row["speed"]
            else:
                observed[ti, ei] = (observed[ti, ei] + row["speed"]) / 2

    # Dense ground truth matrix (used only as prediction target / eval)
    truth_mat = np.zeros((n_t, n_e), dtype=np.float32)
    for _, row in truth.iterrows():
        ti = t_index.get(row["time"])
        ei = e_index.get(row["edge_id"])
        if ti is not None and ei is not None:
            truth_mat[ti, ei] = row["mean_speed"]

    # Forward-fill sparse observations along time per edge (limit 3 steps ~1.5min)
    filled = observed.copy()
    for ei in range(n_e):
        last_val = np.nan
        gap = 0
        for ti in range(n_t):
            if not np.isnan(observed[ti, ei]):
                last_val = observed[ti, ei]
                gap = 0
            else:
                gap += 1
                if gap <= 3 and not np.isnan(last_val):
                    filled[ti, ei] = last_val

    mask = ~np.isnan(filled)  # True where we have (filled) bus-derived info
    filled_for_model = np.nan_to_num(filled, nan=0.0)

    np.savez(
        os.path.join(HERE, "dataset.npz"),
        filled=filled_for_model,
        mask=mask,
        truth=truth_mat,
        edges=np.array(edges),
        times=np.array(times),
    )
    coverage = mask.mean()
    print(f"Grid shape: {n_t} timesteps x {n_e} edges")
    print(f"Bus-report coverage after fill: {coverage:.1%} of (edge,time) cells")
    return coverage


if __name__ == "__main__":
    build()
