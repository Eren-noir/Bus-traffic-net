# 🚌 Bus-Shared Traffic Prediction & Rerouting

Autonomous buses acting as mobile traffic sensors: a subset of vehicles in a
simulated city broadcast **sparse** (position, speed) reports, a spatio-temporal
ML model predicts full road-network traffic state from that sparse signal, and
buses reroute using the predictions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![SUMO](https://img.shields.io/badge/SUMO-1.18-green)
![Status](https://img.shields.io/badge/status-active--development-yellow)

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Quickstart](#quickstart)
- [Repo structure](#repo-structure)
- [Roadmap](#roadmap)

## Overview

Real autonomous buses can't see the whole city's traffic — only what they've
personally driven past recently. This project simulates that constraint
honestly (buses cover ~13% of vehicles and only ~23% of road-segment/time
cells end up with any signal at all) and asks: **how much of the full traffic
state can a model reconstruct from that sparse, bus-only view — and does
knowing the road network's graph structure actually help?**

Built with [SUMO](https://sumo.dlr.de) (traffic simulation), [TraCI](https://sumo.dlr.de/docs/TraCI.html)
(live simulation control), and PyTorch (LSTM + a from-scratch spatio-temporal GNN).

## Architecture

```
SUMO grid city  ──▶  bus fleet (13% of vehicles)  ──▶  sparse reports (TraCI)
                                                              │
                                                              ▼
                                              edge×time matrix (23% coverage)
                                                              │
                                  ┌───────────────────────────┴───────────────────────────┐
                                  ▼                                                        ▼
                     per-edge LSTM (Phase 3)                          graph conv + shared LSTM (Phase 4)
                                  │                                                        │
                                  └───────────────────────────┬───────────────────────────┘
                                                              ▼
                                              predicted next-interval speed
                                                              │
                                                              ▼
                                        Dijkstra reroute vs. free-flow route
```

## Results

**Prediction (held out on the last 20% of timesteps):**

| Model | Overall MAE (m/s) | MAE on zero-bus-coverage edges (m/s) |
|---|---|---|
| Persistence (naive) | 11.42 | — |
| LSTM (per-edge) | 0.81 | 0.735 |
| GNN (graph-aware) | 0.81 | 0.728 |

The GNN essentially ties the LSTM rather than beating it decisively — an
honest negative result, not a bug. At this network size (80 edges, 30 min
sim) most unobserved edges are still near free-flow speed, so there's little
room for neighbor-borrowing to show value. See
[`results/phase4_metrics.txt`](results/phase4_metrics.txt) for the full
writeup and the follow-up experiment that would test this properly.

> The persistence baseline's MAE is inflated by unobserved cells defaulting
> to 0 — a fairer baseline is on the todo list (see [Roadmap](#roadmap)).

**Routing:** at peak congestion, the congestion-aware route saved **~5.7–17%**
travel time over the naive shortest path (varies by run/congestion snapshot —
see [`results/phase5_routing_demo.txt`](results/phase5_routing_demo.txt)).

## Quickstart

**Requirements:** SUMO (`sumo --version` should work, with `SUMO_HOME` set),
Python 3.10+, and `pip install traci sumolib torch networkx numpy pandas`.

```bash
export SUMO_HOME=/usr/share/sumo   # adjust to your install path

cd network
netgenerate --grid --grid.number=5 --grid.length=200 \
  --default.lanenumber=2 --output-file=grid.net.xml
python3 $SUMO_HOME/tools/randomTrips.py -n grid.net.xml -o trips.trips.xml -e 1800 -p 2.0 --seed 42
python3 make_bus_fleet.py

cd ../data
python3 collect_bus_reports.py

cd ../models
python3 prepare_dataset.py
python3 train_lstm.py
python3 build_graph.py
python3 train_gnn.py
python3 compare_on_sparse_edges.py

cd ../routing
python3 reroute_demo.py
```

## Repo structure

```
network/    SUMO city grid, trip/bus generation, sim config
data/       TraCI-collected bus reports + dense ground truth
models/     dataset prep, edge-adjacency graph, LSTM + GNN training
routing/    Dijkstra rerouting demo using model predictions
results/    saved metrics/output from each phase
```

## Roadmap

- [ ] **Validate the GNN at scale** — re-run on a larger network
      (`--grid.number=10`) and/or a longer rush-hour-style simulation to see
      whether graph structure helps more when congestion is spatially
      correlated rather than mostly free-flow.
- [ ] **Live feedback loop** — currently the routing demo uses a saved
      congestion snapshot as a stand-in for the model's live forecast; wire
      it up to the actual trained model and re-run periodically inside a
      TraCI simulation loop so buses genuinely reroute mid-simulation.
- [ ] **Fairer persistence baseline** — predict free-flow speed instead of 0
      for never-observed cells, so the headline MAE comparison isn't
      inflated.
- [ ] Architecture diagram as an image + a short `sumo-gui` screen capture
      showing rerouted buses.

---
Built as a portfolio project exploring sparse-sensor traffic prediction and
graph learning.
