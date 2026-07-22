# Bus-Shared Traffic Prediction & Rerouting

Autonomous buses acting as mobile traffic sensors: a subset of vehicles in a
simulated city broadcast sparse (position, speed) reports, an ML model
predicts full road-network traffic state from those sparse reports, and
buses reroute using the predictions.

## What's implemented (this session)

**Phase 1 — Environment**
`network/` — a 5x5 grid city generated with SUMO's `netgenerate`, populated
with ~900 random vehicle trips (`randomTrips.py`) over a 30-minute window.

**Phase 2 — Bus agents & data collection**
`network/make_bus_fleet.py` tags ~13% of vehicles as `bus` type.
`data/collect_bus_reports.py` drives the simulation via TraCI and logs:
- `bus_reports.csv` — sparse (edge, speed, time) reports, **buses only**
  (this is the "shared traffic info" — what the network actually has to
  work with)
- `edge_ground_truth.csv` — dense per-edge speed from *all* vehicles, used
  only for training supervision and evaluation, never as model input

**Phase 3 — Baseline prediction model**
`models/prepare_dataset.py` pivots sparse bus reports into an (edge x time)
matrix with a short forward-fill and an explicit "was this bus-observed"
mask channel — the model has to know what it actually saw vs. what's
filled-in.
`models/train_lstm.py` trains a weight-shared per-edge LSTM (2 input
features: filled speed + observed-flag) to predict next-interval speed.

Results on held-out (last 20%) timesteps:
| Model | MAE (m/s) |
|---|---|
| Persistence (last observed speed) | 11.42 |
| LSTM | 0.81 |

*Caveat worth knowing for a portfolio writeup:* the persistence baseline's
MAE is inflated because many (edge, time) cells never got a bus report and
default to 0 — a fairer baseline would be "predict free-flow speed when
no bus data exists." Worth swapping in before presenting this number
publicly; see Roadmap.

**Phase 5 — Routing feedback loop**
`routing/reroute_demo.py` builds a graph from the SUMO network, then
compares Dijkstra shortest paths under free-flow weights vs. predicted
(bus-informed) congestion weights. At the peak-congestion snapshot, the
congestion-aware route saves ~5.7% travel time over the naive route on this
small network — bigger/denser networks should show more.

**Phase 4 — GNN upgrade**
`models/build_graph.py` builds an edge-adjacency graph (edges are graph
nodes; two edges are neighbors if traffic can flow directly from one into
the other), aligned to the same edge ordering as the dataset — 80 nodes,
228 connections, avg degree 5.7.

`models/train_gnn.py` implements a small spatio-temporal GNN from scratch
(no torch_geometric dependency): at each timestep, a 1-hop graph
convolution lets an edge's features mix with its neighbors' before a
shared LSTM does the temporal reasoning. Same train/test split as Phase 3.

**Results — and an honest negative finding:**
| Model | Overall MAE (m/s) | MAE on zero-bus-coverage edges (m/s) |
|---|---|---|
| Persistence | 11.42 | — |
| LSTM (Phase 3) | 0.81 | 0.735 |
| GNN (Phase 4) | 0.81 | 0.728 |

`models/compare_on_sparse_edges.py` runs the head-to-head on exactly the
edges where graph structure should help most (edges with zero bus reports
in the input window — the plain LSTM has nothing but zeros to go on there).
The GNN essentially ties the LSTM rather than beating it decisively.

This is a real result worth stating honestly rather than dressing up: at
this network size (80 edges) and simulation length (30 min), most
"unobserved" edges are still near free-flow speed most of the time, so
zero-filling already gets close — there isn't much room for neighbor-
borrowing to show its value. See `results/phase4_metrics.txt` for the full
writeup. The next experiment (larger/denser network, or a longer sim with
rush-hour-style congestion waves) is the honest way to find out whether the
GNN's advantage shows up when the underlying traffic pattern actually needs
spatial correlation — that's a legitimate "next step" to describe on a
portfolio rather than overselling the current numbers.

## Roadmap (not yet built)

**Phase 4 extension — validate GNN's advantage at scale**
Re-run Phase 1–4 on a larger network (e.g. `--grid.number=10`) and/or a
longer simulation with a rush-hour traffic profile, to test whether the
GNN's neighbor-borrowing shows a clearer advantage when congestion is
spatially correlated rather than mostly free-flow.

**Phase 5 extension — live feedback loop**
Currently `reroute_demo.py` uses a ground-truth snapshot as a stand-in for
"the model's latest prediction." Wire it up to actually call the trained
LSTM/GNN's live forecast instead, and re-run periodically during a TraCI
simulation loop (rather than a single before/after snapshot) so buses
genuinely reroute mid-simulation.

**Phase 6 — Polish**
- Architecture diagram (data flow: SUMO -> bus reports -> model -> routing)
- Record a short screen capture of `sumo-gui` showing rerouted buses
- Push to GitHub with this README, add metrics table and the GNN comparison
- Short LinkedIn post: sparse-sensor traffic prediction + graph learning

## Repo structure
```
network/    SUMO city grid, trip/bus generation, sim config
data/       TraCI-collected bus reports + ground truth
models/     dataset prep + LSTM training
routing/    Dijkstra rerouting demo using predictions
results/    saved metrics/output from each phase
```

## Reproducing
```bash
export SUMO_HOME=/usr/share/sumo
cd network && netgenerate --grid --grid.number=5 --grid.length=200 \
  --default.lanenumber=2 --output-file=grid.net.xml
python3 $SUMO_HOME/tools/randomTrips.py -n grid.net.xml -o trips.trips.xml -e 1800 -p 2.0 --seed 42
python3 make_bus_fleet.py
cd ../data && python3 collect_bus_reports.py
cd ../models && python3 prepare_dataset.py && python3 train_lstm.py
python3 build_graph.py && python3 train_gnn.py && python3 compare_on_sparse_edges.py
cd ../routing && python3 reroute_demo.py
```
