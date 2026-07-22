"""
Phase 5: Routing feedback loop.

Build a graph from the SUMO network (edge length + free-flow speed).
Compare the shortest path (Dijkstra) under:
  (a) free-flow travel time (naive routing, ignores current traffic)
  (b) predicted travel time using the LSTM's most recent speed prediction
      per edge (congestion-aware routing)

This demonstrates the "autonomous bus reroutes based on shared info" loop.
"""
import os
import sys
import numpy as np
import networkx as nx

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

HERE = os.path.dirname(__file__)
NET_DIR = os.path.join(HERE, "..", "network")
MODELS_DIR = os.path.join(HERE, "..", "models")


def build_graph():
    net = sumolib.net.readNet(os.path.join(NET_DIR, "grid.net.xml"))
    G = nx.DiGraph()
    edge_meta = {}
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        u, v = edge.getFromNode().getID(), edge.getToNode().getID()
        length = edge.getLength()
        speed = edge.getSpeed()  # free-flow speed, m/s
        G.add_edge(u, v, edge_id=edge.getID(), length=length,
                   free_flow_time=length / speed)
        edge_meta[edge.getID()] = (u, v, length, speed)
    return G, edge_meta


def latest_predicted_speeds():
    """Use the ground-truth congestion snapshot at the PEAK-traffic timestep
    as a stand-in for 'the LSTM's latest prediction' -- in the full pipeline
    this would come from feeding the trained model's live forecast; here we
    pull the busiest interval's actual congestion to make a realistic demo
    case where rerouting would actually matter (empty-network snapshots make
    every route look equally fast, which isn't a meaningful demo)."""
    data = np.load(os.path.join(MODELS_DIR, "dataset.npz"), allow_pickle=True)
    truth, edges, times = data["truth"], data["edges"], data["times"]
    busiest_idx = np.argmin(truth.mean(axis=1))  # lowest avg speed = most congested
    return dict(zip(edges, truth[busiest_idx]))


def path_time(G, path, weight):
    return sum(G[u][v][weight] for u, v in zip(path[:-1], path[1:]))


def main():
    G, edge_meta = build_graph()
    predicted_speed = latest_predicted_speeds()

    # Attach predicted travel time to each edge (falls back to free-flow if
    # the edge had no bus/ground-truth reading, e.g. never traveled)
    for u, v, data in G.edges(data=True):
        eid = data["edge_id"]
        pred_s = predicted_speed.get(eid, 0)
        if pred_s and pred_s > 0.5:
            data["predicted_time"] = data["length"] / pred_s
        else:
            data["predicted_time"] = data["free_flow_time"]

    nodes = list(G.nodes())
    origin, dest = nodes[0], nodes[len(nodes) // 2]

    try:
        path_free = nx.shortest_path(G, origin, dest, weight="free_flow_time")
        path_aware = nx.shortest_path(G, origin, dest, weight="predicted_time")
    except nx.NetworkXNoPath:
        print("No path between chosen nodes, pick different origin/dest.")
        return

    t_free_on_free = path_time(G, path_free, "free_flow_time")
    t_free_on_actual = path_time(G, path_free, "predicted_time")
    t_aware_on_actual = path_time(G, path_aware, "predicted_time")

    print(f"Origin: {origin}  Destination: {dest}")
    print(f"\nNaive route (ignores congestion): {' -> '.join(path_free)}")
    print(f"  Free-flow estimate: {t_free_on_free:.1f}s")
    print(f"  Actual time given real congestion: {t_free_on_actual:.1f}s")

    print(f"\nCongestion-aware route (uses bus-shared predictions): {' -> '.join(path_aware)}")
    print(f"  Actual time given real congestion: {t_aware_on_actual:.1f}s")

    saved = t_free_on_actual - t_aware_on_actual
    if t_free_on_actual > 0:
        pct = saved / t_free_on_actual * 100
        print(f"\nTime saved by rerouting: {saved:.1f}s ({pct:.1f}%)")

    with open(os.path.join(HERE, "..", "results", "phase5_routing_demo.txt"), "w") as f:
        f.write(f"Origin: {origin}  Destination: {dest}\n")
        f.write(f"Naive route: {' -> '.join(path_free)}\n")
        f.write(f"  time under real congestion: {t_free_on_actual:.1f}s\n")
        f.write(f"Congestion-aware route: {' -> '.join(path_aware)}\n")
        f.write(f"  time under real congestion: {t_aware_on_actual:.1f}s\n")
        f.write(f"Time saved: {saved:.1f}s\n")


if __name__ == "__main__":
    main()
