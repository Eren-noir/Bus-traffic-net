"""
Phase 4a: Build the edge-adjacency graph the GNN will operate on.

Our prediction targets are per-EDGE (road segment) speeds, so the graph
nodes are edges, not junctions. Two edges are connected if they share a
junction (i.e. traffic can flow from one directly into the other).

Must produce nodes in the SAME order as dataset.npz's `edges` array so the
adjacency matrix lines up with the feature matrices from prepare_dataset.py.
"""
import os
import sys
import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

HERE = os.path.dirname(__file__)
NET_DIR = os.path.join(HERE, "..", "network")


def build_adjacency():
    # Use the same edge ordering as the dataset (alphabetically sorted ids)
    data = np.load(os.path.join(HERE, "dataset.npz"), allow_pickle=True)
    edges_order = list(data["edges"])
    e_index = {e: i for i, e in enumerate(edges_order)}
    n = len(edges_order)

    net = sumolib.net.readNet(os.path.join(NET_DIR, "grid.net.xml"))

    A = np.zeros((n, n), dtype=np.float32)
    for edge in net.getEdges():
        eid = edge.getID()
        if eid not in e_index:
            continue
        i = e_index[eid]
        to_node = edge.getToNode()
        # neighbors = edges that START where this edge ENDS (can flow into)
        for out_edge in to_node.getOutgoing():
            oid = out_edge.getID()
            if oid in e_index:
                j = e_index[oid]
                A[i, j] = 1.0
                A[j, i] = 1.0  # symmetric: treat as undirected for message passing

    np.savez(os.path.join(HERE, "adjacency.npz"), A=A, edges=np.array(edges_order))
    degree = A.sum(axis=1)
    print(f"Graph: {n} nodes (edges), {int(A.sum()/2)} unique connections")
    print(f"Avg degree: {degree.mean():.1f}, isolated nodes: {(degree == 0).sum()}")
    return A


if __name__ == "__main__":
    build_adjacency()
