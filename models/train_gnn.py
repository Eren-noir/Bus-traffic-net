"""
Phase 4b: Spatio-temporal GNN baseline upgrade.

The Phase 3 LSTM treats every edge independently -- an edge with zero bus
reports has nothing to go on but a zero-filled history. This model adds a
graph convolution step at every timestep so an edge can borrow signal from
its road-network neighbors (which may have been bus-observed) before the
LSTM does its temporal reasoning. This is the standard STGCN/DCRNN idea,
implemented here from scratch (no torch_geometric dependency) so it's easy
to read end-to-end.

Architecture per window:
  for each timestep t in the window:
      h_t = ReLU( A_hat @ x_t @ W )      <- 1-hop graph convolution
  h_1..h_T  ->  shared LSTM (per node)   <- temporal reasoning
  last hidden state -> linear head       <- predicted next-step speed

A_hat is the symmetric-normalized adjacency with self-loops (standard GCN
normalization): A_hat = D^-1/2 (A + I) D^-1/2
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
SEQ_LEN = 6
torch.manual_seed(42)
np.random.seed(42)


def normalize_adjacency(A):
    n = A.shape[0]
    A_self = A + np.eye(n, dtype=np.float32)
    deg = A_self.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=d_inv_sqrt)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_self @ D_inv_sqrt


def make_graph_windows(filled, mask, truth, seq_len=SEQ_LEN):
    """Unlike the per-edge LSTM, windows here are whole-graph snapshots:
    X: (n_windows, seq_len, n_nodes, 2), y: (n_windows, n_nodes)"""
    n_t, n_e = filled.shape
    X, y = [], []
    for ti in range(n_t - seq_len):
        speed_seq = filled[ti:ti + seq_len]           # (seq_len, n_nodes)
        mask_seq = mask[ti:ti + seq_len].astype(np.float32)
        window = np.stack([speed_seq, mask_seq], axis=-1)  # (seq_len, n_nodes, 2)
        target = truth[ti + seq_len]                    # (n_nodes,)
        X.append(window)
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_hat):
        # x: (batch, n_nodes, in_dim), A_hat: (n_nodes, n_nodes)
        agg = torch.einsum("ij,bjd->bid", A_hat, x)
        return F.relu(self.lin(agg))


class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_dim=2, gcn_hidden=16, lstm_hidden=32):
        super().__init__()
        self.gcn = GCNLayer(in_dim, gcn_hidden)
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x, A_hat):
        # x: (batch, seq_len, n_nodes, in_dim)
        b, t, n, _ = x.shape
        gcn_out = [self.gcn(x[:, step], A_hat) for step in range(t)]
        gcn_seq = torch.stack(gcn_out, dim=1)               # (b, t, n, gcn_hidden)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        lstm_out, _ = self.lstm(gcn_seq)
        last = lstm_out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred.reshape(b, n)


def main():
    data = np.load(os.path.join(HERE, "dataset.npz"))
    filled, mask, truth = data["filled"], data["mask"], data["truth"]
    adj = np.load(os.path.join(HERE, "adjacency.npz"))
    A_hat_np = normalize_adjacency(adj["A"])
    A_hat = torch.tensor(A_hat_np, dtype=torch.float32)

    n_t = filled.shape[0]
    split = int(n_t * 0.8)

    X_train, y_train = make_graph_windows(filled[:split], mask[:split], truth[:split])
    X_test, y_test = make_graph_windows(filled[split - SEQ_LEN:], mask[split - SEQ_LEN:], truth[split - SEQ_LEN:])

    Xtr, ytr = torch.tensor(X_train), torch.tensor(y_train)
    Xte, yte = torch.tensor(X_test), torch.tensor(y_test)

    model = SpatioTemporalGNN()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    persistence_pred = X_test[:, -1, :, 0]
    persistence_mae = np.mean(np.abs(persistence_pred - y_test))

    epochs = 100
    n = Xtr.shape[0]
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(Xtr, A_hat)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(Xte, A_hat)
                test_mae = torch.mean(torch.abs(test_pred - yte)).item()
            model.train()
            print(f"epoch {epoch+1:3d}  train_mse={loss.item():.4f}  test_mae={test_mae:.3f} m/s")

    model.eval()
    with torch.no_grad():
        test_pred = model(Xte, A_hat)
        test_mae = torch.mean(torch.abs(test_pred - yte)).item()
        test_rmse = torch.sqrt(torch.mean((test_pred - yte) ** 2)).item()

    print("\n=== GNN results (held-out last 20% of timesteps) ===")
    print(f"Persistence baseline MAE: {persistence_mae:.3f} m/s")
    print(f"GNN model MAE:            {test_mae:.3f} m/s")
    print(f"GNN model RMSE:           {test_rmse:.3f} m/s")

    # Compare specifically on edges that had NO bus coverage in the input
    # window -- this is where the GNN's neighbor-borrowing should matter
    # most versus the plain per-edge LSTM.
    never_observed = (mask[split - SEQ_LEN:split - SEQ_LEN + X_test.shape[0] + SEQ_LEN].sum(axis=0) == 0)
    if never_observed.any():
        sparse_mae = torch.mean(
            torch.abs(test_pred[:, never_observed] - yte[:, never_observed])
        ).item()
        print(f"GNN MAE on edges with zero bus coverage: {sparse_mae:.3f} m/s "
              f"({never_observed.sum()} such edges)")

    torch.save(model.state_dict(), os.path.join(HERE, "gnn_model.pt"))

    with open(os.path.join(HERE, "..", "results", "phase4_metrics.txt"), "w") as f:
        f.write("Phase 4 GNN results\n")
        f.write(f"Persistence baseline MAE: {persistence_mae:.3f} m/s\n")
        f.write(f"LSTM baseline MAE (from Phase 3): see phase3_metrics.txt\n")
        f.write(f"GNN model MAE:            {test_mae:.3f} m/s\n")
        f.write(f"GNN model RMSE:           {test_rmse:.3f} m/s\n")


if __name__ == "__main__":
    main()
