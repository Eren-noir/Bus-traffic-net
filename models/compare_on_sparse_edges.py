"""
Phase 4c: Apples-to-apples comparison.

train_lstm.py reports overall MAE across all edges. train_gnn.py additionally
reports MAE on edges with ZERO bus coverage in the input window -- that's
the specific case the graph structure should help with (an isolated LSTM has
nothing but zeros to go on there; a GNN can borrow from observed neighbors).

This script loads the already-trained LSTM and evaluates it on that exact
same subset of edges, so the two numbers are directly comparable.
"""
import os
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
SEQ_LEN = 6


class SegmentLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_windows_for_edges(filled, mask, truth, edge_idxs, seq_len=SEQ_LEN):
    n_t = filled.shape[0]
    X, y = [], []
    for ei in edge_idxs:
        for ti in range(n_t - seq_len):
            speed_seq = filled[ti:ti + seq_len, ei]
            mask_seq = mask[ti:ti + seq_len, ei].astype(np.float32)
            target = truth[ti + seq_len, ei]
            X.append(np.stack([speed_seq, mask_seq], axis=-1))
            y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    data = np.load(os.path.join(HERE, "dataset.npz"))
    filled, mask, truth = data["filled"], data["mask"], data["truth"]
    n_t = filled.shape[0]
    split = int(n_t * 0.8)

    # same "never observed in the relevant window" definition as train_gnn.py
    window_mask = mask[split - SEQ_LEN:]
    never_observed = np.where(window_mask.sum(axis=0) == 0)[0]
    if len(never_observed) == 0:
        print("No zero-coverage edges found in this split -- nothing to compare.")
        return

    X_test, y_test = make_windows_for_edges(
        filled[split - SEQ_LEN:], mask[split - SEQ_LEN:], truth[split - SEQ_LEN:],
        never_observed,
    )

    model = SegmentLSTM()
    model.load_state_dict(torch.load(os.path.join(HERE, "segment_lstm.pt")))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test))
        mae = torch.mean(torch.abs(pred - torch.tensor(y_test))).item()

    print(f"Zero-coverage edges in this split: {len(never_observed)}")
    print(f"Plain LSTM MAE on those edges:  {mae:.3f} m/s")
    print("(compare to the GNN's reported number on the same edge set)")


if __name__ == "__main__":
    main()
