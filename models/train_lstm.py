"""
Phase 3b: LSTM baseline.

One shared LSTM is applied across every edge's time series (weight-sharing,
like a per-segment model). Input per step: [bus_derived_speed, was_observed_flag].
Target: ground-truth mean speed at t+1 (from ALL vehicles, not just buses).

Train/test split is by TIME (first 80% of timesteps train, last 20% test)
to avoid leaking the future into training.
"""
import os
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
SEQ_LEN = 6
torch.manual_seed(42)
np.random.seed(42)


def make_windows(filled, mask, truth, seq_len=SEQ_LEN):
    n_t, n_e = filled.shape
    X, y = [], []
    for ei in range(n_e):
        for ti in range(n_t - seq_len):
            speed_seq = filled[ti:ti + seq_len, ei]
            mask_seq = mask[ti:ti + seq_len, ei].astype(np.float32)
            target = truth[ti + seq_len, ei]
            X.append(np.stack([speed_seq, mask_seq], axis=-1))  # (seq_len, 2)
            y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SegmentLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def main():
    data = np.load(os.path.join(HERE, "dataset.npz"))
    filled, mask, truth = data["filled"], data["mask"], data["truth"]
    n_t = filled.shape[0]
    split = int(n_t * 0.8)

    X_train, y_train = make_windows(filled[:split], mask[:split], truth[:split])
    X_test, y_test = make_windows(filled[split - SEQ_LEN:], mask[split - SEQ_LEN:], truth[split - SEQ_LEN:])

    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train)
    Xte = torch.tensor(X_test)
    yte = torch.tensor(y_test)

    model = SegmentLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # naive baseline: predict "last observed speed" (persistence model)
    persistence_pred = X_test[:, -1, 0]
    persistence_mae = np.mean(np.abs(persistence_pred - y_test))

    epochs = 60
    batch_size = 256
    n = Xtr.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(Xte)
                test_mae = torch.mean(torch.abs(test_pred - yte)).item()
            model.train()
            print(f"epoch {epoch+1:3d}  train_mse={total_loss/n:.4f}  test_mae={test_mae:.3f} m/s")

    model.eval()
    with torch.no_grad():
        test_pred = model(Xte)
        test_mae = torch.mean(torch.abs(test_pred - yte)).item()
        test_rmse = torch.sqrt(torch.mean((test_pred - yte) ** 2)).item()

    print("\n=== Final results (held-out last 20% of timesteps) ===")
    print(f"Persistence baseline MAE: {persistence_mae:.3f} m/s")
    print(f"LSTM model MAE:           {test_mae:.3f} m/s")
    print(f"LSTM model RMSE:          {test_rmse:.3f} m/s")
    improvement = (1 - test_mae / persistence_mae) * 100
    print(f"Improvement over persistence: {improvement:.1f}%")

    torch.save(model.state_dict(), os.path.join(HERE, "segment_lstm.pt"))

    with open(os.path.join(HERE, "..", "results", "phase3_metrics.txt"), "w") as f:
        f.write("Phase 3 baseline results\n")
        f.write(f"Persistence baseline MAE: {persistence_mae:.3f} m/s\n")
        f.write(f"LSTM model MAE:           {test_mae:.3f} m/s\n")
        f.write(f"LSTM model RMSE:          {test_rmse:.3f} m/s\n")
        f.write(f"Improvement over persistence: {improvement:.1f}%\n")


if __name__ == "__main__":
    main()
