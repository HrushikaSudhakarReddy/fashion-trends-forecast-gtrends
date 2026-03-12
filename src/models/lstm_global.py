# src/models/lstm_global.py
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from src.utils.io import data_path, ensure_dir
from src.utils.config import CONFIG

H = CONFIG.get("horizon_weeks", 12)

class GlobalLSTM(nn.Module):
    def __init__(self, in_dim=4, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, out_steps=H):
        # x: [B, T, F]
        B, T, F = x.shape
        h, _ = self.lstm(x)
        last = h[:, -1, :]  # [B, hidden]
        # naive: repeat same prediction H times (or unroll with teacher forcing if you want)
        y = self.fc(last)   # [B, 1]
        yseq = y.repeat(1, out_steps)  # [B, H]
        return yseq

def build_windows(df, lookback=52):
    # df columns: trend_id, week, combined_signal, is_ss, is_aw, novelty
    feats = []
    targets = []
    groups = []
    for tid, g in df.groupby("trend_id"):
        g = g.sort_values("week")
        # inputs: [combined_signal, is_ss, is_aw, novelty]
        X = g[["combined_signal","is_ss","is_aw","novelty"]].to_numpy(dtype=np.float32)
        if len(X) < lookback + 1: 
            continue
        feats.append(X[-lookback:])
        targets.append(X[-1,0])  # last value as anchor (not used here)
        groups.append(tid)
    if not feats:
        return None, None, []
    Xb = np.stack(feats, axis=0)  # [B, T, F]
    yb = np.array(targets, dtype=np.float32)
    return torch.tensor(Xb), torch.tensor(yb), groups

def run(epochs=20, lr=1e-3, lookback=52):
    df = pd.read_csv(data_path("processed","trend_features.csv"), parse_dates=["week"])
    for c in ["combined_signal","is_ss","is_aw","novelty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X, y, groups = build_windows(df, lookback=lookback)
    if X is None:
        print("Not enough data to train LSTM.")
        return

    model = GlobalLSTM(in_dim=X.shape[-1], hidden=64, layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        yhat_seq = model(X, out_steps=H)  # [B, H]
        # simple loss: match the last observed target repeated into horizon
        loss = loss_fn(yhat_seq.mean(dim=1), y)  # anchor
        loss.backward()
        opt.step()
        if (ep+1) % 5 == 0:
            print(f"epoch {ep+1}/{epochs} loss: {loss.item():.4f}")

    # write per-trend forecasts (naive pattern from the model)
    outdir = data_path("processed","forecasts")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for i, tid in enumerate(groups):
        g = df[df["trend_id"]==tid].sort_values("week")
        last_week = g["week"].max()
        weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
        with torch.no_grad():
            yh = model(X[i:i+1], out_steps=H).numpy().flatten()
        # simple CI
        fc = pd.DataFrame({
            "week": weeks,
            "yhat": yh,
            "yhat_lower": yh * 0.9,
            "yhat_upper": yh * 1.1
        })
        fc["model_used"] = "lstm"
        fc.to_csv(ensure_dir(Path(outdir).joinpath(f"{tid}.csv")), index=False)

if __name__ == "__main__":
    run()
