
import pandas as pd
import numpy as np
from src.utils.config import CONFIG

def minmax_roll(df: pd.DataFrame, window=52):
    df = df.sort_values(["trend_id","week"]).copy()
    def _norm(g):
        g["roll_min"] = g["raw_signal"].rolling(window, min_periods=4).min()
        g["roll_max"] = g["raw_signal"].rolling(window, min_periods=4).max()
        denom = (g["roll_max"] - g["roll_min"]).replace(0, 1e-9)
        g["signal_norm"] = (g["raw_signal"] - g["roll_min"]) / denom
        return g
    return df.groupby("trend_id", group_keys=False).apply(_norm)

def combine_sources(df: pd.DataFrame):
    weights = {k: v["weight"] for k, v in CONFIG["sources"].items()}
    df = df.copy()
    df["signal_weight"] = df["source"].map(weights).fillna(1.0)
    df["signal"] = df["signal_weight"] * df["signal_norm"].fillna(0.0)
    combo = (df.groupby(["trend_id","week"], as_index=False)
               .agg(combined_signal=("signal","sum")))
    return combo

def add_features(combo: pd.DataFrame):
    combo = combo.sort_values(["trend_id","week"]).copy()
    def _feat(g):
        for lag in [1,2,4,8]:
            g[f"lag_{lag}"] = g["combined_signal"].shift(lag)
        g["roll_mean_4"] = g["combined_signal"].rolling(4).mean()
        g["roll_std_4"] = g["combined_signal"].rolling(4).std()
        g["roll_mean_12"] = g["combined_signal"].rolling(12).mean()
        g["roll_mean_52"] = g["combined_signal"].rolling(52).mean()
        g["roll_std_52"] = g["combined_signal"].rolling(52).std()
        g["novelty"] = (g["combined_signal"] - g["roll_mean_52"]) / (g["roll_std_52"] + 1e-9)
        isoweek = g["week"].dt.isocalendar().week
        g["is_ss"] = isoweek.between(10,35).astype(int)
        g["is_aw"] = 1 - g["is_ss"]
        return g
    return combo.groupby("trend_id", group_keys=False).apply(_feat)
