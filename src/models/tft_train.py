# src/models/tft_train.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.utils.io import data_path, ensure_dir
from src.utils.config import CONFIG

# --- at top of tft_train.py (near other constants) ---
ENC = 26  # encoder lookback (was 52; shorter is safer when data is limited)
H = CONFIG.get("horizon_weeks", 12)  # keep your 12-week horizon

MIN_TOTAL_LEN = ENC + H + 4  # require a bit of extra buffer per series


def load_features():
    df = pd.read_csv(data_path("processed", "trend_features.csv"), parse_dates=["week"])
    df = df.sort_values(["trend_id", "week"]).reset_index(drop=True)
    # make time_idx PER series (contiguous integers starting at 0)
    df["time_idx"] = df.groupby("trend_id").cumcount()
    # ensure numeric
    for c in ["combined_signal", "is_ss", "is_aw", "novelty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # drop series that are too short for ENC+H
    lengths = df.groupby("trend_id")["time_idx"].max() + 1  # +1 because starts at 0
    keep_ids = lengths[lengths >= MIN_TOTAL_LEN].index.tolist()
    kept = df[df["trend_id"].isin(keep_ids)].copy()
    if kept.empty:
        raise SystemExit(
            f"No series long enough for ENC={ENC} and H={H}. "
            "Add more weeks or reduce ENC/H."
        )
    return kept


def build_datasets(df: pd.DataFrame):
    # global cutoff: keep last H*1.5 steps for validation/prediction
    max_idx = int(df["time_idx"].max())
    train_cutoff = max_idx - max(H * 2, H + 4)  # leave room; more conservative

    training = df[df.time_idx <= train_cutoff].copy()
    if training.empty:
        # if too aggressive, relax cutoff
        train_cutoff = max_idx - (H + 2)
        training = df[df.time_idx <= train_cutoff].copy()

    # Common dataset spec
    common_kwargs = dict(
        time_idx="time_idx",
        target="combined_signal",
        group_ids=["trend_id"],
        static_categoricals=["trend_id"],
        time_varying_known_reals=["is_ss", "is_aw", "novelty"],
        time_varying_unknown_reals=["combined_signal"],
        max_encoder_length=ENC,
        max_prediction_length=H,
        allow_missing_timesteps=True,
    )

    # Build training dataset
    training_ds = TimeSeriesDataSet(training, **common_kwargs)

    # Validation / prediction dataset: from full df, derived from training_ds
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )

    train_loader = training_ds.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    return training_ds, train_loader, val_loader


def train_tft(train_loader, val_loader, training_ds):
    seed_everything(42)
    # small-ish model to run on CPU
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),  # gives yhat_lower/upper via quantiles
        log_interval=50,
    )
    trainer = Trainer(
        max_epochs=10,   # bump to 30–50 for better accuracy
        accelerator="cpu",
        enable_checkpointing=False,
        logger = False,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return tft

def predict_write_csvs(model, full_df: pd.DataFrame, training_ds: TimeSeriesDataSet):
    outdir = data_path("processed", "forecasts")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # build a prediction dataset that asks for the next H weeks per series
    # take last encoder window of each series and let model predict H ahead
    last_idx = full_df.groupby("trend_id")["time_idx"].max().rename("last_idx")
    df_merged = full_df.merge(last_idx, on="trend_id", how="left")
    # only need the last encoder window per series for prediction
    enc_len = training_ds.max_encoder_length
    pred_df = df_merged[df_merged["time_idx"] >= df_merged["last_idx"] - (enc_len - 1)].copy()

    # add placeholder future rows so TFT can roll out H steps
    future_rows = []
    for tid, g in df_merged.groupby("trend_id"):
        max_t = int(g["time_idx"].max())
        for step in range(1, H + 1):
            future_rows.append({
                "trend_id": tid,
                "time_idx": max_t + step,
                "week": g["week"].max() + pd.Timedelta(weeks=step),
                # known future features – here we can re-compute season flags
                "is_ss": int(((g["week"].max() + pd.Timedelta(weeks=step)).isocalendar().week in range(10, 36))),
                "is_aw": 1 - int(((g["week"].max() + pd.Timedelta(weeks=step)).isocalendar().week in range(10, 36))),
                "novelty": 0.0,
                "combined_signal": np.nan,  # unknown future target
            })
    fut = pd.DataFrame(future_rows)
    fut["week"] = pd.to_datetime(fut["week"])

    full_for_pred = pd.concat([full_df, fut], ignore_index=True).sort_values(["trend_id", "time_idx"])
    for c in ["combined_signal", "is_ss", "is_aw", "novelty"]:
        full_for_pred[c] = pd.to_numeric(full_for_pred[c], errors="coerce")

    pred_ds = TimeSeriesDataSet.from_dataset(training_ds, full_for_pred, predict=True, stop_randomization=True)
    pred_loader = pred_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    preds, idx = model.predict(pred_loader, return_index=True, mode="prediction")
    # preds has shape [N, H, quantiles]; index gives mapping to series
    # Convert to per-trend CSVs with yhat, yhat_lower/upper
    # Quantiles: default QuantileLoss uses [0.1, 0.5, 0.9]
    med = preds[:, :, 1]
    lo = preds[:, :, 0]
    hi = preds[:, :, 2]

    # Build mapping from index rows to (trend_id, start_time_idx)
    index_df = pd.DataFrame(idx)
    # For each row in index_df, write H rows for that trend
    for row in index_df.itertuples():
        tid = row.trend_id
        # find last known week for this trend
        g = full_df[full_df["trend_id"] == tid].sort_values("week")
        last_week = g["week"].max()
        weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
        fc = pd.DataFrame({
            "week": weeks,
            "yhat": med[row.Index],
            "yhat_lower": lo[row.Index],
            "yhat_upper": hi[row.Index],
        })
        fc["model_used"] = "tft"
        fc.to_csv(ensure_dir(Path(outdir).joinpath(f"{tid}__tft.csv")), index=False)


def run():
    df = load_features()
    print(
        f"[TFT] series kept: {df['trend_id'].nunique()} | "
        f"rows: {len(df)} | ENC={ENC} H={H}"
    )
    training_ds, train_loader, val_loader = build_datasets(df)
    model = train_tft(train_loader, val_loader, training_ds)
    predict_write_csvs(model, df, training_ds)


if __name__ == "__main__":
    run()
