import pandas as pd
from pathlib import Path
from src.models.prophet_trainer import fit_prophet_or_dummy, prophet_forecast
from src.utils.io import data_path, ensure_dir

H = 12

def run():
    feats = pd.read_csv(data_path("processed","trend_features.csv"), parse_dates=["week"])
    outdir = data_path("processed","forecasts")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for trend, g in feats.groupby("trend_id"):
        g = g.sort_values("week")
        model = fit_prophet_or_dummy(g)

        if model is None:
            # --- naive fallback ---
            hist = g[["week","combined_signal"]].dropna()
            if len(hist) < 4:
                continue
            last = hist.iloc[-1]["combined_signal"]
            slope = (hist.iloc[-1]["combined_signal"] - hist.iloc[-4]["combined_signal"]) / 3.0
            future_weeks = pd.date_range(hist["week"].iloc[-1] + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
            yhat = [last + slope*(i+1) for i in range(H)]
            yhat_lower = [max(0.0, v*0.9) for v in yhat]
            yhat_upper = [v*1.1 for v in yhat]
            fc = pd.DataFrame({
                "week": future_weeks,
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper
            })
            fc["model_used"] = "naive"
            out_file = Path(outdir).joinpath(f"{trend}__naive.csv")
        else:
            # --- prophet forecast ---
            last_week = g["week"].max()
            future = pd.DataFrame({"ds": pd.date_range(last_week + pd.Timedelta(weeks=1), periods=H, freq="W-MON")})
            isoweek = future["ds"].dt.isocalendar().week
            future["is_ss"] = isoweek.between(10,35).astype(int)
            future["is_aw"] = 1 - future["is_ss"]
            future["novelty"] = 0.0
            fc = prophet_forecast(model, future, periods=H).rename(columns={"ds":"week"})
            fc["model_used"] = "prophet"
            out_file = Path(outdir).joinpath(f"{trend}__prophet.csv")

        fc.to_csv(ensure_dir(out_file), index=False)

if __name__ == "__main__":
    run()
