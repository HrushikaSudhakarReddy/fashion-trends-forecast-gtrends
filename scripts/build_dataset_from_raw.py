
import pandas as pd
from src.data.build_dataset import minmax_roll, combine_sources, add_features
from src.utils.io import data_path, ensure_dir

def main():
    parts = []
    p = data_path("raw","google_trends.csv")
    if p.exists():
        parts.append(pd.read_csv(p, parse_dates=["week"]))
    p2 = data_path("raw","synthetic_raw.csv")
    if p2.exists():
        parts.append(pd.read_csv(p2, parse_dates=["week"]))
    if not parts:
        raise SystemExit("No raw data files found. Run ingest_google_trends.py or use synthetic.")
    raw = pd.concat(parts, ignore_index=True)
    normed = minmax_roll(raw, window=52)
    combo = combine_sources(normed)
    feats = add_features(combo)
    out = data_path("processed","trend_features.csv")
    ensure_dir(out)
    feats.to_csv(out, index=False)
    print(f"Wrote features to {out}")

if __name__ == "__main__":
    main()
