# scripts/ingest_google_trends.py
import argparse, sys
from src.utils.io import data_path, ensure_dir
from src.data.google_trends import fetch_weekly_interest_pairs

def read_keywords_by_dim():
    base = data_path("keywords")
    files = {
        "color": base / "colors.txt",
        "fabric": base / "fabrics.txt",
        "silhouette": base / "silhouettes.txt",
    }
    out = {}
    for dim, p in files.items():
        if p.exists():
            with open(p) as f:
                out[dim] = [line.strip() for line in f if line.strip()]
        else:
            out[dim] = []
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="US")
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()

    by_dim = read_keywords_by_dim()

    # Build (trend_id, query) pairs WITH prefixes
    pairs = []
    for dim, terms in by_dim.items():
        for t in terms:
            tid = f"{dim}_{t.replace(' ', '_')}"   # <-- critical
            pairs.append((tid, t))

    if not pairs:
        print("No keywords found. Add terms to data/keywords/*.txt")
        sys.exit(1)

    df = fetch_weekly_interest_pairs(pairs, geo=args.geo, years=args.years)
    out = data_path("raw", "google_trends.csv")
    ensure_dir(out)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
