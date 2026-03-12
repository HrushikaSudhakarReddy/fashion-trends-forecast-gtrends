# src/data/google_trends.py
from typing import List, Tuple
import pandas as pd
from pytrends.request import TrendReq

def fetch_weekly_interest_pairs(pairs: List[Tuple[str, str]], geo="US", years=5, tz=360):
    """
    pairs: list of (trend_id, query_term)
           e.g. [("color_sage_green","sage green"),
                 ("fabric_linen","linen"),
                 ("silhouette_slip_dress","slip dress")]
    Returns DataFrame: trend_id, source, week, raw_signal
    """
    if not pairs:
        return pd.DataFrame(columns=["trend_id","source","week","raw_signal"])

    pytrends = TrendReq(hl='en-US', tz=tz)
    timeframe = f"today {years}-y"
    all_rows = []
    batch = 5  # pytrends behaves better with <=5 terms per batch

    for i in range(0, len(pairs), batch):
        batch_pairs = pairs[i:i+batch]
        queries = [q for _, q in batch_pairs]
        pytrends.build_payload(queries, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"date": "week"})
        for trend_id, q in batch_pairs:
            if q not in df.columns:
                continue
            tmp = df[["week", q]].copy()
            tmp["trend_id"] = trend_id
            tmp["source"] = "google_trends"
            tmp.rename(columns={q: "raw_signal"}, inplace=True)
            tmp = tmp[["trend_id","source","week","raw_signal"]]
            all_rows.append(tmp)

    if not all_rows:
        return pd.DataFrame(columns=["trend_id","source","week","raw_signal"])

    out = pd.concat(all_rows, ignore_index=True)
    out["week"] = pd.to_datetime(out["week"]).dt.to_period("W-MON").dt.start_time
    return out
