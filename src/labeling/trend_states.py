
import numpy as np
import pandas as pd
def slope(values):
    x = np.arange(len(values))
    if len(values) < 2: return 0.0
    return float(np.polyfit(x, values, 1)[0])
def label_trend(history: pd.DataFrame, forecast: pd.DataFrame, tau1=0.02, tau2=0.05):
    last4 = history.tail(4)["combined_signal"].values
    s4 = slope(last4) if len(last4) >= 2 else 0.0
    y0 = float(forecast["yhat"].iloc[0])
    y4 = float(forecast["yhat"].iloc[min(3, len(forecast)-1)])
    ci_width = float((forecast.get("yhat_upper", forecast["yhat"]).iloc[0] - forecast.get("yhat_lower", forecast["yhat"]).iloc[0]))
    rel_uncert = ci_width / (abs(y0) + 1e-9)
    if rel_uncert > 0.5: return "Uncertain"
    if s4 > tau1 and (y4 - y0) > tau2: return "Rising"
    if s4 > 0 and abs(y4 - y0) <= tau2: return "Peaking"
    if s4 < -tau1 and (y4 - y0) < -tau2: return "Declining"
    return "Uncertain"
