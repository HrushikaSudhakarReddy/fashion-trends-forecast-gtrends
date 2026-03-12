
import numpy as np
def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-9)))
def mase(y_true, y_pred, m=52):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) <= m: return float(np.nan)
    naive = np.abs(y_true[m:] - y_true[:-m]).mean()
    return float(np.abs(y_pred - y_true).mean() / (naive + 1e-9))
