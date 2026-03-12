
from src.metrics.timeseries_metrics import smape, mase
import numpy as np
def test_smape_zero():
    y = np.array([1,2,3])
    assert abs(smape(y, y)) < 1e-9
def test_mase_simple():
    y_true = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
    y_pred = y_true.copy()
    m = mase(y_true, y_pred, m=4)
    assert m <= 1.0
