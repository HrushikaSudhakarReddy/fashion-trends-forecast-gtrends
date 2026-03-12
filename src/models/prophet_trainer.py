
import pandas as pd
EXOG = ["is_ss","is_aw","novelty"]
def fit_prophet_or_dummy(df_trend: pd.DataFrame):
    try:
        from prophet import Prophet
    except Exception:
        return None
    df = df_trend.rename(columns={"week":"ds","combined_signal":"y"})[["ds","y"] + EXOG].dropna()
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True, seasonality_mode="additive")
    for x in EXOG:
        m.add_regressor(x)
    m.fit(df)
    return m
def prophet_forecast(m, df_future_exog: pd.DataFrame, periods=12):
    future = m.make_future_dataframe(periods=periods, freq="W-MON")
    future = future.merge(df_future_exog, on="ds", how="left").fillna(0)
    fcst = m.predict(future)
    return fcst[["ds","yhat","yhat_lower","yhat_upper"]]
