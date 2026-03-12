import sys
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import altair as alt
from src.utils.io import data_path

st.set_page_config(
    page_title="Fashion Trend Forecasts",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Banner image loader
# ---------------------------
def get_base64_image(image_path: Path):
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


banner_file = Path(__file__).parent / "assets" / "banner.jpg"
banner_base64 = get_base64_image(banner_file)

banner_css = ""
if banner_base64:
    banner_css = f"""
    background-image: url("data:image/jpg;base64,{banner_base64}");
    """
else:
    banner_css = "background: linear-gradient(135deg, #d8cfc3 0%, #b9aa99 100%);"

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #f7f4ef;
        color: #2f2a26;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #efe9e1;
        border-right: 1px solid rgba(47, 42, 38, 0.08);
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    h1, h2, h3 {{
        color: #2f2a26;
        letter-spacing: -0.02em;
    }}

    .hero {{
        position: relative;
        height: 360px;
        {banner_css}
        background-size: cover;
        background-position: center;
        border-radius: 28px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-end;
        overflow: hidden;
        border: 1px solid rgba(47, 42, 38, 0.08);
    }}

    .hero-overlay {{
        background: linear-gradient(
            to top,
            rgba(22, 18, 16, 0.72) 0%,
            rgba(22, 18, 16, 0.48) 38%,
            rgba(22, 18, 16, 0.18) 100%
        );
        padding: 2.2rem 2rem 1.8rem 2rem;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: flex-end;
    }}

    .hero-title {{
        font-size: 3.3rem;
        line-height: 1.02;
        font-weight: 700;
        color: white;
        letter-spacing: -0.03em;
        max-width: 700px;
    }}

    .section-card {{
        background: #fbf8f3;
        border: 1px solid rgba(47, 42, 38, 0.08);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        margin-bottom: 1rem;
    }}

    .metric-card {{
        background: #fcfaf7;
        border: 1px solid rgba(47, 42, 38, 0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        min-height: 132px;
    }}

    .metric-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b8177;
        margin-bottom: 0.45rem;
        font-weight: 600;
    }}

    .metric-value {{
        font-size: 1.55rem;
        font-weight: 700;
        color: #2f2a26;
        margin-bottom: 0.35rem;
    }}

    .metric-note {{
        font-size: 0.92rem;
        color: #6f665d;
        line-height: 1.45;
    }}

    .insight-box {{
        background: #f8f3ec;
        border: 1px solid rgba(47, 42, 38, 0.08);
        border-radius: 22px;
        padding: 1.15rem;
        margin-top: 0.75rem;
    }}

    .small-note {{
        color: #7a7066;
        font-size: 0.92rem;
    }}

    .trend-pill {{
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #ebe3d9;
        color: #4a423b;
        font-size: 0.82rem;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
    }}

    div[data-testid="stDownloadButton"] > button {{
        border-radius: 999px;
        border: 1px solid rgba(47, 42, 38, 0.12);
        background: #f8f4ee;
        color: #2f2a26;
    }}

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {{
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Hero banner
# ---------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-overlay">
            <div class="hero-title">Fashion Trends Forecasting</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def prettify(tid: str) -> str:
    parts = tid.split("_", 1)
    return parts[1].replace("_", " ").strip() if len(parts) > 1 else tid.replace("_", " ").strip()


def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def classify_direction(delta):
    if delta >= 0.08:
        return "Rising"
    if delta <= -0.08:
        return "Declining"
    return "Steady"


def confidence_label(width_mean):
    if width_mean <= 0.10:
        return "High"
    if width_mean <= 0.22:
        return "Moderate"
    return "Low"


def seasonality_hint(hist_df):
    if len(hist_df) < 26:
        return "Not enough history to confidently assess seasonality."
    month_strength = hist_df.groupby(hist_df["week"].dt.month)["combined_signal"].mean()
    spread = safe_float(month_strength.max() - month_strength.min())
    if spread >= 0.25:
        return "The pattern looks seasonal, with some parts of the year consistently stronger than others."
    if spread >= 0.12:
        return "There may be a mild seasonal effect, but the pattern is not strongly consistent."
    return "The trend appears fairly evenly distributed across the year, with limited seasonality."


def momentum_note(recent_avg, prior_avg):
    delta = recent_avg - prior_avg
    if delta >= 0.10:
        return "Recent search interest is materially stronger than the prior month."
    if delta <= -0.10:
        return "Recent search interest has softened compared with the prior month."
    return "Recent search interest is relatively stable versus the prior month."


def explain_trend(selected_name, direction, recent_avg, forecast_end, conf_label, seasonal_text):
    recent_level = "high" if recent_avg >= 0.70 else "moderate" if recent_avg >= 0.40 else "low"
    future_word = (
        "continue strengthening"
        if forecast_end == "Rising"
        else "soften"
        if forecast_end == "Declining"
        else "hold relatively steady"
    )

    return {
        "summary": (
            f"{selected_name.title()} currently shows {recent_level} search interest. "
            f"The near-term pattern is classified as {direction.lower()}, and the 12-week outlook suggests it may {future_word.lower()}."
        ),
        "meaning": (
            f"This signal is best read as a measure of consumer attention rather than direct sales demand. "
            f"A {direction.lower()} pattern means search behavior is moving meaningfully, while a {conf_label.lower()} confidence forecast "
            f"means the projected range is "
            f"{'narrower and more stable' if conf_label == 'High' else 'usable but somewhat uncertain' if conf_label == 'Moderate' else 'wide and more uncertain'}."
        ),
        "business": (
            f"For a brand or retailer, {selected_name.lower()} could be used as an input for assortment planning, campaign timing, "
            f"or limited product testing. Rather than overcommitting inventory, this kind of signal is often strongest when paired with sell-through, conversion, or social engagement data."
        ),
        "seasonality": seasonal_text,
    }


def build_metric_card(label, value, note):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


# ---------------------------
# Load data
# ---------------------------
features_path = data_path("processed", "trend_features.csv")
df = pd.read_csv(features_path, parse_dates=["week"])
df["dimension"] = df["trend_id"].astype(str).str.split("_", n=1).str[0]

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.markdown("## Explore trends")

dim = st.sidebar.selectbox("Dimension", ["color", "fabric", "silhouette"])

subset = df[df["dimension"] == dim].copy().sort_values("trend_id")
subset_names = {prettify(tid): tid for tid in subset["trend_id"].unique()}

query = st.sidebar.text_input("Search trend", "").strip().lower()
if query:
    subset_names = {name: tid for name, tid in subset_names.items() if query in name.lower()}

if not subset_names:
    st.warning("No trends found for this dimension or search.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Forecast engine",
    ["Auto", "TFT", "Prophet", "LSTM", "Naive"],
    help="Auto prefers TFT → Prophet → LSTM → Naive based on available forecast files.",
)

latest_week = pd.to_datetime(subset["week"]).max()
recent = subset[pd.to_datetime(subset["week"]) >= latest_week - pd.Timedelta(weeks=4)]

top_rising_ids = (
    recent.groupby("trend_id")["novelty"].mean().sort_values(ascending=False).head(8).index.tolist()
)

st.sidebar.markdown("### Top rising now")
for tid in top_rising_ids:
    st.sidebar.markdown(f"- {prettify(tid).title()}")

# ---------------------------
# Selected trend
# ---------------------------
selected_name = st.selectbox("Pick a trend", list(subset_names.keys()))
selected = subset_names[selected_name]

hist = subset[subset["trend_id"] == selected].sort_values("week").copy()

if hist.empty:
    st.warning("No data available for the selected trend.")
    st.stop()

# ---------------------------
# Forecast fallback
# ---------------------------
last_val = safe_float(hist["combined_signal"].iloc[-1])
last4 = hist.tail(4)["combined_signal"].dropna().values
slope = (last4[-1] - last4[0]) / max(1, (len(last4) - 1)) if len(last4) >= 2 else 0.0

start = pd.to_datetime(hist["week"]).max() + pd.Timedelta(weeks=1)
H = 12
weeks = pd.date_range(start, periods=H, freq="W-MON")
fc = pd.DataFrame({"week": weeks, "yhat": [last_val + slope * (i + 1) for i in range(H)]})
fc["yhat"] = fc["yhat"].clip(lower=0, upper=1)
fc["yhat_lower"] = (fc["yhat"] * 0.9).clip(lower=0, upper=1)
fc["yhat_upper"] = (fc["yhat"] * 1.1).clip(lower=0, upper=1)
fc["model_used"] = "naive"


def try_load(suffix):
    path = data_path("processed", "forecasts", f"{selected}__{suffix}.csv")
    p = Path(path)
    if not p.exists():
        return None
    loaded = pd.read_csv(p, parse_dates=["week"])
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col not in loaded.columns:
            if col == "yhat_lower":
                loaded[col] = loaded["yhat"] * 0.9
            elif col == "yhat_upper":
                loaded[col] = loaded["yhat"] * 1.1
    loaded["yhat"] = pd.to_numeric(loaded["yhat"], errors="coerce").clip(lower=0, upper=1)
    loaded["yhat_lower"] = pd.to_numeric(loaded["yhat_lower"], errors="coerce").clip(lower=0, upper=1)
    loaded["yhat_upper"] = pd.to_numeric(loaded["yhat_upper"], errors="coerce").clip(lower=0, upper=1)
    loaded["model_used"] = suffix
    return loaded[["week", "yhat", "yhat_lower", "yhat_upper", "model_used"]]


if model_choice == "Auto":
    order = ["tft", "prophet", "lstm", "naive"]
elif model_choice == "TFT":
    order = ["tft", "naive"]
elif model_choice == "Prophet":
    order = ["prophet", "naive"]
elif model_choice == "LSTM":
    order = ["lstm", "naive"]
else:
    order = ["naive"]

fc_used = None
for name in order:
    if name == "naive":
        fc_used = fc
        break
    loaded = try_load(name)
    if loaded is not None:
        fc_used = loaded
        break

fc = fc_used.copy()

# ---------------------------
# Analytics / explanations
# ---------------------------
recent_avg = safe_float(hist.tail(4)["combined_signal"].mean())
prior_avg = safe_float(hist.iloc[-8:-4]["combined_signal"].mean()) if len(hist) >= 8 else recent_avg
direction = classify_direction(recent_avg - prior_avg)

forecast_start_val = safe_float(fc["yhat"].iloc[0], recent_avg)
forecast_end_val = safe_float(fc["yhat"].iloc[-1], recent_avg)
forecast_direction = classify_direction(forecast_end_val - forecast_start_val)

ci_width = (fc["yhat_upper"] - fc["yhat_lower"]).dropna()
avg_ci_width = safe_float(ci_width.mean(), 0.2)
conf_label = confidence_label(avg_ci_width)

seasonal_text = seasonality_hint(hist)
insights = explain_trend(selected_name, direction, recent_avg, forecast_direction, conf_label, seasonal_text)

model_label = str(fc["model_used"].iloc[0]).upper()
latest_hist_date = pd.to_datetime(hist["week"]).max().strftime("%b %d, %Y")

# ---------------------------
# Title row
# ---------------------------
st.markdown(f"## {selected_name.title()}")
st.markdown(
    f"""
    <span class="trend-pill">{dim.title()}</span>
    <span class="trend-pill">Latest observed week: {latest_hist_date}</span>
    <span class="trend-pill">Forecast engine: {model_label}</span>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Metric cards
# ---------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        build_metric_card(
            "Current momentum",
            direction,
            momentum_note(recent_avg, prior_avg),
        ),
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        build_metric_card(
            "12-week outlook",
            forecast_direction,
            f"The model suggests the trend may move from {forecast_start_val:.2f} to {forecast_end_val:.2f} over the forecast window.",
        ),
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        build_metric_card(
            "Trend strength",
            f"{recent_avg:.2f}",
            "This is the recent average of the normalized search-interest signal, scaled roughly between 0 and 1.",
        ),
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        build_metric_card(
            "Forecast confidence",
            conf_label,
            f"Average forecast-band width is {avg_ci_width:.2f}. Wider bands indicate greater uncertainty.",
        ),
        unsafe_allow_html=True,
    )

# ---------------------------
# History + insights
# ---------------------------
left, right = st.columns([1.8, 1.1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Historical trend")
    st.markdown(
        '<div class="small-note">This chart shows past search interest for the selected trend. Spikes represent periods when the keyword drew more attention relative to its own baseline.</div>',
        unsafe_allow_html=True,
    )

    hist_chart = (
        alt.Chart(hist)
        .mark_line(point=False)
        .encode(
            x=alt.X("week:T", title="Week"),
            y=alt.Y("combined_signal:Q", title="Trend strength", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("week:T", title="Week"),
                alt.Tooltip("combined_signal:Q", title="Trend strength", format=".2f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(hist_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### What this means")
    st.markdown(
        f"""
        <div class="insight-box">
            <b>Summary</b><br>
            {insights["summary"]}
            <br><br>
            <b>Interpretation</b><br>
            {insights["meaning"]}
            <br><br>
            <b>Business view</b><br>
            {insights["business"]}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Forecast section
# ---------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### Forecast")
st.markdown(
    '<div class="small-note">The forecast extends the series by 12 weeks. The central line shows the expected path, and the shaded band shows uncertainty. Narrower bands indicate more stable estimates.</div>',
    unsafe_allow_html=True,
)

hist_plot = (
    hist[["week", "combined_signal"]]
    .rename(columns={"combined_signal": "yhat"})
    .assign(type="History")
)
hist_plot["yhat_lower"] = pd.NA
hist_plot["yhat_upper"] = pd.NA
hist_plot = hist_plot[["week", "yhat", "yhat_lower", "yhat_upper", "type"]]

fc_plot = fc[["week", "yhat", "yhat_lower", "yhat_upper"]].copy()
fc_plot["type"] = "Forecast"

for col in ["yhat", "yhat_lower", "yhat_upper"]:
    hist_plot[col] = pd.to_numeric(hist_plot[col], errors="coerce")
    fc_plot[col] = pd.to_numeric(fc_plot[col], errors="coerce")

plot_df = pd.concat([hist_plot, fc_plot], ignore_index=True)

line = (
    alt.Chart(plot_df)
    .mark_line(strokeWidth=2.5)
    .encode(
        x=alt.X("week:T", title="Week"),
        y=alt.Y("yhat:Q", title="Trend strength / forecast", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["History", "Forecast"], range=["#8fb9e8", "#2f2a26"]),
            legend=alt.Legend(title=None, orient="right"),
        ),
        tooltip=[
            alt.Tooltip("week:T", title="Week"),
            alt.Tooltip("type:N", title="Series"),
            alt.Tooltip("yhat:Q", title="Value", format=".2f"),
        ],
    )
    .properties(height=380)
)

band = (
    alt.Chart(plot_df.dropna(subset=["yhat_lower", "yhat_upper"]))
    .mark_area(opacity=0.18, color="#bda992")
    .encode(
        x="week:T",
        y="yhat_lower:Q",
        y2="yhat_upper:Q",
        tooltip=[
            alt.Tooltip("week:T", title="Week"),
            alt.Tooltip("yhat_lower:Q", title="Lower", format=".2f"),
            alt.Tooltip("yhat_upper:Q", title="Upper", format=".2f"),
        ],
    )
)

st.altair_chart(band + line, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Explanation section
# ---------------------------
e1, e2 = st.columns(2)

with e1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### How to read this analysis")
    st.write(
        """
- **Trend strength** is a normalized signal of search attention, not raw sales.
- **Current momentum** compares the recent month to the month before it.
- **12-week outlook** summarizes whether the forecast is pointing upward, downward, or sideways.
- **Forecast confidence** reflects how wide the uncertainty band is. Wider bands mean less certainty.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with e2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Seasonal pattern")
    st.write(insights["seasonality"])
    st.write(
        "A trend can be interesting for two very different reasons: because it is truly emerging, or because it predictably returns every season. This distinction matters when using the signal for planning."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Downloads
# ---------------------------
st.markdown("### Export data")
d1, d2 = st.columns(2)

pretty = prettify(selected).replace(" ", "_")

with d1:
    st.download_button(
        "Download history (CSV)",
        hist.to_csv(index=False).encode(),
        file_name=f"{pretty}_history.csv",
    )

with d2:
    st.download_button(
        "Download forecast (CSV)",
        fc.to_csv(index=False).encode(),
        file_name=f"{pretty}_forecast.csv",
    )