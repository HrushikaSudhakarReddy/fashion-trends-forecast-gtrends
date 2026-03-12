# Fashion Trend Forecasting using Google Trends

An end-to-end **data pipeline and forecasting dashboard** that identifies emerging fashion trends using **Google Trends search data**.

This project collects search interest for fashion keywords (colors, fabrics, silhouettes), processes the data into a structured dataset, and generates **time-series forecasts** visualized through an interactive **Streamlit dashboard**.

---

# 🚀 Live Interactive Dashboard

👉 **Click here to view the live dashboard**

https://fashion-trends-forecast-gtrends.streamlit.app

The dashboard allows users to:

* Explore trending fashion keywords
* View historical trend signals
* Analyze trend momentum
* See 12-week forecasts
* Download historical and forecast datasets

---

# Project Overview

This system demonstrates a **complete data science workflow**, including:

* Data ingestion from external APIs
* Data preprocessing and dataset construction
* Time-series forecasting
* Interactive data visualization
* Deployment of a production-style data app

The goal of the project is to simulate how **fashion retailers, designers, or analysts** could use search data to identify emerging trends.

---

# Key Features

* Google Trends API integration using **PyTrends**
* Weekly fashion keyword trend analysis
* Automated dataset construction pipeline
* Time-series forecasting using **Prophet**
* Interactive dashboard using **Streamlit**
* Forecast visualization with confidence intervals
* CSV export of historical and forecast data

The application also includes **fallback forecasting logic**, allowing the dashboard to run even if model forecasts are unavailable.

---

# Installation

Clone the repository:

git clone https://github.com/HrushikaSudhakarReddy/fashion-trends-forecast-gtrends.git
cd fashion-trends-forecast-gtrends

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

# Running the Dashboard Locally

The project ships with **precomputed demo data**, so the dashboard can run immediately.

streamlit run app/app.py

Open the dashboard in your browser:

http://localhost:8501

---

# Fetching Live Google Trends Data

To collect fresh data from Google Trends:

PYTHONPATH=. python scripts/ingest_google_trends.py --geo US --years 5

This downloads weekly search interest for fashion keywords defined in:

data/keywords/

Example keyword categories:

* Colors
* Fabrics
* Silhouettes

---

# Build the Dataset

After downloading raw data:

PYTHONPATH=. python scripts/build_dataset_from_raw.py

This step:

* Cleans raw Google Trends output
* Normalizes keyword time series
* Generates the modeling dataset used for forecasting.

---

# Train Forecast Models

To generate updated forecasts locally:

PYTHONPATH=. python src/models/train_prophet_all.py

This will:

* Train forecasting models for each fashion keyword
* Generate forecast outputs
* Save results for visualization.

---

# Running the Dashboard with Forecasts

Once forecasts are generated:

streamlit run app/app.py

The dashboard will display:

* Historical trend data
* Forecast projections
* Keyword comparisons

---

# Example Use Cases

* Fashion retail demand forecasting
* Trend scouting for designers and merchandisers
* Market research and consumer insight
* Seasonal product planning
* Data-driven fashion analytics

---

# Technology Stack

Python
Pandas
Prophet (Time Series Forecasting)
PyTrends (Google Trends API)
Streamlit (Interactive Data Apps)
Altair (Data Visualization)

---

# Data Notes

* Data frequency: Weekly
* Data source: Google Trends
* Storage format: CSV

The deployed dashboard uses **precomputed forecasts**, while full model training is available locally.

---

# Author

Hrushika Sudhakar Reddy
Data Science • Machine Learning • Forecasting Systems



