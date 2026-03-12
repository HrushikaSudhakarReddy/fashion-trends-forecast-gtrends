# Fashion Trend Forecasting using Google Trends

A data pipeline and forecasting dashboard that predicts emerging fashion trends using **Google Trends search data**.

This project collects search interest for fashion keywords (colors, fabrics, silhouettes), processes the data into a structured dataset, and generates **time-series forecasts using Prophet**. Results are visualized through an interactive **Streamlit dashboard**.

The system demonstrates an **end-to-end data science workflow** including:

- Data ingestion from external APIs
- Data preprocessing and dataset construction
- Time-series forecasting
- Interactive visualization
---
# Project Structure

fashion-trends-forecast-gtrends

app/  
│ └── app.py (Streamlit dashboard)

data/  
│  
├── keywords/  
│ ├── colors.txt  
│ ├── fabrics.txt  
│ └── silhouettes.txt  

├── raw/ (raw Google Trends downloads)  
└── processed/ (clean modeling dataset)

scripts/  
├── ingest_google_trends.py  
└── build_dataset_from_raw.py  

src/  
└── models/  
    └── train_prophet_all.py  

requirements.txt  
README.md  

---

# Features

- Google Trends API integration using **pytrends**
- Weekly fashion keyword trend analysis
- Automated dataset construction pipeline
- Time-series forecasting using **Prophet**
- Interactive visualization using **Streamlit**
- Works with **synthetic demo data out of the box**

---

# Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/fashion-trends-forecast-gtrends.git  
cd fashion-trends-forecast-gtrends  

Create a virtual environment:

python3 -m venv .venv  
source .venv/bin/activate  

Install dependencies:

pip install --upgrade pip  
pip install -r requirements.txt  

---

# Running the Dashboard

The project includes **precomputed demo data**, so the dashboard can run immediately.

streamlit run app/app.py  

Open the dashboard in your browser:

http://localhost:8501

---

# Fetching Live Google Trends Data

To download fresh data from Google Trends:

PYTHONPATH=. python scripts/ingest_google_trends.py --geo US --years 5

This script fetches weekly search interest for fashion keywords defined in:

data/keywords/

Example keyword categories:

- Colors
- Fabrics
- Silhouettes

---

# Build the Dataset

After downloading raw data:

PYTHONPATH=. python scripts/build_dataset_from_raw.py

This step:

- Cleans raw Google Trends output
- Normalizes keyword time series
- Produces the modeling dataset used for forecasting.

---

# Train Forecast Models

Generate updated forecasts using Prophet:

PYTHONPATH=. python src/models/train_prophet_all.py

This will:

- Train forecasting models for each fashion keyword
- Generate forecast outputs
- Save results for visualization.

---

# Running the Dashboard with Forecasts

After generating forecasts:

streamlit run app/app.py

The dashboard will display:

- Historical trend data
- Forecast projections
- Keyword comparisons

---

# Example Use Cases

- Fashion retail demand prediction
- Trend scouting for designers and merchandisers
- Market research and seasonal planning
- Data-driven fashion analytics

---

# Technology Stack

- Python
- Pandas
- Prophet (Time Series Forecasting)
- PyTrends (Google Trends API)
- Streamlit (Data Apps)

---

# Data Notes

- Data frequency: **weekly**
- Data source: **Google Trends**
- Storage format: **CSV**

Prophet and PyTorch are optional dependencies. If unavailable, the application falls back to **precomputed forecasts or simple heuristics**.

---

# Future Improvements

Potential extensions include:

- Forecast evaluation metrics (MAPE, RMSE)
- Backtesting framework
- Automated keyword discovery
- Trend anomaly detection
- Integration with retail sales data

---

# Author

Hrushika Sudhakar Reddy  
MS Computer Science – University of Dayton  

Data Science | Machine Learning | Forecasting Systems


