"""Shared helpers: data loading, styling, and the price model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

BASE = Path(__file__).parent
DATA_PATH = BASE / "used_car_sales.csv"
MODEL_PATH = BASE / "car_price_fitted.joblib"
TARGET = "Price-$"

NUM_FEATURES = ["Number of Seats", "Number of Doors", "Manufactured Year",
                "Mileage-KM", "Engine Power-HP"]
OHE_FEATURES = ["Gearbox"]
BIN_FEATURES = ["Location", "Manufacturer Name", "Car Type", "Color", "Energy"]
FEATURES = NUM_FEATURES + OHE_FEATURES + BIN_FEATURES

COLUMN_INFO = {
    "ID": ("Identifier", "Unique code of each car record."),
    "Distributor Name": ("Text", "Company/marketplace that distributed the car."),
    "Location": ("Category", "US state/city where the car is located."),
    "Car Name": ("Category", "Model name of the car (e.g. Creta, Thar)."),
    "Manufacturer Name": ("Category", "Brand that made the car."),
    "Car Type": ("Category", "Body style: SUV, Sedan, Hatchback, Convertible, Truck."),
    "Color": ("Category", "Exterior color of the car."),
    "Gearbox": ("Category", "Transmission type: Automatic or Manual."),
    "Number of Seats": ("Number", "How many seats the car has."),
    "Number of Doors": ("Number", "How many doors the car has."),
    "Energy": ("Category", "Fuel / power source: Petrol, Diesel, Hybrid, Electric."),
    "Manufactured Year": ("Number", "Year the car was built."),
    "Price-$": ("Number (target)", "Listed asking price of the car in US dollars."),
    "Mileage-KM": ("Number", "Total distance the car has driven, in kilometers."),
    "Engine Power-HP": ("Number", "Engine power in horsepower."),
    "Purchased Date": ("Date", "Date the dealer bought the car."),
    "Car Sale Status": ("Category", "Whether the car is Sold or Un Sold."),
    "Sold Date": ("Date", "Date the car was sold (1970-01-01 means not sold)."),
    "Purchased Price-$": ("Number", "Price the dealer paid for the car, in dollars."),
    "Sold Price-$": ("Number", "Final selling price in dollars (0 if not sold)."),
    "Margin-%": ("Number", "Profit/loss percentage on the sale (0 if not sold)."),
    "Sales Agent Name": ("Text", "Salesperson who handled the car."),
    "Sales Rating": ("Number", "Rating of the sales agent (1 to 5)."),
    "Sales Commission-$": ("Number", "Commission paid to the agent, in dollars."),
    "Feedback": ("Category", "Customer feedback: Poor, Average, Good, Excellent."),
}

PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#fb7185", "#a78bfa"]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def apply_style():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, #0b1220 55%); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }
section[data-testid="stSidebar"] { background: #0e1626; border-right: 1px solid #1e2a44; }

.hero { padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 60%, #8b5cf6 100%);
  box-shadow: 0 10px 40px rgba(99,102,241,.35); }
.hero h1 { color: #fff; font-size: 2.2rem; font-weight: 800; margin: 0 0 .3rem 0; }
.hero p { color: #e0e7ff; font-size: 1.05rem; margin: 0; }

.kpi { background: #111a2e; border: 1px solid #1e2a44; border-radius: 16px;
  padding: 1.1rem 1.3rem; transition: transform .15s, border-color .15s; }
.kpi:hover { transform: translateY(-3px); border-color: #38bdf8; }
.kpi .label { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
.kpi .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin-top: .2rem; }
.kpi .sub { color: #64748b; font-size: .8rem; }

.insight { background: linear-gradient(90deg, rgba(56,189,248,.12), rgba(129,140,248,.06));
  border-left: 4px solid #38bdf8; border-radius: 10px; padding: .8rem 1.1rem;
  color: #dbeafe; margin: .2rem 0 1.4rem 0; font-size: .98rem; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin: 1.6rem 0 .3rem 0; }
.section-sub { color: #94a3b8; margin-bottom: .6rem; font-size: .92rem; }

.result { text-align: center; padding: 2rem; border-radius: 20px;
  background: linear-gradient(135deg, #059669, #0ea5e9);
  box-shadow: 0 10px 40px rgba(14,165,233,.3); }
.result .big { font-size: 3rem; font-weight: 800; color: #fff; }
.result .small { color: #d1fae5; }

div.stButton > button, div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; border: 0;
  border-radius: 12px; font-weight: 600; padding: .6rem 1.2rem; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
  filter: brightness(1.1); color: white; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#cbd5e1"),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(font=dict(size=16, color="#f1f5f9")),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor="#1e2a44", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2a44", zeroline=False)
    return fig


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", RobustScaler(), NUM_FEATURES),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), OHE_FEATURES),
        ("bin", BinaryEncoder(), BIN_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", LinearRegression())])


@st.cache_resource(show_spinner="Preparing the price model...")
def get_model():
    """Return (fitted pipeline, metrics). The shipped car_price_model.joblib is
    not fitted, so we train the same kind of pipeline and cache it to disk."""
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    eval_pipe = build_pipeline().fit(X_tr, y_tr)
    pred = eval_pipe.predict(X_te)
    metrics = {"r2": r2_score(y_te, pred), "mae": mean_absolute_error(y_te, pred)}
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), metrics
        except Exception:
            pass
    final = build_pipeline().fit(X, y)
    joblib.dump(final, MODEL_PATH)
    return final, metrics

