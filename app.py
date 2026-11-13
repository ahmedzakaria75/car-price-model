import streamlit as st

from utils import apply_style

st.set_page_config(page_title="Used Car Studio", page_icon="🚗", layout="wide")
apply_style()

with st.sidebar:
    st.markdown("## 🚗 Used Car Studio")
    st.caption("Explore the data, understand prices, predict your car's value.")

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/analysis.py", title="Price Insights", icon="📊"),
    st.Page("views/predict.py", title="Price Prediction", icon="🔮"),
]
st.navigation(pages).run()

