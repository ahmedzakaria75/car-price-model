import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import FEATURES, TARGET, hero, kpi, get_model, load_data, section, style_fig

df = load_data()
model, metrics = get_model()

hero("Car Price Prediction", "Describe the car and get an estimated price in seconds.")

with st.form("predict"):
    section("Car details")
    c1, c2, c3 = st.columns(3)
    with c1:
        manufacturer = st.selectbox("Brand", sorted(df["Manufacturer Name"].unique()))
        car_type = st.selectbox("Car type", sorted(df["Car Type"].unique()))
        energy = st.selectbox("Fuel type", sorted(df["Energy"].unique()))
        gearbox = st.selectbox("Gearbox", sorted(df["Gearbox"].unique()))
    with c2:
        year = st.slider("Year built", int(df["Manufactured Year"].min()), int(df["Manufactured Year"].max()), 2020)
        mileage = st.number_input("Kilometers driven", 0, 200000, 40000, step=1000)
        power = st.slider("Engine power (HP)", int(df["Engine Power-HP"].min()), int(df["Engine Power-HP"].max()), 140)
    with c3:
        seats = st.selectbox("Seats", sorted(df["Number of Seats"].unique()), index=2)
        doors = st.selectbox("Doors", sorted(df["Number of Doors"].unique()), index=2)
        color = st.selectbox("Color", sorted(df["Color"].unique()))
        location = st.selectbox("Location", sorted(df["Location"].unique()))
    go_btn = st.form_submit_button("✨ Predict price", width="stretch")

if go_btn:
    row = pd.DataFrame([{
        "Number of Seats": seats, "Number of Doors": doors, "Manufactured Year": year,
        "Mileage-KM": mileage, "Engine Power-HP": power, "Gearbox": gearbox,
        "Location": location, "Manufacturer Name": manufacturer, "Car Type": car_type,
        "Color": color, "Energy": energy,
    }])[FEATURES]
    price = float(model.predict(row)[0])
    mae = metrics["mae"]
    avg = df[TARGET].mean()

    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f'<div class="result"><div class="small">Estimated price</div>'
            f'<div class="big">${price:,.0f}</div>'
            f'<div class="small">Likely range: ${price - mae:,.0f} – ${price + mae:,.0f}</div></div>',
            unsafe_allow_html=True)
    with right:
        diff = price - avg
        kpi("Compared with the average car", f"{'+' if diff >= 0 else '-'}${abs(diff):,.0f}",
            f"Average price is ${avg:,.0f}")
        st.write("")
        similar = df[(df["Manufacturer Name"] == manufacturer) & (df["Car Type"] == car_type)][TARGET]
        if len(similar):
            kpi("Similar cars in the data", f"${similar.mean():,.0f}",
                f"average of {len(similar)} {manufacturer} {car_type}s")

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=price, number={"prefix": "$", "valueformat": ",.0f"},
        gauge={"axis": {"range": [df[TARGET].min(), df[TARGET].max()]},
               "bar": {"color": "#38bdf8"}, "bgcolor": "#111a2e",
               "threshold": {"line": {"color": "#fbbf24", "width": 3}, "value": avg}}))
    fig.update_layout(title="Where this car sits in the market (yellow line = average)")
    st.plotly_chart(style_fig(fig, 300), width="stretch")
else:
    st.info("Fill in the details above and press **Predict price**.")

with st.expander("ℹ️ About this estimate"):
    st.write(f"The model was trained on {len(df):,} cars and explains about "
             f"**{metrics['r2'] * 100:.0f}%** of price variation on unseen data. "
             f"Typical error is about **${metrics['mae']:,.0f}**. Treat the result as a guide, not a quote.")

