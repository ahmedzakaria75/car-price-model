import pandas as pd
import plotly.express as px
import streamlit as st

from utils import PALETTE, TARGET, hero, insight, kpi, load_data, section, style_fig

df = load_data()
avg = df[TARGET].mean()

hero("What Drives a Car's Price?", "Simple charts that show how each feature relates to the listed price in dollars.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Average price", f"${avg:,.0f}")
with c2:
    kpi("Typical (median)", f"${df[TARGET].median():,.0f}")
with c3:
    kpi("Cheapest", f"${df[TARGET].min():,}")
with c4:
    kpi("Most expensive", f"${df[TARGET].max():,}")


def bar_avg(col, title, horizontal=True, top=None):
    g = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
    if top:
        g = g.head(top)
    g = g.sort_values(ascending=True).reset_index()  # highest bar ends up on top
    fig = px.bar(g, x=TARGET, y=col, orientation="h", text=g[TARGET].map("${:,.0f}".format),
                 title=title, color=TARGET, color_continuous_scale=["#334155", "#38bdf8", "#818cf8"])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Average price ($)", range=[0, g[TARGET].max() * 1.18])
    fig.update_yaxes(title="")
    return style_fig(fig, 60 + 38 * len(g)), g


def best_worst(g, col):
    hi, lo = g.loc[g[TARGET].idxmax()], g.loc[g[TARGET].idxmin()]
    return hi[col], hi[TARGET], lo[col], lo[TARGET]


# ---- Price distribution
section("1 · How are prices spread?", "Most cars fall in the tall bars.")
fig = px.histogram(df, x=TARGET, nbins=30, title="Number of cars by price")
fig.update_traces(marker_color="#38bdf8", marker_line_width=0)
fig.add_vline(x=avg, line_dash="dash", line_color="#fbbf24", annotation_text=f"Average ${avg:,.0f}")
fig.update_xaxes(title="Price ($)")
fig.update_yaxes(title="Number of cars")
st.plotly_chart(style_fig(fig), width="stretch")
q1, q3 = df[TARGET].quantile([.25, .75])
insight(f"Half of all cars cost between <b>${q1:,.0f}</b> and <b>${q3:,.0f}</b>.")

# ---- Brand + type
section("2 · Brand and body style", "Which brands and car types are priced higher?")
col1, col2 = st.columns(2)
with col1:
    fig, g = bar_avg("Manufacturer Name", "Average price by brand")
    st.plotly_chart(fig, width="stretch")
    h, hp, l, lp = best_worst(g, "Manufacturer Name")
    insight(f"<b>{h}</b> is the priciest brand (${hp:,.0f}); <b>{l}</b> is the cheapest (${lp:,.0f}).")
with col2:
    fig, g = bar_avg("Car Type", "Average price by car type")
    st.plotly_chart(fig, width="stretch")
    h, hp, l, lp = best_worst(g, "Car Type")
    insight(f"<b>{h}</b> cars cost the most on average; <b>{l}</b> cars the least.")

# ---- Energy + gearbox
section("3 · Fuel and gearbox", "Does the engine or transmission change the price?")
col1, col2 = st.columns(2)
with col1:
    fig, g = bar_avg("Energy", "Average price by fuel type")
    st.plotly_chart(fig, width="stretch")
    h, hp, l, lp = best_worst(g, "Energy")
    insight(f"<b>{h}</b> cars are the most expensive, <b>{l}</b> the cheapest.")
with col2:
    fig, g = bar_avg("Gearbox", "Average price by gearbox")
    st.plotly_chart(fig, width="stretch")
    h, hp, l, lp = best_worst(g, "Gearbox")
    insight(f"<b>{h}</b> gearbox cars cost about ${hp - lp:,.0f} more than <b>{l}</b>.")

# ---- Age
section("4 · Age of the car", "Newer cars versus older cars.")
yr = df.groupby("Manufactured Year")[TARGET].mean().reset_index()
fig = px.line(yr, x="Manufactured Year", y=TARGET, markers=True, title="Average price by year of manufacture")
fig.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=8, color="#818cf8"))
fig.update_xaxes(title="Year built", dtick=1)
fig.update_yaxes(title="Average price ($)")
st.plotly_chart(style_fig(fig), width="stretch")
slope = (yr[TARGET].iloc[-1] - yr[TARGET].iloc[0]) / (yr["Manufactured Year"].iloc[-1] - yr["Manufactured Year"].iloc[0])
word = "gains" if slope > 0 else "loses"
insight(f"Each extra year of age {'adds' if slope < 0 else 'removes'} roughly <b>${abs(slope):,.0f}</b> "
        f"– the average price {word} about ${abs(slope):,.0f} per newer model year.")

# ---- Mileage & power
section("5 · Mileage and engine power", "Cars are grouped into easy-to-read ranges.")
col1, col2 = st.columns(2)
with col1:
    m = df.assign(Range=pd.cut(df["Mileage-KM"], [0, 20000, 40000, 60000, 80000, 100000],
                               labels=["0–20k", "20–40k", "40–60k", "60–80k", "80–100k"]))
    g = m.groupby("Range", observed=True)[TARGET].mean().reset_index()
    fig = px.bar(g, x="Range", y=TARGET, title="Average price by kilometers driven",
                 text=g[TARGET].map("${:,.0f}".format))
    fig.update_traces(marker_color="#34d399", textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="Kilometers driven")
    fig.update_yaxes(title="Average price ($)", range=[0, g[TARGET].max() * 1.15])
    st.plotly_chart(style_fig(fig), width="stretch")
    diff = g[TARGET].iloc[0] - g[TARGET].iloc[-1]
    insight(f"Cars with the fewest km cost about <b>${abs(diff):,.0f}</b> "
            f"{'more' if diff > 0 else 'less'} than the highest-km cars.")
with col2:
    p = df.assign(Range=pd.cut(df["Engine Power-HP"], [0, 100, 150, 200, 260],
                               labels=["Up to 100 HP", "100–150 HP", "150–200 HP", "200+ HP"]))
    g = p.groupby("Range", observed=True)[TARGET].mean().reset_index()
    fig = px.bar(g, x="Range", y=TARGET, title="Average price by engine power",
                 text=g[TARGET].map("${:,.0f}".format))
    fig.update_traces(marker_color="#fbbf24", textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Average price ($)", range=[0, g[TARGET].max() * 1.15])
    st.plotly_chart(style_fig(fig), width="stretch")
    diff = g[TARGET].iloc[-1] - g[TARGET].iloc[0]
    insight(f"The most powerful cars cost about <b>${abs(diff):,.0f}</b> "
            f"{'more' if diff > 0 else 'less'} than the least powerful.")

# ---- Location & color
section("6 · Where and what color?")
col1, col2 = st.columns(2)
with col1:
    fig, g = bar_avg("Location", "Top 8 locations by average price", top=8)
    st.plotly_chart(fig, width="stretch")
with col2:
    fig, g = bar_avg("Color", "Average price by color")
    st.plotly_chart(fig, width="stretch")
    spread = g[TARGET].max() - g[TARGET].min()
    insight(f"Color barely matters: the gap between colors is only <b>${spread:,.0f}</b>.")

# ---- Ranking of influence
section("7 · What matters most?", "Longer bar = that feature is more strongly linked to price.")
num = ["Manufactured Year", "Mileage-KM", "Engine Power-HP", "Number of Seats", "Number of Doors"]
corr = df[num].corrwith(df[TARGET])
labels = {"Manufactured Year": "Newer car", "Mileage-KM": "More km driven",
          "Engine Power-HP": "More engine power", "Number of Seats": "More seats",
          "Number of Doors": "More doors"}
c = pd.DataFrame({"Feature": corr.index.map(labels), "Link": corr.values})
c["Effect"] = c["Link"].map(lambda v: "Raises price" if v >= 0 else "Lowers price")
c = c.reindex(c["Link"].abs().sort_values().index)
fig = px.bar(c, x="Link", y="Feature", orientation="h", color="Effect", title="Strength of link with price",
             color_discrete_map={"Raises price": "#34d399", "Lowers price": "#fb7185"})
fig.update_xaxes(title="← lowers price      |      raises price →")
fig.update_yaxes(title="")
st.plotly_chart(style_fig(fig, 340), width="stretch")
top = c.iloc[-1]
insight(f"Among numeric features, <b>{top['Feature'].lower()}</b> has the strongest link to price "
        f"({top['Effect'].lower()}).")

