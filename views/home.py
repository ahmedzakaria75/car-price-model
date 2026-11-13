import pandas as pd
import streamlit as st

from utils import COLUMN_INFO, TARGET, hero, kpi, load_data, section

df = load_data()

hero("Used Car Sales Dashboard",
     "A friendly overview of the used-car dataset: what it contains and what each column means.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Cars", f"{len(df):,}", "records in the dataset")
with c2:
    kpi("Columns", f"{df.shape[1]}", "information per car")
with c3:
    kpi("Average price", f"${df[TARGET].mean():,.0f}", f"range ${df[TARGET].min():,} – ${df[TARGET].max():,}")
with c4:
    sold = (df["Car Sale Status"] == "Sold").mean() * 100
    kpi("Sold cars", f"{sold:.0f}%", "of all listed cars")

st.write("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Brands", f"{df['Manufacturer Name'].nunique()}", "manufacturers")
with c2:
    kpi("Car models", f"{df['Car Name'].nunique()}", "different models")
with c3:
    kpi("Years covered", f"{df['Manufactured Year'].min()} – {df['Manufactured Year'].max()}", "manufacture year")
with c4:
    kpi("Missing values", f"{int(df.isna().sum().sum())}", "data is complete" if df.isna().sum().sum() == 0 else "needs cleaning")

section("📖 Column guide", "What each column in the data means, with a quick example of its values.")


def summary(col: str) -> str:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return f"{s.min():,} → {s.max():,} (avg {s.mean():,.1f})"
    top = s.value_counts().head(3).index.tolist()
    return f"{s.nunique()} unique, e.g. " + ", ".join(map(str, top))


rows = [{
    "Column": c,
    "Type": COLUMN_INFO.get(c, ("", ""))[0],
    "Description": COLUMN_INFO.get(c, ("", ""))[1],
    "Values": summary(c),
} for c in df.columns]
guide = pd.DataFrame(rows)
q = st.text_input("🔎 Search columns", placeholder="e.g. price, date, color...")
if q:
    mask = guide.apply(lambda r: q.lower() in " ".join(r.astype(str)).lower(), axis=1)
    guide = guide[mask]
st.dataframe(guide, hide_index=True, width="stretch", height=min(35 * len(guide) + 40, 920))

section("👀 Data preview", "The first rows of the dataset.")
n = st.slider("Rows to show", 5, 50, 10)
st.dataframe(df.head(n), hide_index=True, width="stretch")

