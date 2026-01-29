# rank.py

import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Stage-wise Operator Ranking",
    layout="wide"
)

st.title("🏭 Stage-wise Operator Suitability Dashboard")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# --------------------------------------------------
# Features
# --------------------------------------------------
features = [
    'Model','Line','Stage','Stage_Criticality',
    'Units_Checked','Total_Errors','Defect_Rate',
    'Pass_Ratio','Error_Std_Dev','Critical_Exposure'
]

target = 'Suitability_Score'
cat_features = ['Model','Line','Stage','Stage_Criticality']

# --------------------------------------------------
# Train Model
# --------------------------------------------------
@st.cache_resource
def train_model(X, y):
    model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function='RMSE',
        verbose=False,
        random_seed=42
    )
    model.fit(X, y, cat_features=cat_features)
    return model

model = train_model(df[features], df[target])
df["Predicted_Suitability"] = model.predict(df[features])

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

selected_stage = st.sidebar.selectbox(
    "Select Stage",
    sorted(df["Stage"].unique())
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    sorted(df["Model"].unique())
)

filtered_df = df[
    (df["Stage"] == selected_stage) &
    (df["Model"] == selected_model)
]

# --------------------------------------------------
# Stage-wise Ranking
# --------------------------------------------------
ranking = (
    filtered_df
    .groupby(["Emp_ID","Emp_Name"])
    .agg(
        Avg_Suitability=("Predicted_Suitability","mean"),
        Avg_Defect_Rate=("Defect_Rate","mean"),
        Consistency=("Predicted_Suitability","std")
    )
    .reset_index()
)

ranking["Consistency"] = ranking["Consistency"].fillna(0)
ranking["Consistency_Score"] = 1 / (1 + ranking["Consistency"])

ranking = ranking.sort_values("Avg_Suitability", ascending=False)
ranking["Rank"] = range(1, len(ranking) + 1)

# --------------------------------------------------
# Recommendation Logic
# --------------------------------------------------
def recommend(row):
    if row.Avg_Suitability >= 90 and row.Consistency_Score >= 0.7:
        return f"Assign to {selected_stage}"
    elif row.Avg_Suitability >= 85:
        return "Safe"
    elif row.Avg_Suitability >= 75:
        return "Backup"
    else:
        return "Avoid critical"

ranking["Recommendation"] = ranking.apply(recommend, axis=1)

# --------------------------------------------------
# TOP 5 Operators Table
# --------------------------------------------------
st.subheader(f"📊 Top 5 Operators – {selected_stage} ({selected_model})")

top5 = ranking.head(5)[
    ["Rank","Emp_Name","Avg_Suitability","Recommendation"]
]

top5.columns = ["Rank","Employee","Avg Suitability","Recommendation"]
top5["Avg Suitability"] = top5["Avg Suitability"].round(2)

st.dataframe(top5, use_container_width=True)

# --------------------------------------------------
# Key Insights
# --------------------------------------------------
best = ranking.iloc[0]
worst = ranking.iloc[-1]

st.subheader("📌 Key Insights")

c1, c2, c3 = st.columns(3)

c1.metric(
    "✅ Best Operator",
    best.Emp_Name,
    f"{best.Avg_Suitability:.2f}"
)

c2.metric(
    "⚠️ Risky Operator",
    worst.Emp_Name,
    f"{worst.Avg_Suitability:.2f}"
)

c3.metric(
    "👥 Total Operators",
    len(ranking)
)

# --------------------------------------------------
# Risky Operators (High Defect + Exposure)
# --------------------------------------------------
st.subheader("⚠️ High Risk Operators")

risk_table = ranking.sort_values(
    ["Avg_Defect_Rate","Avg_Suitability"],
    ascending=[False, True]
).head(5)

risk_table = risk_table[
    ["Emp_Name","Avg_Defect_Rate","Avg_Suitability"]
]

risk_table.columns = ["Employee","Defect Rate","Avg Suitability"]
risk_table["Defect Rate"] = risk_table["Defect Rate"].round(4)
risk_table["Avg Suitability"] = risk_table["Avg Suitability"].round(2)

st.dataframe(risk_table, use_container_width=True)

# --------------------------------------------------
# Stage Health Insight
# --------------------------------------------------
st.subheader("🏗 Stage Health Overview")

stage_health = (
    df.groupby("Stage")
    .agg(
        Avg_Suitability=("Predicted_Suitability","mean"),
        Avg_Defect_Rate=("Defect_Rate","mean")
    )
    .reset_index()
)

stage_health["Avg_Suitability"] = stage_health["Avg_Suitability"].round(2)
stage_health["Avg_Defect_Rate"] = stage_health["Avg_Defect_Rate"].round(4)

st.dataframe(stage_health, use_container_width=True)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.subheader("🧠 What the Model Cares About")

imp = model.get_feature_importance(prettified=True)
st.dataframe(imp, use_container_width=True)
