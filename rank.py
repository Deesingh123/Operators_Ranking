# rank.py
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Stage-wise Operator Suitability",
    layout="wide"
)

st.title("🏭 Stage-wise Operator Suitability")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# --------------------------------------------------
# Feature Columns
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
# Employee Ranking
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
# Top Operators
# --------------------------------------------------
st.subheader(f"📊 Top 5 Operators – {selected_stage} ({selected_model})")

top5 = ranking.head(5)[
    ["Rank","Emp_Name","Avg_Suitability","Recommendation"]
]

top5.columns = ["Rank","Employee","Avg Suitability","Recommendation"]
top5["Avg Suitability"] = top5["Avg Suitability"].round(2)

st.dataframe(top5, use_container_width=True)

# --------------------------------------------------
# Key Metrics
# --------------------------------------------------
st.subheader("📌 Key Management Insights")

c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 Operators", len(ranking))
c2.metric("⭐ Best Avg Score", f"{ranking.Avg_Suitability.max():.2f}")
c3.metric("⚠ Avg Defect Rate", f"{ranking.Avg_Defect_Rate.mean():.4f}")
c4.metric("🏗 Stage Criticality", selected_stage)

# --------------------------------------------------
# Risky Operators
# --------------------------------------------------
st.subheader("⚠️ Operators Needing Attention")

risk_ops = ranking[
    (ranking.Avg_Defect_Rate > ranking.Avg_Defect_Rate.mean()) &
    (ranking.Avg_Suitability < ranking.Avg_Suitability.mean())
].head(5)

st.dataframe(
    risk_ops[["Emp_Name","Avg_Defect_Rate","Avg_Suitability","Recommendation"]],
    use_container_width=True
)

# --------------------------------------------------
# Best Employee per Stage (Global Insight)
# --------------------------------------------------
st.subheader("🎯 Best Employee–Stage Fit")

best_fit = (
    df.groupby(["Stage","Emp_Name"])
    .agg(Avg_Suitability=("Predicted_Suitability","mean"))
    .reset_index()
)

best_fit = best_fit.sort_values("Avg_Suitability", ascending=False)\
                   .groupby("Stage").head(1)

st.dataframe(best_fit, use_container_width=True)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.subheader("🧠 What the Model Cares About")

imp = model.get_feature_importance(prettified=True)
st.dataframe(imp, use_container_width=True)
