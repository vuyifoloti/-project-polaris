import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "attrition_model.pkl"
DATA_PATH = BASE_DIR / "model_ready_dataset.csv"
DRIVERS_CSV = BASE_DIR / "top_drivers.csv"

clf_lr = joblib.load(MODEL_PATH)

clf_lr = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

st.title("Employee Attrition Predictive Model 💼")
st.write("Predicts the likihood of employees leaving Polaris HQ.")

numeric_features = [
    'Age','Education','JobInvolvement','JobLevel','JobSatisfaction',
    'NumCompaniesWorked','PerformanceRating','RelationshipSatisfaction',
    'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear',
    'WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
    'YearsSinceLastPromotion','YearsWithCurrManager','DeptAvgIncome',
    'RelIncomeRatio','MonthlyIncome','EnvironmentSatisfaction','DistanceFromHome'
]

categorical_features = [
    'BusinessTravel','Department','EducationField','Gender','JobRole',
    'MaritalStatus','OverTime','AgeBand','OrgIncomeBucket','Gender_Marital',
    'RoleTenureBand','RecentPromotion','EducationLevel','EnvSatLevel',
    'CommuteBand','JobHopperBand','MgrTenureBand','DeptIncomeBucket','Promo_AgeBand'
]

all_model_features = numeric_features + categorical_features

defaults = {}

for col in all_model_features:
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            # mode() can return empty if all NaN — guard it
            mode_vals = df[col].mode(dropna=True)
            defaults[col] = str(mode_vals.iloc[0]) if len(mode_vals) else ""
    else:
        # If the CSV doesn’t have a column (edge case), backfill sensible defaults
        defaults[col] = 0.0 if col in numeric_features else ""

st.header("🔮 Quick Prediction")

dept_opts = sorted(df['Department'].dropna().unique().tolist()) if 'Department' in df.columns else ["Sales","Research & Development","Human Resources"]
jobrole_opts = sorted(df['JobRole'].dropna().unique().tolist()) if 'JobRole' in df.columns else [
    "Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director",
    "Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"
]
overtime_opts = ["Yes","No"]

age = st.slider("Age", 18, 60, int(defaults.get('Age', 30)))
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000,
                                 value=5000, step=500)
overtime = st.selectbox("OverTime", overtime_opts, index=overtime_opts.index(defaults.get('OverTime', "No")) if defaults.get('OverTime',"No") in overtime_opts else 1)
businesstravel_opts = sorted(df['BusinessTravel'].dropna().unique().tolist()) if 'BusinessTravel' in df.columns else ["Non-Travel","Travel_Rarely","Travel_Frequently"]
business_travel = st.selectbox("Business Travel", businesstravel_opts)

row = defaults.copy()
row['Age'] = age
row['MonthlyIncome'] = monthly_income  
row['OverTime'] = overtime
row['BusinessTravel'] = business_travel

input_full = pd.DataFrame([{k: row.get(k, np.nan) for k in all_model_features}])

threshold = st.slider("Decision threshold (probability for 'Leave')", 0.05, 0.95, 0.30, 0.05)

if st.button("Predict"):
    proba = clf_lr.predict_proba(input_full)[0][1]
    pred = "Likely to Leave" if proba >= threshold else "Likely to Stay"

    st.subheader(f"Prediction: {pred}")
    st.write(f"Attrition Probability: **{proba:.2f}** (threshold = {threshold:.2f})")
    st.caption("Note: All other features set to typical values from your dataset.")

st.header("🏢 Attrition by Department")
try:
    dept_summary = (
        df.groupby("Department")["Attrition"]
          .value_counts(normalize=True)
          .unstack()
          .mul(100).round(1)
          .rename(columns={"No":"Stay %","Yes":"Leave %"})
          .sort_values("Leave %", ascending=False)
    )
    st.dataframe(dept_summary)
except Exception as e:
    st.warning(f"Could not compute department summary: {e}")

st.header("👔 Attrition by Job Role")
try:
    role_summary = (
        df.groupby("JobRole")["Attrition"]
          .value_counts(normalize=True)
          .unstack()
          .mul(100).round(1)
          .rename(columns={"No":"Stay %","Yes":"Leave %"})
          .sort_values("Leave %", ascending=False)
    )
    st.dataframe(role_summary)
except Exception as e:
    st.warning(f"Could not compute job role summary: {e}")

st.header("🔥 Top Drivers of Attrition (Logistic Regression)")
try:

    if DRIVERS_CSV.exists():
        coef_df = pd.read_csv(DRIVERS_CSV)

        top15 = coef_df.sort_values("abs_coeff", ascending=False).head(15)
        st.dataframe(top15)
        st.bar_chart(top15.set_index("feature")["abs_coeff"])
    else:
      
        pre = clf_lr.named_steps['preprocessor']
        ohe = pre.named_transformers_['cat'].named_steps['onehot']
        
        cat_names = ohe.get_feature_names_out(categorical_features)
        all_names = np.array(numeric_features + list(cat_names))

        coefs = clf_lr.named_steps['model'].coef_[0]
        coef_df = pd.DataFrame({
            "feature": all_names,
            "coefficient": coefs,
            "abs_coeff": np.abs(coefs)
        }).sort_values("abs_coeff", ascending=False)

        st.dataframe(coef_df.head(15))
        st.bar_chart(coef_df.head(15).set_index("feature")["abs_coeff"])
       
        try:
            coef_df.to_csv(DRIVERS_CSV, index=False)
        except Exception:
            pass
except Exception as e:
    st.warning(f"Could not display drivers: {e}")
