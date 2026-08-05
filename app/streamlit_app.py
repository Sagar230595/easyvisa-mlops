"""EasyVisa visa-approval predictor - Streamlit UI.

Loads the champion model from the Unity Catalog registry and scores a single
application entered by the user. Reuses the exact feature engineering from the
training package so the UI and the model agree.

Run locally:   streamlit run app/streamlit_app.py
On Databricks: deploy as a Databricks App (see app.yaml).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st

from easyvisa.config import CATEGORY_OPTIONS, FEATURES, Config
from easyvisa.models import ChampionModel
from easyvisa.transforms import annualise_wage, company_age, fix_employee_count

st.set_page_config(page_title="EasyVisa approval predictor", page_icon="check", layout="centered")


@st.cache_resource(show_spinner="Loading champion model...")
def load(env: str):
    cfg = Config.load(env)
    return cfg, ChampionModel(cfg)


st.title("EasyVisa - visa approval predictor")
st.write("Enter the application details and get the model's certification prediction.")

env = st.sidebar.selectbox("Environment", ["prod", "dev"], index=0)
try:
    cfg, champion = load(env)
    st.sidebar.success(f"Model: {champion.model_name}\nversion {champion.version}")
except Exception as exc:
    st.error(f"Could not load the champion model: {exc}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    continent = st.selectbox("Continent", CATEGORY_OPTIONS["continent"])
    education = st.selectbox("Education of employee", CATEGORY_OPTIONS["education_of_employee"])
    has_exp = st.selectbox("Has job experience", CATEGORY_OPTIONS["has_job_experience"])
    req_train = st.selectbox("Requires job training", CATEGORY_OPTIONS["requires_job_training"])
    region = st.selectbox("Region of employment", CATEGORY_OPTIONS["region_of_employment"])
with col2:
    full_time = st.selectbox("Full-time position", CATEGORY_OPTIONS["full_time_position"])
    unit = st.selectbox("Unit of wage", CATEGORY_OPTIONS["unit_of_wage"])
    wage = st.number_input("Prevailing wage", min_value=0.0, value=70000.0, step=1000.0)
    n_emp = st.number_input("Number of employees", min_value=0, value=1500, step=50)
    yr = st.number_input("Year of establishment", min_value=1800, max_value=2025, value=2000)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "no_of_employees": fix_employee_count(n_emp),
        "company_age": company_age(yr, cfg.reference_year),
        "annual_wage": annualise_wage(wage, unit),
        "continent": continent,
        "education_of_employee": education,
        "has_job_experience": has_exp,
        "requires_job_training": req_train,
        "region_of_employment": region,
        "unit_of_wage": unit,
        "full_time_position": full_time,
    }])[FEATURES]

    proba = float(champion.predict_proba(row)[0])
    label = "Certified" if proba >= 0.5 else "Denied"

    st.subheader(f"Prediction: {label}")
    st.metric("Approval probability", f"{proba * 100:.1f}%")
    st.progress(proba)
    with st.expander("Engineered features sent to the model"):
        st.dataframe(row.T.rename(columns={0: "value"}))
