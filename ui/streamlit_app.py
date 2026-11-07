import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

NICE = {
    "status_checking": "Checking account status", "duration_months": "Loan duration (months)",
    "credit_history": "Credit history", "purpose": "Purpose", "credit_amount": "Credit amount (DM)",
    "savings": "Savings account / bonds", "employment_since": "Employed since",
    "installment_rate": "Installment rate (% of disposable income, 1-4)",
    "other_debtors": "Other debtors / guarantors", "residence_since": "Present residence since (1-4)",
    "property": "Property", "age": "Age (years)", "other_installment_plans": "Other installment plans",
    "housing": "Housing", "existing_credits": "Existing credits at this bank",
    "job": "Job", "num_dependents": "People liable for maintenance",
    "telephone": "Telephone",
}
DEFAULTS = {"duration_months": 24, "credit_amount": 3000, "installment_rate": 3,
            "residence_since": 2, "age": 35, "existing_credits": 1, "num_dependents": 1}
STEP = {"credit_amount": 100}
DEFAULT_CAT = {
    "status_checking": "A11", "credit_history": "A32", "purpose": "A43", "savings": "A61",
    "employment_since": "A73", "other_debtors": "A101", "property": "A121",
    "other_installment_plans": "A143", "housing": "A152", "job": "A173", "telephone": "A191",
}

st.set_page_config(page_title="Loan default risk", page_icon="💳", layout="wide")
st.title("Loan default risk estimator")
st.warning("Educational demo. This is a decision-support estimate, not a lending decision.")


@st.cache_data(ttl=30)
def get_schema():
    r = requests.get(f"{API_URL}/schema", timeout=5)
    r.raise_for_status()
    return r.json()


try:
    schema = get_schema()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it first.")
    st.stop()

tab_predict, tab_stats = st.tabs(["Estimate risk", "Service statistics"])

with tab_predict:
    with st.form("applicant"):
        cols = st.columns(3)
        values = {}
        i = 0
        for f in schema["numeric"]:
            lo, hi = schema["hard_limits"][f]
            with cols[i % 3]:
                values[f] = st.number_input(NICE.get(f, f), min_value=int(lo), max_value=int(hi),
                                            value=int(DEFAULTS.get(f, lo)), step=STEP.get(f, 1))
            i += 1
        for f in schema["categorical"]:
            labels = schema["labels"][f]
            options = list(labels)
            default_code = DEFAULT_CAT.get(f, options[0])
            default_index = options.index(default_code) if default_code in options else 0
            with cols[i % 3]:
                values[f] = st.selectbox(NICE.get(f, f), options=options, index=default_index,
                                         format_func=lambda c, l=labels: l[c], key=f)
            i += 1
        submitted = st.form_submit_button("Estimate risk")

    if submitted:
        try:
            r = requests.post(f"{API_URL}/predict", json=values, timeout=10)
        except requests.RequestException:
            st.error("Could not reach the API. Please try again.")
            st.stop()

        if r.status_code == 200:
            res = r.json()
            c1, c2 = st.columns(2)
            c1.metric("Risk score", f"{res['risk_score']:.2f}")
            c2.metric("Flag threshold", f"{res['threshold']:.2f}")
            st.progress(min(max(res["risk_score"], 0.0), 1.0))
            if res["risk_class"] == "high":
                st.error("Higher risk: refer for manual review.")
            else:
                st.success("Lower risk, based on this model alone.")
            for w in res["warnings"]:
                st.warning(w)
            if res["ignored_fields"]:
                st.info(f"Ignored fields (not used by the model): {', '.join(res['ignored_fields'])}")
            st.caption(f"Model version: {res['model_version']}")
            st.caption(res["notice"])
        elif r.status_code == 422:
            st.error("Please fix these inputs:")
            for e in r.json().get("errors", []):
                st.write("•", e)
        else:
            st.error(f"Service error ({r.status_code}). Please try again later.")

with tab_stats:
    if st.button("Refresh"):
        st.cache_data.clear()
    try:
        st.json(requests.get(f"{API_URL}/metrics", timeout=5).json())
        st.json(requests.get(f"{API_URL}/health", timeout=5).json())
    except requests.RequestException:
        st.error("Cannot reach the API.")