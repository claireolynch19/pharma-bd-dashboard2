import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# FDA Approvals Functions
# -------------------------------
def fetch_fda_approvals(limit=100):
    url = f"https://api.fda.gov/drug/drugsfda.json?search=products.marketing_status:'1'&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("results", [])

def parse_fda(data):
    records = []
    for entry in data:
        sponsor = entry.get("sponsor_name")
        submissions = entry.get("submissions", [])
        for sub in submissions:
            approval_date = sub.get("submission_date")
            if approval_date:
                records.append({
                    "source": "FDA",
                    "sponsor": sponsor,
                    "date": approval_date,
                    "phase": None,
                    "trial_id": None,
                    "submission_type": sub.get("submission_type"),
                    "submission_class": sub.get("submission_class_code")
                })
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df

# -------------------------------
# ClinicalTrials.gov Functions
# -------------------------------
def fetch_clinical_trials(term="oncology", phase="Phase 3", max_studies=100):
    base_url = "https://clinicaltrials.gov/api/query/study_fields"
    params = {
        "expr": f"{term} AND {phase}",
        "fields": "NCTId,Condition,Phase,StartDate,CompletionDate,Sponsors",
        "min_rnk": 1,
        "max_rnk": max_studies,
        "fmt": "json"
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()["StudyFieldsResponse"]["StudyFields"]

def parse_trials(data, phase):
    records = []
    for trial in data:
        trial_id = trial.get("NCTId", [""])[0]
        sponsor = trial.get("Sponsors", [""])[0]
        completion_date = trial.get("CompletionDate", [""])[0]

        if completion_date:  # focus on completions as milestone
            records.append({
                "source": "ClinicalTrials.gov",
                "sponsor": sponsor,
                "date": completion_date,
                "phase": phase,
                "trial_id": trial_id,
                "submission_type": None,
                "submission_class": None
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Pharma BD Dashboard", layout="wide")

st.title("📊 Pharma R&D Pipeline & FDA Approvals Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
term = st.sidebar.text_input("Therapeutic Area", "oncology")
phases = st.sidebar.multiselect("Trial Phases", ["Phase 2", "Phase 3"], default=["Phase 2", "Phase 3"])
fda_limit = st.sidebar.slider("Number of FDA Approvals to Fetch", 50, 500, 200)

# Load FDA data
with st.spinner("Fetching FDA approvals..."):
    fda_raw = fetch_fda_approvals(limit=fda_limit)
    fda_df = parse_fda(fda_raw)

# Load ClinicalTrials data
trials_dfs = []
for ph in phases:
    with st.spinner(f"Fetching {ph} trials for {term}..."):
        trials = parse_trials(fetch_clinical_trials(term=term, phase=ph, max_studies=200), ph)
        trials_dfs.append(trials)

trials_df = pd.concat(trials_dfs, ignore_index=True) if trials_dfs else pd.DataFrame()

# Combine
combined = pd.concat([fda_df, trials_df], ignore_index=True)
if not combined.empty:
    combined["month"] = combined["date"].dt.to_period("M")

# Show data
st.subheader("Data Preview")
st.dataframe(combined.head(20))

# Trends plot
if not combined.empty:
    st.subheader("Activity Over Time")
    counts = combined.groupby([combined["month"], "source"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="line", marker="o", ax=ax)
    plt.title(f"FDA Approvals & Trials in {term}")
    plt.xlabel("Month")
    plt.ylabel("Count")
    st.pyplot(fig)
else:
    st.warning("No data available for the selected filters.")
