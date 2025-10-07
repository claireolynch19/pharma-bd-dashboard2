import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import logging

logger = logging.getLogger("pharma_bd")

def fetch_fda_approvals(limit=100, retries=3, backoff=1.5):
    """Robust fetch for FDA approvals using openFDA with safe defaults and retries."""
    if limit is None or limit <= 0:
        limit = 100
    if limit > 100:
        limit = 100

    urls = [
        f"https://api.fda.gov/drug/drugsfda.json?search=products.marketing_status:'1'&limit={limit}",
        f"https://api.fda.gov/drug/drugsfda.json?limit={limit}"
    ]

    last_exception = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code != 200:
                    logger.warning(f"FDA API returned status {resp.status_code} for url {url}")
                    st.warning(f"FDA API returned status {resp.status_code} (attempt {attempt}). Retrying...")
                    time.sleep(backoff ** attempt)
                    last_exception = requests.HTTPError(f"Status {resp.status_code}")
                    continue
                data = resp.json()
                return data.get("results", [])
            except requests.RequestException as e:
                logger.warning(f"Error fetching FDA data: {e}")
                st.warning(f"Error contacting FDA API (attempt {attempt}). Retrying...")
                last_exception = e
                time.sleep(backoff ** attempt)

    raise RuntimeError("Failed to fetch FDA approvals after multiple attempts.") from last_exception

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

def fetch_clinical_trials(term="oncology", phase="Phase 3", max_studies=100, retries=2):
    base_url = "https://clinicaltrials.gov/api/query/study_fields"
    params = {
        "expr": f"{term} AND {phase}",
        "fields": "NCTId,Condition,Phase,StartDate,CompletionDate,Sponsors",
        "min_rnk": 1,
        "max_rnk": max_studies,
        "fmt": "json"
    }
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(base_url, params=params, timeout=20)
            if response.status_code != 200:
                logger.warning(f"ClinicalTrials.gov returned {response.status_code}")
                st.warning(f"ClinicalTrials.gov returned {response.status_code} (attempt {attempt}). Retrying...")
                time.sleep(1.5 ** attempt)
                last_exception = requests.HTTPError(f"Status {response.status_code}")
                continue
            return response.json()["StudyFieldsResponse"]["StudyFields"]
        except requests.RequestException as e:
            logger.warning(f"Error contacting ClinicalTrials.gov: {e}")
            st.warning("Error contacting ClinicalTrials.gov. Retrying...")
            last_exception = e
            time.sleep(1.5 ** attempt)

    raise RuntimeError("Failed to fetch ClinicalTrials.gov data after retries.") from last_exception

def parse_trials(data, phase):
    records = []
    for trial in data:
        trial_id = trial.get("NCTId", [""])[0]
        sponsor = trial.get("Sponsors", [""])[0]
        completion_date = trial.get("CompletionDate", [""])[0]
        if completion_date:
            records.append({
                "source": "ClinicalTrials.gov",
                "sponsor": sponsor,
                "date": completion_date,
                "phase": phase,
                "trial_id": trial_id
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df

# Streamlit App
st.set_page_config(page_title="Pharma Dashboard v1", layout="wide")
st.title("📊 Pharma R&D Pipeline & FDA Approvals Dashboard v1")

st.sidebar.header("Filters & Diagnostics")
term = st.sidebar.text_input("Therapeutic Area", "oncology")
phases = st.sidebar.multiselect("Trial Phases", ["Phase 2", "Phase 3"], default=["Phase 2", "Phase 3"])
fda_limit = st.sidebar.slider("Number of FDA Approvals to Fetch (<=100)", 10, 100, 50)

with st.spinner("Fetching FDA approvals..."):
    try:
        fda_raw = fetch_fda_approvals(limit=fda_limit)
        fda_df = parse_fda(fda_raw)
        st.success(f"Fetched {len(fda_raw)} FDA entries.")
    except Exception as e:
        st.error(f"FDA API Error: {e}")
        fda_df = pd.DataFrame()

trials_dfs = []
for ph in phases:
    with st.spinner(f"Fetching {ph} trials for {term}..."):
        try:
            trials_raw = fetch_clinical_trials(term=term, phase=ph, max_studies=200)
            trials_df = parse_trials(trials_raw, ph)
            trials_dfs.append(trials_df)
        except Exception as e:
            st.error(f"ClinicalTrials.gov Error ({ph}): {e}")

trials_df = pd.concat(trials_dfs, ignore_index=True) if trials_dfs else pd.DataFrame()
combined = pd.concat([fda_df, trials_df], ignore_index=True) if not fda_df.empty or not trials_df.empty else pd.DataFrame()
if not combined.empty:
    combined["month"] = combined["date"].dt.to_period("M")

st.subheader("Data Preview")
st.dataframe(combined.head(40))

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
