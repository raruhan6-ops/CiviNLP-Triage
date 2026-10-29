import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src/ to path if needed (for local dev without `pip install -e .`)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from civinlp.data.download_roads_311 import download_roads_311
from civinlp.data.preprocess_bridges import clean_bridges
from civinlp.data.preprocess_roads import normalize_zip, preprocess_roads
from civinlp.models.baseline_rules import RuleBaseline
from civinlp.models.tfidf_lr import TfidfBaseline
from civinlp.ranking.bridge_priority import BridgePriorityScorer
from civinlp.ranking.road_priority import RoadPriorityScorer
from civinlp.utils.io import load_parquet

st.set_page_config(page_title="CiviNLP Triage", layout="wide", page_icon="🏗️")

st.title("🏗️ CiviNLP Triage: Maintenance Prioritization")
st.markdown("*Intelligent road & bridge maintenance prioritization using NLP*")

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")
asset_type = st.sidebar.radio("Asset", ["Roads", "Bridges"])


# ─── Bridges ──────────────────────────────────────────────────────────────────


def show_bridges():
    """Rank bridges from an NBI CSV upload or the processed bridge file."""
    st.subheader("Bridge Priority Ranking (NBI condition ratings)")
    proc_path = Path("data_processed/bridges_processed.parquet")
    uploaded = st.sidebar.file_uploader(
        "Upload NBI CSV", type=["csv"], key="nbi_upload"
    )

    if uploaded is not None:
        bridges = clean_bridges(pd.read_csv(uploaded, dtype=str, encoding="latin1"))
    elif proc_path.exists():
        bridges = load_parquet(proc_path)
    else:
        st.info(
            "👈 Upload an NBI CSV (from FHWA), or run "
            "`python -m civinlp.data.preprocess_bridges` first."
        )
        return

    traffic_weight = st.sidebar.slider("Traffic weight", 0.0, 2.0, 0.5, 0.1)
    ranked = BridgePriorityScorer(traffic_weight=traffic_weight).rank_bridges(bridges)

    n_missing = int(ranked.get("all_ratings_missing", pd.Series(dtype=bool)).sum())
    st.write(f"**{len(ranked)}** bridges ranked.")
    if n_missing:
        st.warning(
            f"{n_missing} bridges have no usable condition rating (severity shown as 0)."
        )
    st.dataframe(ranked, width="stretch")
    st.bar_chart(ranked.head(20).set_index("bridge_id")["priority_score"])
    st.download_button(
        label="Download Bridge Rankings CSV",
        data=ranked.to_csv(index=False),
        file_name="civinlp_bridge_rankings.csv",
        mime="text/csv",
    )


if asset_type == "Bridges":
    show_bridges()
    st.stop()

data_source = st.sidebar.radio("Data Source", ["Use Sample Data", "Upload CSV"])
model_choice = st.sidebar.selectbox(
    "Model",
    ["TF-IDF + Logistic Regression", "Rule Baseline"],
)


# ─── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_data
def load_sample_data():
    """Load or download sample data."""
    raw_path = "data_raw/roads_311.parquet"
    proc_path = "data_processed/roads_processed.parquet"

    if not Path(raw_path).exists():
        download_roads_311(raw_path, limit=500)

    if not Path(proc_path).exists():
        preprocess_roads(raw_path, proc_path)

    return load_parquet(proc_path)


@st.cache_resource
def _load_tfidf(model_path: str):
    """Load the TF-IDF model once per process, not on every rerun."""
    return TfidfBaseline.load(model_path)


def load_model(choice: str):
    """Load the selected model."""
    if choice == "TF-IDF + Logistic Regression":
        model_path = Path("models/tfidf_baseline.joblib")
        if model_path.exists():
            return _load_tfidf(str(model_path))
        else:
            st.warning("TF-IDF model not found. Falling back to Rule Baseline.")
            return RuleBaseline()
    return RuleBaseline()


def set_data(df: pd.DataFrame, key) -> bool:
    """Store a new dataset; drop predictions made on a different one. Returns True if new."""
    if st.session_state.get("data_key") == key:
        return False
    st.session_state["df"] = df
    st.session_state["data_key"] = key
    st.session_state.pop("pred_df", None)
    return True


# ─── Main Flow ────────────────────────────────────────────────────────────────

# Switching data source clears data and predictions from the previous source
if st.session_state.get("data_source") != data_source:
    for k in ("df", "pred_df", "data_key"):
        st.session_state.pop(k, None)
    st.session_state["data_source"] = data_source

if data_source == "Use Sample Data":
    if st.sidebar.button("🔄 Load/Refresh Data"):
        try:
            df = load_sample_data()
            st.session_state.pop("data_key", None)  # explicit refresh always reloads
            set_data(df, "sample")
            st.success(f"✅ Loaded {len(df)} records from sample data.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    upload_key = (
        ("upload", uploaded_file.name, uploaded_file.size)
        if uploaded_file is not None
        else None
    )
    if uploaded_file is not None and st.session_state.get("data_key") != upload_key:
        try:
            df = pd.read_csv(uploaded_file)
            # Ensure 'text' column exists
            if "text" not in df.columns:
                text_cols = [
                    c
                    for c in df.columns
                    if "complaint" in c.lower()
                    or "text" in c.lower()
                    or "descriptor" in c.lower()
                ]
                if text_cols:
                    df["text"] = (
                        df[text_cols].fillna("").astype(str).agg(" - ".join, axis=1)
                    )
                else:
                    st.error(
                        "CSV must have a 'text' column or complaint-related columns."
                    )
                    st.stop()
            if "area_id" not in df.columns:
                for col in ["incident_zip", "zip", "zip_code", "area"]:
                    if col in df.columns:
                        df["area_id"] = df[col].map(normalize_zip)
                        break
                else:
                    df["area_id"] = "UNKNOWN"
            set_data(df, upload_key)
            st.success(f"✅ Uploaded {len(df)} records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ─── Display Data & Predictions ──────────────────────────────────────────────

if "df" in st.session_state:
    df = st.session_state["df"]

    tab1, tab2, tab3 = st.tabs(["📋 Data", "🔍 Predictions", "📊 Priority Ranking"])

    # ── Tab 1: Data Overview ──
    with tab1:
        st.subheader("Road Complaints")
        st.write(f"**{len(df)}** records loaded.")
        st.dataframe(df.head(50), width="stretch")

    # ── Tab 2: Per-Record Predictions ──
    with tab2:
        st.subheader("Per-Record Predictions")
        if st.button("🚀 Run Predictions"):
            model = load_model(model_choice)
            texts = df["text"].fillna("").astype(str).tolist()

            with st.spinner("Running predictions..."):
                # Defect predictions
                defect_preds = [str(p) for p in model.predict(texts)]

                # Severity predictions
                rule_model = RuleBaseline()
                severity_preds = [str(s) for s in rule_model.predict_severity(texts)]

                # Confidence
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(texts)
                        confidences = [round(float(max(row)), 3) for row in proba]
                    except Exception:
                        confidences = [1.0] * len(texts)
                else:
                    confidences = [1.0] * len(texts)

            pred_df = df.copy()
            pred_df["predicted_defect"] = defect_preds
            pred_df["predicted_severity"] = severity_preds
            pred_df["confidence"] = confidences
            st.session_state["pred_df"] = pred_df

            st.success("Predictions complete!")

        if "pred_df" in st.session_state:
            pred_df = st.session_state["pred_df"]

            # Display columns
            display_cols = [
                "text",
                "predicted_defect",
                "predicted_severity",
                "confidence",
            ]
            display_cols = [c for c in display_cols if c in pred_df.columns]
            st.dataframe(pred_df[display_cols].head(100), width="stretch")

            # Summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Defect Distribution**")
                st.bar_chart(pred_df["predicted_defect"].value_counts())
            with col2:
                st.write("**Severity Distribution**")
                st.bar_chart(pred_df["predicted_severity"].value_counts())

            # Explanation snippets (TF-IDF features)
            if model_choice == "TF-IDF + Logistic Regression":
                model_path = Path("models/tfidf_baseline.joblib")
                if model_path.exists():
                    try:
                        loaded_model = _load_tfidf(str(model_path))
                        top_features = loaded_model.get_top_features(n=5)
                        st.write("**Top TF-IDF Features per Class**")
                        for cls, features in top_features.items():
                            st.write(f"- **{cls}**: {', '.join(features)}")
                    except Exception:
                        pass

            # Download predictions
            st.subheader("📥 Download")
            csv_buffer = io.StringIO()
            pred_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="civinlp_predictions.csv",
                mime="text/csv",
            )

    # ── Tab 3: Priority Ranking ──
    with tab3:
        st.subheader("Area Priority Ranking")

        if "pred_df" not in st.session_state:
            st.info("Run predictions first (tab 2) to generate rankings.")
        else:
            pred_df = st.session_state["pred_df"]
            if "area_id" in pred_df.columns:
                scorer = RoadPriorityScorer()
                ranked = scorer.rank_areas(pred_df)
                st.write(f"**{len(ranked)}** areas ranked.")
                st.dataframe(ranked, width="stretch")

                # Bar chart of top areas
                top_n = min(20, len(ranked))
                chart_data = ranked.head(top_n).set_index("area_id")["priority_score"]
                st.bar_chart(chart_data)

                # Download rankings
                csv_buffer = io.StringIO()
                ranked.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Rankings CSV",
                    data=csv_buffer.getvalue(),
                    file_name="civinlp_rankings.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No 'area_id' column found for area-level ranking.")

else:
    st.info("👈 Select a data source and load data to begin.")
