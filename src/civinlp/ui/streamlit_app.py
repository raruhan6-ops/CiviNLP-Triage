import streamlit as st
import pandas as pd
from pathlib import Path
import os
import sys

# Add src to path if needed (for local dev)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from civinlp.data.download_roads_311 import download_roads_311
from civinlp.data.preprocess_roads import preprocess_roads
from civinlp.ranking.road_priority import RoadPriorityScorer
from civinlp.utils.io import load_parquet

st.set_page_config(page_title="CiviNLP Triage", layout="wide")

st.title("CiviNLP Triage: Maintenance Prioritization")

st.sidebar.header("Data Loading")
data_source = st.sidebar.radio("Source", ["Use Sample Data", "Upload CSV"])

@st.cache_data
def load_data():
    # Helper to get sample data
    # In a real app, this would be robust.
    raw_path = "data_raw/roads_311.parquet"
    proc_path = "data_processed/roads_processed.parquet"
    
    if not Path(raw_path).exists():
        st.info("Downloading sample data...")
        download_roads_311(raw_path, limit=500)
        
    if not Path(proc_path).exists():
        st.info("Preprocessing data...")
        preprocess_roads(raw_path, proc_path)
        
    return load_parquet(proc_path)

if data_source == "Use Sample Data":
    if st.sidebar.button("Load/Refresh Data"):
        try:
            df = load_data()
            st.session_state['df'] = df
            st.success("Data loaded!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
if "df" in st.session_state:
    df = st.session_state['df']
    st.subheader("Road Complaints")
    st.write(f"Loaded {len(df)} records.")
    st.dataframe(df.head())
    
    st.subheader("Analysis")
    if st.button("Run Priority Scoring"):
        scorer = RoadPriorityScorer()
        ranked = scorer.rank_areas(df)
        st.write("Top Priority Areas")
        st.dataframe(ranked)
        
        st.bar_chart(ranked.set_index("area_id")['priority_score'].head(20))
else:
    st.info("Please load data to begin.")
