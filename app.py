
import streamlit as st
import pandas as pd
import joblib
import json
import os

# Page Config
st.set_page_config(
    page_title="Spotify Harmony | AI Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown('
<style>
.main { background-color: #121212; color: #FFFFFF; }
.stButton>button { width: 100%; border-radius: 20px; background-color: #1DB954; color: white; }
.stSelectbox { color: #1DB954; }
</style>
', unsafe_allow_html=True)

@st.cache_resource
def load_production_artifacts():
    engine = joblib.load('models/recommendation_engine.pkl')
    lookup = joblib.load('models/song_lookup.pkl')
    matrix = joblib.load('models/feature_matrix.pkl')
    with open('deployment/metadata.json', 'r') as f:
        meta = json.load(f)
    return engine, lookup, matrix, meta

try:
    engine, lookup, matrix, meta = load_production_artifacts()
    lookup['display_name'] = lookup['song_name'].astype(str) + " (" + lookup['genre'].astype(str) + ")"
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.image("https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-icon-green-logo-8.png", width=100)
st.sidebar.title("Spotify AI Engine")
st.sidebar.info(f"Dataset: {meta['dataset_name']}\n\nSongs Indexed: {meta['number_of_songs']}")

# --- Main Interface ---
st.title("🎵 Spotify Song Recommendation System")
st.markdown("--- ")

selected_song_display = st.selectbox(
    "Search or select a song from the database:",
    options=lookup['display_name'].values,
    index=0
)

selected_idx = lookup[lookup['display_name'] == selected_song_display].index[0]

if st.button("Generate Recommendations"):
    with st.spinner("Analyzing audio features and finding matches..."):
        distances, indices = engine.kneighbors(matrix.getrow(selected_idx), n_neighbors=6)
        rec_indices = indices.flatten()[1:]
        rec_distances = distances.flatten()[1:]
        
        st.subheader("Recommended for You")
        cols = st.columns(5)
        
        for i, idx in enumerate(rec_indices):
            with cols[i]:
                song_data = lookup.iloc[idx]
                st.image("https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-icon-green-logo-8.png", width=50)
                st.markdown(f"**{song_data['song_name']}**")
                st.caption(f"Genre: {song_data['genre']}")
                st.markdown(f"[Listen on Spotify]({song_data['uri']})")
                similarity = (1 - rec_distances[i]) * 100
                st.progress(int(similarity))
                st.caption(f"Match: {similarity:.1f}%")

st.markdown("--- ")
st.caption("Built with Scikit-Learn NearestNeighbors & Streamlit Community Cloud")
