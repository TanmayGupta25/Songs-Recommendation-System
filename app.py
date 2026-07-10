import streamlit as st
import pandas as pd
import joblib
import json
from scipy.sparse import issparse

# Page Configuration
st.set_page_config(
    page_title="Spotify Harmony | AI Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        color: #FFFFFF;
    }

    .stButton > button {
        width: 100%;
        border-radius: 20px;
        background-color: #1DB954;
        color: white;
        border: none;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #18a64d;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Resources
@st.cache_resource
def load_production_artifacts():

    engine = joblib.load("models/recommendation_engine.pkl")
    lookup = joblib.load("models/song_lookup.pkl")
    feature_matrix = joblib.load("models/feature_matrix.pkl")

    with open("deployment/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return engine, lookup, feature_matrix, metadata


try:

    engine, lookup, feature_matrix, metadata = load_production_artifacts()

    lookup = lookup.reset_index(drop=True)

    required_columns = [
        "song_name",
        "genre",
        "id",
        "uri"
    ]

    for col in required_columns:
        if col not in lookup.columns:
            raise ValueError(f"Missing required column: {col}")

    lookup["display_name"] = (
        lookup["song_name"].fillna("Unknown Song").astype(str)
        + "  |  "
        + lookup["genre"].fillna("Unknown").astype(str)
    )

except Exception as e:

    st.error(f"Application failed to load:\n\n{e}")
    st.stop()


# Sidebar
st.sidebar.image(
    "https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-icon-green-logo-8.png",
    width=110
)

st.sidebar.title("Spotify Harmony")

st.sidebar.success(
    f"Songs Indexed : {metadata.get('number_of_songs','N/A')}"
)

st.sidebar.info(
    f"""
Dataset

{metadata.get('dataset_name','N/A')}

Algorithm

Nearest Neighbors

Similarity

Cosine Distance
"""
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎵 Recommend Songs",
        "📊 Dataset Information",
        "👨‍💻 Developer"
    ]
)

# Home
if page == "🏠 Home":

    st.title("🎵 Spotify Songs Recommendation System")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Songs",
            metadata.get("number_of_songs", "N/A")
        )

    with c2:
        st.metric(
            "Algorithm",
            "NearestNeighbors"
        )

    with c3:
        st.metric(
            "Similarity",
            "Cosine"
        )

    st.write(
        """
This application recommends songs based on audio characteristics
using a trained Nearest Neighbors recommendation engine.
"""
    )

# Recommendation Page
elif page == "🎵 Recommend Songs":

    st.title("🎵 Song Recommendation Engine")

    selected_song = st.selectbox(
        "Select a Song",
        lookup["display_name"].tolist()
    )

    selected_index = lookup.index[
        lookup["display_name"] == selected_song
    ][0]

    top_n = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if st.button("Generate Recommendations"):

        with st.spinner("Finding similar songs..."):

            query_vector = (
                feature_matrix.getrow(selected_index)
                if issparse(feature_matrix)
                else feature_matrix[selected_index:selected_index + 1]
            )

            distances, indices = engine.kneighbors(
                query_vector,
                n_neighbors=top_n + 1
            )

            recommendation_indices = indices.flatten()[1:]
            recommendation_distances = distances.flatten()[1:]
            st.success("Recommendations Generated")

            results = []

            for idx, dist in zip(
                recommendation_indices,
                recommendation_distances
            ):

                row = lookup.iloc[idx]

                similarity = max(
                    0.0,
                    (1 - float(dist)) * 100
                )

                results.append(
                    {
                        "Song": row["song_name"],
                        "Genre": row["genre"],
                        "Spotify URI": row["uri"],
                        "Similarity (%)": round(similarity, 2)
                    }
                )

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

# Dataset Information
elif page == "📊 Dataset Information":

    st.title("📊 Dataset Information")

    st.json(metadata)

    st.markdown("---")

    st.subheader("Sample Records")

    st.dataframe(
        lookup.head(20),
        use_container_width=True
    )

# Developer
elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

    st.markdown(
        """
### Spotify Songs Recommendation System

**Course**
- M.Sc. Data Science

**Technology Stack**
- Python
- Scikit-Learn
- NearestNeighbors
- Streamlit

**Deployment**
- Streamlit Community Cloud

**Repository**
- GitHub
"""
    )

st.markdown("---")

st.caption(
    "Spotify Songs Recommendation System | Assignment 2 | Streamlit Deployment Ready"
)
