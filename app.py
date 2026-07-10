
import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title='Spotify Recommender', page_icon='🎵', layout='wide')

@st.cache_resource
def load_data():
    engine = joblib.load('models/recommendation_engine.pkl')
    lookup = joblib.load('models/song_lookup.pkl')
    with open('deployment/metadata.json', 'r') as f: meta = json.load(f)
    return engine, lookup, meta

engine, lookup, meta = load_data()

st.title('🎵 Spotify Recommendation System')
st.sidebar.header('Navigation')
page = st.sidebar.radio('Select Page', ['Home', 'Recommend', 'Details'])

if page == 'Home':
    st.subheader('Project Overview')
    st.write(f"Analyzing {meta.get('number_of_songs')} songs using Content-Based Filtering.")

elif page == 'Recommend':
    song_choice = st.selectbox('Pick a song', lookup.index)
    if st.button('Get Recommendations'):
        st.success('Finding similar tracks...')
