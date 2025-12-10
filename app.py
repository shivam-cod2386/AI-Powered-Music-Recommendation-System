import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")

# --- 1. LOAD DATA (Backend) ---
# We use @st.cache_data to keep this in memory so it doesn't reload every time you click a button
@st.cache_data
def load_data():
    data = {
        'Song_Name': [
            'Blinding Lights', 'Shape of You', 'Bohemian Rhapsody', 'Stairway to Heaven', 
            'Bad Guy', 'Butter', 'Levitating', 'Hotel California', 'Imagine', 'Rolling in the Deep', 'Ehsaas'
        ],
        'Artist': [
            'The Weeknd', 'Ed Sheeran', 'Queen', 'Led Zeppelin', 
            'Billie Eilish', 'BTS', 'Dua Lipa', 'Eagles', 'John Lennon', 'Adele', 'Faheem Abdullah' 
        ],
        'Genre': [
            'Pop Synthwave', 'Pop', 'Rock Classic', 'Rock Classic', 
            'Pop Dark', 'K-Pop', 'Pop Disco', 'Rock Classic', 'Rock Ballad', 'Pop Soul', 'Romantic'
        ]
    }
    df = pd.DataFrame(data)
    # Combine Artist and Genre
    df['Content_Tags'] = df['Genre'] + " " + df['Artist']
    return df

df = load_data()

# --- 2. TRAIN MODEL (Backend) ---
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Content_Tags'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['Song_Name']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = train_model(df)

# --- 3. RECOMMENDATION ENGINE ---
def get_recommendations(title):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4] # Top 3
        song_indices = [i[0] for i in sim_scores]
        return df['Song_Name'].iloc[song_indices]
    except KeyError:
        return []

# --- 4. THE FRONT-END (UI) ---
st.title("🎵 AI Music Recommender")
st.markdown("Select a song you like, and we'll suggest similar tracks based on **Genre** and **Artist**.")

# Display the dataframe (Optional, good for debugging)
with st.expander("View Music Database"):
    st.dataframe(df)

# Dropdown box for user selection
song_list = df['Song_Name'].values
selected_song = st.selectbox("Type or select a song you like:", song_list)

# Button to trigger recommendation
if st.button("Recommend"):
    with st.spinner('Calculating similarities...'):
        recommendations = get_recommendations(selected_song)
        
    st.success(f"Because you liked **{selected_song}**, you might like:")
    
    # Display results nicely
    for i, song in enumerate(recommendations):

        st.subheader(f"{i+1}. {song}")
