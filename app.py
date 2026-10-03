import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------------
# Load and Prepare Dataset
# -----------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("anime.csv")

    # Handle missing values
    df["genre"] = df["genre"].fillna("Unknown")
    df["type"] = df["type"].fillna("Unknown")
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Create content feature
    df["content"] = (
        df["genre"].fillna("") + " " +
        df["type"].fillna("")
    )

    df = df.reset_index(drop=True)

    return df


df = load_data()


# -----------------------------------
# TF-IDF Vectorization
# -----------------------------------

@st.cache_resource
def create_tfidf(data):

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(
        data["content"]
    )

    return tfidf_matrix


tfidf_matrix = create_tfidf(df)


# -----------------------------------
# Create Anime Index
# -----------------------------------

indices = pd.Series(
    df.index,
    index=df["name"]
)


# -----------------------------------
# Recommendation Function
# -----------------------------------

def recommend_anime(title, n=10):

    idx = indices[title]

    # Handle duplicate anime names
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Calculate cosine similarity
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Create recommendation dataframe
    recommendations = df.copy()

    recommendations["similarity_score"] = sim_scores

    # Remove selected anime
    recommendations = recommendations.drop(index=idx)

    # Sort by similarity and rating
    recommendations = recommendations.sort_values(
        by=["similarity_score", "rating"],
        ascending=[False, False]
    )

    return recommendations[
        [
            "name",
            "genre",
            "type",
            "rating",
            "members",
            "similarity_score"
        ]
    ].head(n).reset_index(drop=True)


# -----------------------------------
# Streamlit Interface
# -----------------------------------

st.title("🎬 Anime Recommendation System")

st.write(
    "Discover anime similar to your favourites "
    "using a content-based recommendation system."
)

st.divider()


# Anime Selection

anime_name = st.selectbox(
    "Select an Anime",
    sorted(df["name"].unique())
)


# Number of recommendations

number_of_recommendations = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# Recommendation Button

if st.button("Get Recommendations"):

    recommendations = recommend_anime(
        anime_name,
        number_of_recommendations
    )

    st.subheader(
        f"Recommended Anime Similar to {anime_name}"
    )

    st.dataframe(
        recommendations,
        use_container_width=True
    )