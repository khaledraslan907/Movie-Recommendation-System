import numpy as np
import pandas as pd
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the dataset
def load_data():
    # Replace 'movies.csv' with the actual path to your dataset
    movies_data = pd.read_csv(r'E:\Data analysis\Projects\learning\18\movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    movies_data['combined_features'] = movies_data[selected_features].apply(lambda row: ' '.join(row), axis=1)
    return movies_data

# Build recommendation system
def recommend_movies(movie_name, movies_data):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

    similarity = cosine_similarity(feature_vectors)

    all_movies = movies_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, all_movies, n=1)

    if not close_match:
        return ["No matching movie found."]

    index_of_movie = movies_data[movies_data.title == close_match[0]].index[0]
    similarity_scores = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i in range(1, 11):  # Skip the first movie (itself)
        recommended_movies.append(movies_data.iloc[sorted_similar_movies[i][0]].title)

    return recommended_movies

# Streamlit App
st.title("Movie Recommendation System")
st.write("Enter a movie name to get recommendations!")

movies_data = load_data()

movie_name = st.text_input("Movie Name", "")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = recommend_movies(movie_name, movies_data)
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
