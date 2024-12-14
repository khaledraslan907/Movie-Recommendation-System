# Movie-Recommendation-System
This project is a simple movie recommendation system built using Python and deployed using Streamlit. Users can input a movie name and receive recommendations for similar movies based on their features.

# Features

Input a movie name to get 10 recommended movies.

Uses TF-IDF Vectorization and Cosine Similarity to find similar movies.

Streamlit-based web interface.

# Dataset

The application uses a dataset (movies.csv) containing movie information. Ensure this file is placed in the project directory.

# CI/CD

This project includes a GitHub Actions workflow for continuous integration and deployment. To set up:

Add a STREAMLIT_CLOUD_API_TOKEN as a secret in your GitHub repository.

Push changes to the main branch to trigger the workflow.
