import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your saved model components
filtered_df = joblib.load("filtered_df.pkl")
tfidf_similarity = joblib.load("tfidf_similarity.pkl")
embedding_similarity = joblib.load("embedding_similarity.pkl")

class RecommenderEvaluator:
    def __init__(self, filtered_df):
        self.filtered_df = filtered_df

    def get_recommendations_with_explanation(self, title, hybrid_similarity, n_recommendations=5):
        idx = self.filtered_df[self.filtered_df['original_title'] == title].index[0]
        sim_scores = list(enumerate(hybrid_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        recommendations = [{
            'title': self.filtered_df['original_title'].iloc[i[0]],
            'similarity_score': i[1],
            'explanation': 'Based on similar tags and user feedback'  # Customize this as needed
        } for i in sim_scores]
        return recommendations

    def record_user_feedback(self, user_id, movie_id, rating, watch_time, click_through, alpha_value):
        # Implement feedback recording logic here
        pass

# Initialize evaluator
evaluator = RecommenderEvaluator(filtered_df)

# Streamlit UI
st.title("Movie Recommendation System")

# Movie selection
st.header("Get Recommendations")
selected_movie = st.selectbox("Select a movie:", filtered_df['original_title'])
n_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):
    hybrid_similarity = (0.5 * tfidf_similarity + 0.5 * embedding_similarity)  # Example hybrid
    recommendations = evaluator.get_recommendations_with_explanation(
        title=selected_movie,
        hybrid_similarity=hybrid_similarity,
        n_recommendations=n_recommendations
    )
    st.write("Recommendations:")
    for rec in recommendations:
        st.write(f"**{rec['title']}** - Similarity Score: {rec['similarity_score']:.2f}")
        st.write(f"Explanation: {rec['explanation']}")

# Feedback section
st.header("Provide Feedback")
user_id = st.text_input("User ID:")
movie_id = st.text_input("Movie ID (from recommendations):")
rating = st.slider("Rating (1-5):", 1.0, 5.0, 3.0)
watch_time = st.number_input("Watch Time (minutes):", min_value=0, value=0)
click_through = st.checkbox("Did you click through?")
alpha_value = st.slider("Alpha Value (if applicable):", 0.0, 1.0, 0.5)

if st.button("Submit Feedback"):
    evaluator.record_user_feedback(
        user_id=user_id,
        movie_id=movie_id,
        rating=rating,
        watch_time=watch_time,
        click_through=click_through,
        alpha_value=alpha_value
    )
    st.success("Feedback submitted successfully!")
