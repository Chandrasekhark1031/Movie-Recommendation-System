from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
movies_db = pd.read_csv("HTML Page Layout/IMDB_Dataset.csv")  # Ensure this file exists
movies_db.fillna("", inplace=True)  # Fill any missing values

# Combine relevant features (like genres, director, cast) for better recommendations
movies_db["combined_features"] = (
    movies_db["Genre"].astype(str) + " " +
    movies_db["Director"].astype(str) + " " +
    movies_db["Star Cast"].astype(str)  
)

# Convert text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(movies_db["combined_features"])

# Function to get movie recommendations based on similarity
def get_recommendations(movie_name):
    movie_name = movie_name.lower()
    
    if movie_name not in movies_db["Movie Name"].str.lower().values:
        return ["Sorry, no recommendations available for this movie."]
    
    # Find the index of the movie in the dataset
    movie_index = movies_db[movies_db["Movie Name"].str.lower() == movie_name].index[0]
    
    # Compute similarity scores with all movies
    similarity_scores = cosine_similarity(feature_matrix[movie_index], feature_matrix).flatten()
    
    # Get the indices of the most similar movies (excluding the movie itself)
    similar_movie_indices = similarity_scores.argsort()[-6:-1][::-1]  # Top 5 similar movies

    return movies_db["Movie Name"].iloc[similar_movie_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form.get('movie_name', '').strip()
        if movie_name:
            recommendations = get_recommendations(movie_name)
    
    return render_template('index1.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
