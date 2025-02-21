import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# Load ratings data
ratings_file = "Synthetic_Movie_Lens/ratings.csv"
movies_file = "Synthetic_Movie_Lens/movies.csv"

ratings_df = pd.read_csv(ratings_file)
movies_df = pd.read_csv(movies_file)

# Create the ratings matrix (Users × Movies)
ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Perform NMF Decomposition (k = 20)
k = 20  
nmf_model = NMF(n_components=k, random_state=42)
U = nmf_model.fit_transform(ratings_matrix)  # User-latent factors
V = nmf_model.components_.T  # Movie-latent factors

# Map movie IDs to their indices in V
movie_ids = ratings_matrix.columns.to_numpy()
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Merge movie metadata (genres)
movies_df['genres'] = movies_df['genres'].str.split('|')  # Convert genres to list format
movie_id_to_genres = movies_df.set_index('movieId')['genres'].to_dict()

# Analyze each latent factor
top_movies_per_factor = {}

for factor in range(k):
    print(f"\nLatent Factor {factor+1}:")
    
    # Sort movies by their values in the latent factor column
    top_movie_indices = np.argsort(-V[:, factor])[:10]  # Get top 10 movie indices
    top_movie_ids = movie_ids[top_movie_indices]  # Convert indices back to movie IDs
    
    # Retrieve movie genres
    genre_counts = {}
    for movie_id in top_movie_ids:
        genres = movie_id_to_genres.get(movie_id, ["Unknown"])
        print(f"  Movie ID: {movie_id}, Genres: {', '.join(genres)}")
        
        # Count genre occurrences
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Identify dominant genres
    sorted_genres = sorted(genre_counts.items(), key=lambda x: -x[1])
    print(f"  Dominant Genres: {', '.join([f'{g} ({c})' for g, c in sorted_genres])}")

    # Store results
    top_movies_per_factor[factor] = {"movies": top_movie_ids, "genres": genre_counts}