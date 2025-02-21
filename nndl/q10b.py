import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import SVD
from surprise.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

def count_unique_genres(movies_df):
    """
    Counts the number of unique genres in the dataset.

    Parameters:
    - movies_df: DataFrame containing movie metadata (movieId, title, genres).

    Returns:
    - num_genres: The number of unique genres in the dataset.
    """
    # Split genres and count unique ones
    unique_genres = set()
    for genre_list in movies_df['genres'].str.split('|'):
        unique_genres.update(genre_list)

    num_genres = len(unique_genres)
    print(f"Number of unique movie genres: {num_genres}")
    return num_genres

def find_optimal_k(results_df):
    """
    Identifies the optimal number of latent factors (k) that minimizes RMSE and MAE.

    Parameters:
    - results_df: DataFrame containing k, RMSE, and MAE values.

    Returns:
    - best_k_rmse: The k value that minimizes RMSE.
    - min_rmse: The minimum RMSE value.
    - best_k_mae: The k value that minimizes MAE.
    - min_mae: The minimum MAE value.
    """
    # Find k that minimizes RMSE
    best_k_rmse = results_df.loc[results_df['RMSE'].idxmin(), 'k']
    min_rmse = results_df['RMSE'].min()

    # Find k that minimizes MAE
    best_k_mae = results_df.loc[results_df['MAE'].idxmin(), 'k']
    min_mae = results_df['MAE'].min()

    print(f"Optimal k (Min RMSE): {best_k_rmse}, Minimum RMSE: {min_rmse:.4f}")
    print(f"Optimal k (Min MAE): {best_k_mae}, Minimum MAE: {min_mae:.4f}")

    return best_k_rmse, min_rmse, best_k_mae, min_mae

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np

def compute_rmse_mae(k_values, ratings_df):
    """
    Computes RMSE and MAE for different values of k using 10-fold cross-validation.

    Parameters:
    - k_values: List of latent factor values to test.
    - ratings_df: DataFrame containing userId, movieId, and rating.

    Returns:
    - results_df: DataFrame containing RMSE and MAE for each k.
    """
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    results = []

    for k in k_values:
        print(f"Evaluating k = {k}...")
        algo = SVD(n_factors=k, random_state=42)
        cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=False)

        avg_rmse = np.mean(cv_results['test_rmse'])
        avg_mae = np.mean(cv_results['test_mae'])

        results.append((k, avg_rmse, avg_mae))

    results_df = pd.DataFrame(results, columns=['k', 'RMSE', 'MAE'])
    return results_df



if __name__ == "__main__":
    # Load dataset
    ratings_file = "Synthetic_Movie_Lens/ratings.csv"
    movies_file = "Synthetic_Movie_Lens/movies.csv"
    # Load the dataset
    ratings_df = pd.read_csv("Synthetic_Movie_Lens/ratings.csv")

    # Define k values to test (from 2 to 50, step size = 2)
    k_values = range(2, 52, 2)

    # Compute RMSE and MAE for different k values
    results_df = compute_rmse_mae(k_values, ratings_df)

    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)

    # Find optimal k values
    best_k_rmse, min_rmse, best_k_mae, min_mae = find_optimal_k(results_df)

    # Count number of unique genres
    num_genres = count_unique_genres(movies_df)

    # Compare optimal k with the number of genres
    if best_k_rmse == num_genres or best_k_mae == num_genres:
        print(f"The optimal number of latent factors ({best_k_rmse} or {best_k_mae}) matches the number of movie genres ({num_genres}).")
    else:
        print(f"The optimal number of latent factors ({best_k_rmse} or {best_k_mae}) is different from the number of movie genres ({num_genres}).")