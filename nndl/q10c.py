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

def plot_rmse_subsets(subset_results):
    """
    Plots RMSE vs. k for different dataset subsets.

    Parameters:
    - subset_results: Dictionary containing RMSE results for each subset.
    """
    plt.figure(figsize=(10, 6))

    # Plot RMSE for each subset
    for subset_name, df in subset_results.items():
        plt.plot(df['k'], df['RMSE'], marker='o', linestyle='-', label=subset_name.capitalize())

    # Formatting the plot
    plt.xlabel("Number of Latent Factors (k)")
    plt.ylabel("Average RMSE")
    plt.title("RMSE vs. Number of Latent Factors (k) for Different Subsets")
    plt.legend()
    plt.grid()
    plt.show()
    
def evaluate_mf(data, k_values):
    """
    Evaluates the performance of a Matrix Factorization (MF) collaborative filter using SVD.
    Computes RMSE and MAE for different values of latent factors (k) using 10-fold cross-validation.

    Parameters:
    - data: Surprise Dataset object containing user-movie ratings.
    - k_values: List of values for the number of latent factors (k).

    Returns:
    - results_df: DataFrame containing k, average RMSE, and average MAE.
    """
    results = []
    kf = KFold(n_splits=10)  # 10-fold cross-validation
    
    for k in k_values:
        algo = SVD(n_factors=k, random_state=42)  # SVD-based MF model
        rmse_scores, mae_scores = [], []
        
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            
            # Extract true and predicted ratings
            y_true = [pred.r_ui for pred in predictions]
            y_pred = [pred.est for pred in predictions]
            
            # Compute RMSE and MAE
            rmse_scores.append(mean_squared_error(y_true, y_pred, squared=False))
            mae_scores.append(mean_absolute_error(y_true, y_pred))
        
        # Compute average RMSE and MAE across all folds
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)
        results.append((k, avg_rmse, avg_mae))
    
    results_df = pd.DataFrame(results, columns=['k', 'RMSE', 'MAE'])

    return results_df

def plot_error(results, metric):
    """
    Plots RMSE or MAE against the number of latent factors (k).

    Parameters:
    - results: DataFrame containing k, RMSE, and MAE values.
    - metric: String, either "RMSE" or "MAE", to indicate which metric to plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(results['k'], results[metric], marker='o', linestyle='-', color='b' if metric == "RMSE" else 'r')
    plt.xlabel("Number of Latent Factors (k)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Number of Latent Factors (k)")
    plt.grid()
    plt.show()

def filter_subsets(rating_df, movies_df, popularity_threshold=50, variance_threshold=1.5):
    """
    Splits the dataset into three subsets: popular, unpopular, and high-variance movies.

    Parameters:
    - rating_df: DataFrame containing user ratings (columns: userId, movieId, rating).
    - movies_df: DataFrame containing movie metadata (columns: movieId, title, genres).
    - popularity_threshold: Minimum number of ratings for a movie to be considered "popular".
    - variance_threshold: Minimum rating variance for a movie to be considered "high-variance".

    Returns:
    - subsets: Dictionary containing three DataFrames (popular, unpopular, high_variance).
    """

    # Compute the number of ratings and rating variance for each movie
    movie_stats = rating_df.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        rating_variance=('rating', 'var')
    ).fillna(0)  # Fill NaN variance (for movies with only one rating) with 0

    # Identify popular movies (movies with more than `popularity_threshold` ratings)
    popular_movies = movie_stats[movie_stats['num_ratings'] >= popularity_threshold].index

    # Identify unpopular movies (movies with very few ratings)
    unpopular_movies = movie_stats[movie_stats['num_ratings'] < popularity_threshold].index

    # Identify high-variance movies (movies where rating variance is high)
    high_variance_movies = movie_stats[movie_stats['rating_variance'] >= variance_threshold].index

    # Filter rating data based on these subsets
    popular_df = rating_df[rating_df['movieId'].isin(popular_movies)]
    unpopular_df = rating_df[rating_df['movieId'].isin(unpopular_movies)]
    high_variance_df = rating_df[rating_df['movieId'].isin(high_variance_movies)]

    # Return the subsets as a dictionary
    subsets = {
        "popular": popular_df,
        "unpopular": unpopular_df,
        "high_variance": high_variance_df
    }

    return subsets

def find_optimal_k(results):
    """
    Find the optimal number of latent factors (k) that minimizes RMSE and MAE.

    Parameters:
    - results: DataFrame containing k, RMSE, and MAE values.

    Returns:
    - best_k_rmse: The value of k that minimizes RMSE.
    - min_rmse: The minimum RMSE value.
    - best_k_mae: The value of k that minimizes MAE.
    - min_mae: The minimum MAE value.
    """
    best_k_rmse = results.loc[results['RMSE'].idxmin(), 'k']
    min_rmse = results['RMSE'].min()

    best_k_mae = results.loc[results['MAE'].idxmin(), 'k']
    min_mae = results['MAE'].min()

    print(f"Optimal k (Min RMSE): {best_k_rmse}, Minimum RMSE: {min_rmse:.4f}")
    print(f"Optimal k (Min MAE): {best_k_mae}, Minimum MAE: {min_mae:.4f}")

    return best_k_rmse, min_rmse, best_k_mae, min_mae

def evaluate_mf_on_subsets(subsets, reader, k_values):
    """
    Evaluate MF model on different dataset subsets and return RMSE results.

    Parameters:
    - subsets: Dictionary containing different movie subsets.
    - reader: Surprise Reader object.
    - k_values: List of k values to evaluate.

    Returns:
    - subset_results: Dictionary storing RMSE results as DataFrames for each subset.
    """
    subset_results = {}

    for subset_name, subset_df in subsets.items():
        print(f"\nEvaluating MF model on {subset_name} movie subset...")

        data = Dataset.load_from_df(subset_df[['userId', 'movieId', 'rating']], reader)
        results = []

        for k in k_values:
            algo = SVD(n_factors=k, random_state=42)
            cv_results = cross_validate(algo, data, measures=['RMSE'], cv=10, verbose=False)
            avg_rmse = np.mean(cv_results['test_rmse'])

            results.append((k, avg_rmse))

        df_results = pd.DataFrame(results, columns=['k', 'RMSE'])
        subset_results[subset_name] = df_results

    return subset_results

def compute_roc_auc(predictions, threshold):
    """
    Compute the ROC curve and AUC value.

    Parameters:
    - predictions: List of Surprise predictions.
    - threshold: Rating threshold for binary classification.

    Returns:
    - fpr: False Positive Rates.
    - tpr: True Positive Rates.
    - auc_score: AUC value.
    """
    y_true = [1 if pred.r_ui >= threshold else 0 for pred in predictions]
    y_scores = [pred.est for pred in predictions]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score

def plot_roc_thresholds(algo, data, best_k, mode, thresholds=[2.5, 3, 3.5, 4]):
    """
    Plot the ROC curves for different rating thresholds.

    Parameters:
    - algo: Trained MF model.
    - data: Surprise Dataset object.
    - best_k: Optimal number of latent factors.
    - mode: Dataset subset name.
    - thresholds: List of relevance thresholds.
    """
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=best_k, random_state=42)
    algo.fit(trainset)

    predictions = algo.test(testset)

    plt.figure(figsize=(8, 6))
    
    for threshold in thresholds:
        fpr, tpr, auc_score = compute_roc_auc(predictions, threshold)
        plt.plot(fpr, tpr, label=f"Threshold {threshold} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for MF ({mode.capitalize()} Movies)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load dataset
    ratings_file = "Synthetic_Movie_Lens/ratings.csv"
    movies_file = "Synthetic_Movie_Lens/movies.csv"
    
    rating_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)

    # Filter dataset into subsets
    subsets = filter_subsets(rating_df, movies_df)

    # Prepare Surprise reader
    reader = Reader(rating_scale=(0, 5))

    # Define k values (range 2 to 50, step size 2)
    k_values = range(2, 52, 2)

    # Evaluate MF model on subsets
    subset_results = evaluate_mf_on_subsets(subsets, reader, k_values)

    # Find the best k for each subset
    best_k_values = {}
    for subset_name, df in subset_results.items():
        best_k = df.loc[df['RMSE'].idxmin(), 'k']
        best_rmse = df['RMSE'].min()
        best_k_values[subset_name] = best_k
        print(f"Optimal k for {subset_name} dataset: {best_k}, Minimum RMSE: {best_rmse:.4f}")

    # Plot RMSE vs. k for all subsets
    plot_rmse_subsets(subset_results)

    # Compute and plot ROC curves for each subset
    for subset_name, subset_df in subsets.items():
        print(f"\nGenerating ROC curve for {subset_name} dataset...")
        data = Dataset.load_from_df(subset_df[['userId', 'movieId', 'rating']], reader)
        plot_roc_thresholds(SVD, data, best_k_values[subset_name], subset_name)