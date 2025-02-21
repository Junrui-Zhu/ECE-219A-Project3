import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, NMF
from surprise.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, auc
from surprise.model_selection import train_test_split

def evaluate_nmf(data, k_values):
    """
    Evaluates the performance of an NMF-based collaborative filtering model.
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
        algo = NMF(n_factors=k, random_state=42)  # NMF-based collaborative filtering model
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
    plt.title(f"{metric} vs. Number of Latent Factors (k) (NMF Collaborative Filtering)")
    plt.grid()
    plt.show()

def find_optimal_k(results):
    """
    Finds the optimal number of latent factors (k) based on minimum RMSE and MAE.

    Parameters:
    - results: DataFrame containing k, RMSE, and MAE values.

    Returns:
    - A dictionary containing optimal k values for RMSE and MAE.
    """
    best_k_rmse = results.loc[results['RMSE'].idxmin(), 'k']
    best_rmse = results['RMSE'].min()
    
    best_k_mae = results.loc[results['MAE'].idxmin(), 'k']
    best_mae = results['MAE'].min()

    print(f"Optimal k (min RMSE) = {best_k_rmse}, Minimum RMSE = {best_rmse:.4f}")
    print(f"Optimal k (min MAE) = {best_k_mae}, Minimum MAE = {best_mae:.4f}")

    return {"best_k_rmse": best_k_rmse, "best_rmse": best_rmse, 
            "best_k_mae": best_k_mae, "best_mae": best_mae}

def filter_subsets(rating_df, movies_df):
    """
    Filters the dataset into three subsets: Popular, Unpopular, and High-Variance movies.

    Parameters:
    - rating_df: DataFrame containing user-movie ratings.
    - movies_df: DataFrame containing movie information.

    Returns:
    - Dictionary containing three subsets: 'popular', 'unpopular', and 'high_variance'.
    """
    # Calculate number of ratings per movie
    movie_counts = rating_df.groupby('movieId').size()
    
    # Define thresholds for popularity
    popular_threshold = movie_counts.quantile(0.75)  # Top 25% most rated movies
    unpopular_threshold = movie_counts.quantile(0.25)  # Bottom 25% least rated movies

    # Identify popular and unpopular movies
    popular_movies = movie_counts[movie_counts >= popular_threshold].index
    unpopular_movies = movie_counts[movie_counts <= unpopular_threshold].index

    # Calculate rating variance per movie
    rating_variance = rating_df.groupby('movieId')['rating'].var()
    high_variance_threshold = rating_variance.quantile(0.75)  # Top 25% highest variance movies
    high_variance_movies = rating_variance[rating_variance >= high_variance_threshold].index

    # Filter subsets
    subsets = {
        "popular": rating_df[rating_df['movieId'].isin(popular_movies)],
        "unpopular": rating_df[rating_df['movieId'].isin(unpopular_movies)],
        "high_variance": rating_df[rating_df['movieId'].isin(high_variance_movies)]
    }
    
    return subsets

def evaluate_nmf_on_subsets(subsets, reader, k_values):
    """
    Evaluates the NMF filter on different dataset subsets using 10-fold cross-validation.

    Parameters:
    - subsets: Dictionary containing filtered datasets.
    - reader: Surprise Reader object.
    - k_values: List of values for the number of latent factors (k).

    Returns:
    - Dictionary containing RMSE results for each subset.
    """
    results = {}

    for subset_name, subset_df in subsets.items():
        print(f"\nEvaluating NMF on {subset_name} movies...")
        
        # Load dataset for Surprise
        data = Dataset.load_from_df(subset_df[['userId', 'movieId', 'rating']], reader)
        
        # Perform evaluation
        results[subset_name] = evaluate_nmf(data, k_values)

    return results

def compute_roc_auc(algo, testset):
    """
    Computes ROC curve and AUC for the NMF collaborative filter.

    Parameters:
    - algo: Trained Surprise NMF model.
    - testset: Test dataset from Surprise.

    Returns:
    - fpr: False positive rates.
    - tpr: True positive rates.
    - auc_score: Area under the curve.
    """
    predictions = algo.test(testset)

    # Convert ratings into binary labels (1 = relevant, 0 = not relevant)
    y_true = [1 if pred.r_ui >= 4.0 else 0 for pred in predictions]  # Ground truth
    y_scores = [pred.est for pred in predictions]  # Predicted scores

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score

def compute_roc_auc(predictions, threshold):
    """
    Computes ROC curve and AUC for the NMF collaborative filter at a given threshold.

    Parameters:
    - predictions: List of Surprise predictions.
    - threshold: Rating threshold for relevance classification.

    Returns:
    - fpr: False Positive Rates.
    - tpr: True Positive Rates.
    - auc_score: Area Under the Curve (AUC).
    """
    y_true = [1 if pred.r_ui >= threshold else 0 for pred in predictions]  # Ground truth (binary)
    y_scores = [pred.est for pred in predictions]  # Predicted scores

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score

def plot_roc_thresholds(algo, data, best_k, mode, thresholds=[2.5, 3, 3.5, 4]):
    """
    Plots ROC curves for different relevance thresholds.

    Parameters:
    - algo: Trained NMF model.
    - data: Surprise Dataset object.
    - best_k: Optimal number of latent factors.
    - mode: Dataset subset name (popular, unpopular, high_variance).
    - thresholds: List of relevance thresholds.
    """
    # Split dataset
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = NMF(n_factors=best_k, random_state=42)
    algo.fit(trainset)

    # Get predictions
    predictions = algo.test(testset)

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    
    for threshold in thresholds:
        fpr, tpr, auc_score = compute_roc_auc(predictions, threshold)
        plt.plot(fpr, tpr, label=f"Threshold {threshold} (AUC = {auc_score:.4f})")

    # Random classifier reference line
    plt.plot([0, 1], [0, 1], 'k--')

    # Labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for NMF ({mode.capitalize()} Movies)")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_auc_on_subsets(subsets, reader, best_k):
    """
    Computes and plots ROC curves for the NMF collaborative filter on different dataset subsets.

    Parameters:
    - subsets: Dictionary containing filtered datasets.
    - reader: Surprise Reader object.
    - best_k: Dictionary with the best k values for each subset.

    Returns:
    - Prints AUC scores and displays ROC curves.
    """
    plt.figure(figsize=(8, 6))

    for subset_name, subset_df in subsets.items():
        print(f"\nComputing ROC for {subset_name} movies...")

        # Load dataset for Surprise
        data = Dataset.load_from_df(subset_df[['userId', 'movieId', 'rating']], reader)

        # ✅ Correctly split the dataset using Surprise’s `train_test_split`
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        # Train NMF model with optimal k
        algo = NMF(n_factors=best_k[subset_name], random_state=42)
        algo.fit(trainset)

        # Compute ROC and AUC
        fpr, tpr, auc_score = compute_roc_auc(algo, testset)
        print(f"AUC for {subset_name} movies: {auc_score:.4f}")

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{subset_name.capitalize()} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for NMF Collaborative Filtering")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rmse_subsets(results):
    """
    Plots RMSE vs. k for each dataset subset.

    Parameters:
    - results: Dictionary containing RMSE results for each subset.
    """
    plt.figure(figsize=(8, 5))
    
    for subset_name, df in results.items():
        plt.plot(df['k'], df['RMSE'], marker='o', linestyle='-', label=f"{subset_name.capitalize()} Movies")

    plt.xlabel("Number of Latent Factors (k)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Number of Latent Factors (k) for Different Subsets")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load MovieLens dataset
    ratings_file = "Synthetic_Movie_Lens/ratings.csv"
    movies_file = "Synthetic_Movie_Lens/movies.csv"
    
    rating_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)

    # Filter dataset into subsets
    subsets = filter_subsets(rating_df, movies_df)

    # Prepare Surprise reader
    reader = Reader(rating_scale=(0, 5))

    # Define k values (2 to 50, step size of 2)
    k_values = range(2, 52, 2)

    # Evaluate NMF on each subset
    subset_results = evaluate_nmf_on_subsets(subsets, reader, k_values)

    # Find and report the minimum RMSE for each subset
    best_k_values = {}
    for subset_name, df in subset_results.items():
        best_k = df.loc[df['RMSE'].idxmin(), 'k']
        best_rmse = df['RMSE'].min()
        best_k_values[subset_name] = best_k  # Store best k for AUC evaluation
        print(f"Optimal k for {subset_name} movies: {best_k}, Minimum RMSE: {best_rmse:.4f}")

    # Plot RMSE vs. k for all subsets
    plot_rmse_subsets(subset_results)

    # Compute and plot ROC curves for each subset with multiple thresholds
    for subset_name, subset_df in subsets.items():
        print(f"\nGenerating ROC curves for {subset_name} movies...")
        data = Dataset.load_from_df(subset_df[['userId', 'movieId', 'rating']], reader)
        plot_roc_thresholds(NMF, data, best_k_values[subset_name], subset_name)