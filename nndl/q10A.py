import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

if __name__ == "__main__":
    # Load MovieLens dataset from CSV file
    file_path = "Synthetic_Movie_Lens/ratings.csv"
    rating_df = pd.read_csv(file_path)

    # Prepare dataset for Surprise library
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(rating_df[['userId', 'movieId', 'rating']], reader)

    # Define range of latent factors (k) from 2 to 50 with step size of 2
    k_values = range(2, 52, 2)

    # Evaluate MF model for different values of k
    results = evaluate_mf(data, k_values)

    # Find the optimal k values for RMSE and MAE
    best_k_rmse = results.loc[results['RMSE'].idxmin(), 'k']
    best_rmse = results['RMSE'].min()
    
    best_k_mae = results.loc[results['MAE'].idxmin(), 'k']
    best_mae = results['MAE'].min()

    # Print optimal k values and minimum errors
    print(f"Optimal k (min RMSE) = {best_k_rmse}, Minimum RMSE = {best_rmse:.4f}")
    print(f"Optimal k (min MAE) = {best_k_mae}, Minimum MAE = {best_mae:.4f}")

    # Plot RMSE vs. k
    plot_error(results, "RMSE")

    # Plot MAE vs. k
    plot_error(results, "MAE")