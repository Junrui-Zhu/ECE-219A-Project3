import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, NMF, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def train_and_get_predictions(algo, data):
    """
    Train a model and get test set predictions.

    Parameters:
    - algo: Surprise recommendation algorithm (KNN, NMF, or SVD).
    - data: Surprise Dataset object.

    Returns:
    - predictions: List of Surprise Prediction objects.
    """
    trainset, testset = train_test_split(data, test_size=0.2)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return predictions

def compute_roc_curve(predictions, threshold=3):
    """Compute ROC curve and AUC for a model's predictions."""
    y_true = [1 if pred.r_ui >= threshold else 0 for pred in predictions]
    y_scores = [pred.est for pred in predictions]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def plot_roc_comparison(predictions_knn, predictions_nmf, predictions_mf, threshold=3):
    """Plot ROC curves for k-NN, NMF, and MF models in a single figure."""
    fpr_knn, tpr_knn, auc_knn = compute_roc_curve(predictions_knn, threshold)
    fpr_nmf, tpr_nmf, auc_nmf = compute_roc_curve(predictions_nmf, threshold)
    fpr_mf, tpr_mf, auc_mf = compute_roc_curve(predictions_mf, threshold)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_knn, tpr_knn, label=f'k-NN (AUC = {auc_knn:.4f})', linestyle='--', color='blue')
    plt.plot(fpr_nmf, tpr_nmf, label=f'NMF (AUC = {auc_nmf:.4f})', linestyle='-', color='red')
    plt.plot(fpr_mf, tpr_mf, label=f'MF (AUC = {auc_mf:.4f})', linestyle='-.', color='green')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison of k-NN, NMF, and MF (Threshold = 3)")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    BEST_K_KNN = 50  # optimal k for k-NN
    BEST_K_NMF = 20  # optimal k for NMF
    BEST_K_MF = 20   # optimal k for MF (SVD)
    # Load MovieLens dataset from CSV file
    file_path = "Synthetic_Movie_Lens/ratings.csv"
    rating_df = pd.read_csv(file_path)

    # Prepare dataset for Surprise library
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(rating_df[['userId', 'movieId', 'rating']], reader)

    print(f"Training k-NN model with k={BEST_K_KNN}...")
    knn_algo = KNNBasic(k=BEST_K_KNN, sim_options={'name': 'cosine', 'user_based': False})
    predictions_knn = train_and_get_predictions(knn_algo, data)

    print(f"Training NMF model with n_factors={BEST_K_NMF}...")
    nmf_algo = NMF(n_factors=BEST_K_NMF)
    predictions_nmf = train_and_get_predictions(nmf_algo, data)

    print(f"Training MF (SVD) model with n_factors={BEST_K_MF}...")
    mf_algo = SVD(n_factors=BEST_K_MF)
    predictions_mf = train_and_get_predictions(mf_algo, data)

    print("Generating ROC curve comparison...")
    plot_roc_comparison(predictions_knn, predictions_nmf, predictions_mf)


def compute_roc_curve(predictions, threshold=3):
    """
    Compute the ROC curve and AUC for a given model's predictions.

    Parameters:
    - predictions: List of Surprise prediction objects.
    - threshold: Rating threshold to define positive vs. negative labels.

    Returns:
    - fpr: False Positive Rate
    - tpr: True Positive Rate
    - auc_score: Area Under Curve (AUC)
    """
    # Convert ratings to binary (1 if rating >= threshold, else 0)
    y_true = [1 if pred.r_ui >= threshold else 0 for pred in predictions]
    y_scores = [pred.est for pred in predictions]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score

# plot_roc_comparison(predictions_knn, predictions_nmf, predictions_mf)

if __name__ == "__main__":
    main()
