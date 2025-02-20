"""
Answer for question 11
"""


from scipy.sparse import coo_matrix
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from proj3_q1 import *


class NaiveCollaborativeFilter:
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix.tocsr()
        self.user_means = None

    def fit(self, train_indices):
        train_matrix = self.rating_matrix[train_indices, :].toarray()
        mask = train_matrix != 0
        sum_ratings = train_matrix.sum(axis=1)
        count_ratings = mask.sum(axis=1)
        self.user_means = np.divide(
            sum_ratings, count_ratings, out=np.zeros_like(sum_ratings, dtype=float), where=count_ratings != 0
        )

    def predict(self, test_indices):
        predictions, actuals = [], []
        for idx, user_idx in enumerate(test_indices):
            user_mean = self.user_means[idx]
            user_ratings = self.rating_matrix[user_idx, :].toarray().flatten()
            mask = user_ratings != 0
            predictions.extend([user_mean] * np.sum(mask))
            actuals.extend(user_ratings[mask])
        return np.array(actuals), np.array(predictions)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def evaluate(self, n_splits=10):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmses = []
        for train_index, test_index in kf.split(self.rating_matrix):
            self.fit(train_index)
            y_true, y_pred = self.predict(test_index)
            rmses.append(self.rmse(y_true, y_pred))
        return np.mean(rmses)

def subset_popular_movies(rating_matrix):
    item_counts = np.diff(rating_matrix.tocsc().indptr)
    selected_items = np.where(item_counts > 2)[0]
    return rating_matrix[:, selected_items]

def subset_unpopular_movies(rating_matrix):
    item_counts = np.diff(rating_matrix.tocsc().indptr)
    selected_items = np.where(item_counts <= 2)[0]
    return rating_matrix[:, selected_items]

def subset_high_variance_movies(rating_matrix):
    matrix_dense = rating_matrix.toarray()
    mask = matrix_dense != 0
    item_variances = np.nanvar(np.where(mask, matrix_dense, np.nan), axis=0, ddof=0)
    item_counts = mask.sum(axis=0)
    selected_items = np.where((item_variances >= 2) & (item_counts >= 5))[0]
    return rating_matrix[:, selected_items]

if __name__ == "__main__":
    rating_df = load_rating_data()
    rating_matrix, _, _ = construct_ratings_matrix(rating_df)
    rating_matrix = rating_matrix.tocsr()
    model = NaiveCollaborativeFilter(rating_matrix)
    avg_rmse = model.evaluate()
    print(f"RMSE of the oringinal dataset: {avg_rmse:.4f}")
    popular_subset = subset_popular_movies(rating_matrix)
    popular_model = NaiveCollaborativeFilter(popular_subset)
    avg_rmse_popular = popular_model.evaluate()
    print(f"RMSE of popular dataset: {avg_rmse_popular:.4f}")
    unpopular_subset = subset_unpopular_movies(rating_matrix)
    unpopular_model = NaiveCollaborativeFilter(unpopular_subset)
    avg_rmse_unpopular = unpopular_model.evaluate()
    print(f"RMSE of umpopular dataset: {avg_rmse_unpopular:.4f}")
    high_variance_subset = subset_high_variance_movies(rating_matrix)
    high_variance_model = NaiveCollaborativeFilter(high_variance_subset)
    avg_rmse_high_variance = high_variance_model.evaluate()
    print(f"RMSE of high variance dataset: {avg_rmse_high_variance:.4f}")

