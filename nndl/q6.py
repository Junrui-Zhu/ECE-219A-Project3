import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
from surprise.model_selection import train_test_split  
from scipy.sparse import coo_matrix


def load_rating_data(data_path='Synthetic_Movie_Lens/ratings.csv'):
    rating_df = pd.read_csv(data_path)
    return rating_df

def construct_ratings_matrix(rating_df):
    movie_Id_2_Order = {}
    movie_Order_2_Id = []
    Order = 0
    movie_orders = []
    for movieId in rating_df['movieId']:
        if movieId not in movie_Id_2_Order:
            movie_Order_2_Id.append(movieId)
            movie_Id_2_Order[movieId] = Order
            Order += 1
        movie_orders.append(movie_Id_2_Order[movieId])
    movie_orders = np.array(movie_orders)

    ratings_matrix = coo_matrix((rating_df['rating'].to_numpy(), 
                               (rating_df['userId'].to_numpy() - 1, movie_orders)))

    return ratings_matrix, movie_Id_2_Order, movie_Order_2_Id

# ==============================
# data selecting
# ==============================
def trim_dataset(df, ratings_matrix, mode="popular"):
    movie_counts = np.array((ratings_matrix > 0).sum(axis=0)).flatten()
    
    if mode == "popular":
        trimmed_movies = np.where(movie_counts > 2)[0] 
    elif mode == "unpopular":
        trimmed_movies = np.where(movie_counts <= 2)[0]  
    elif mode == "high_variance":
        ratings_matrix = ratings_matrix.tocsc()
        variances = np.zeros(ratings_matrix.shape[1])
        for j in range(ratings_matrix.shape[1]):
            movie_j_ratings = ratings_matrix[:, j].toarray()[:, 0]
            if len(movie_j_ratings) < 5:  
                continue
            variances[j] = np.var(movie_j_ratings)
        trimmed_movies = np.where(variances >= 2)[0]

    return df[df['movieId'].isin(trimmed_movies)]


def train_knn(data, k):
    trainset = data
    sim_options = {'name': 'pearson', 'user_based': True}
    algo = KNNBasic(k=k, sim_options=sim_options)
    algo.fit(trainset)
    return algo


def evaluate_knn(data, k_values):
    results = []
    kf = KFold(n_splits=10)  
    
    for k in k_values:
        algo = KNNBasic(k=k, sim_options={'name': 'pearson', 'user_based': True})
        rmse_scores = []
        
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            y_true = [pred.r_ui for pred in predictions]
            y_pred = [pred.est for pred in predictions]  
            rmse_scores.append(mean_squared_error(y_true, y_pred, squared=False)) 
        
        avg_rmse = np.mean(rmse_scores)  
        results.append((k, avg_rmse))
    
    results_df = pd.DataFrame(results, columns=['k', 'RMSE'])
    
    best_k = results_df.loc[results_df['RMSE'].idxmin(), 'k']
    min_rmse = results_df['RMSE'].min()
    print(f"best k = {best_k}, minimal RMSE = {min_rmse:.4f}")
    
    return results_df, best_k


def plot_rmse(results, mode):
    plt.figure(figsize=(8, 5))
    plt.plot(results['k'], results['RMSE'], marker='o', linestyle='-', color='b', label='RMSE')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs. k ({mode} Movies)")
    plt.legend()
    plt.grid()
    plt.show()


def compute_roc(predictions, threshold=3):
    y_true = [1 if pred.r_ui >= threshold else 0 for pred in predictions] 
    y_score = [pred.est for pred in predictions]  
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc(algo, data, k, mode, thresholds=[2.5, 3, 3.5, 4]):
    trainset, testset = train_test_split(data, test_size=0.1)
    algo = train_knn(trainset, k) 
    predictions = algo.test(testset) 

    plt.figure(figsize=(8, 6))
    for threshold in thresholds:
        fpr, tpr, roc_auc = compute_roc(predictions, threshold)
        plt.plot(fpr, tpr, label=f'Threshold {threshold} (AUC = {roc_auc:.4f})')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for k-NN ({mode} Movies)")
    plt.legend()
    plt.grid()
    plt.show()

# ==============================
# main
# ==============================
if __name__ == "__main__":
    file_path = "Synthetic_Movie_Lens/ratings.csv"
    rating_df = load_rating_data(file_path)
    ratings_matrix, _, _ = construct_ratings_matrix(rating_df)

    k_values = range(2, 102, 2) 

    for mode in ["popular", "unpopular", "high_variance"]:

        trimmed_df = trim_dataset(rating_df, ratings_matrix, mode=mode)
        trimmed_data = Dataset.load_from_df(trimmed_df[['userId', 'movieId', 'rating']], Reader(rating_scale=(0, 5)))

        # **compute RMSE and plot RMSE vs. k**
        results, best_k = evaluate_knn(trimmed_data, k_values)
        plot_rmse(results, mode)
        
        # **plot ROC**
        print(f"best k = {best_k}")
        algo = train_knn(trimmed_data, best_k)
        plot_roc(algo, trimmed_data, best_k, mode)