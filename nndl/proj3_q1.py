import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                               (rating_df['userId'].to_numpy() - 1, movie_orders) # rating_df['movieId'].to_numpy() - 1
                               ))

    return ratings_matrix, movie_Id_2_Order, movie_Order_2_Id

if __name__ == "__main__":

    rating_df = load_rating_data()
    ratings_matrix, _, _ = construct_ratings_matrix(rating_df)

    # Answer of 1.A
    (num_users, num_movies) = ratings_matrix.shape
    print(f"num of users: {num_users}")
    print(f"num of movies: {num_movies}")
    print(f"num of available ratings: {rating_df.shape[0]}")
    print(f"num of possible ratings: {num_users * num_movies}")
    sparsity = rating_df.shape[0] / (num_users * num_movies)
    print(f"sparsity: {sparsity}")

    # Answer of 1.B
    rating_bins = np.arange(0, 5.5, 0.5)
    counts, bin_edges = np.histogram(rating_df['rating'], bins=rating_bins)
    plt.bar(bin_edges[:-1], counts, width=0.5, edgecolor='black', align='edge')
    plt.xlabel("Binned Rating Values")
    plt.ylabel("Count")
    plt.title("Histogram of Ratings")
    plt.xticks(rating_bins)
    plt.show()

    # Answer of 1.C
    movie_rating_counts = np.sum(ratings_matrix > 0, axis=0).tolist()
    movie_sorted_counts = np.sort(np.array(movie_rating_counts[0]))[::-1]

    plt.figure(figsize=(10, 5))
    plt.plot(movie_sorted_counts, linestyle='-')
    plt.xlabel("Movie Index")
    plt.ylabel("Number of Ratings")
    plt.title("Distribution of Ratings among Movies")
    plt.grid()
    plt.show()

    # Answer of 1.D
    user_rating_counts = np.sum(ratings_matrix > 0, axis=1).tolist()
    user_sorted_counts = np.sort(np.array(user_rating_counts)[:, 0])[::-1]

    plt.figure(figsize=(10, 5))
    plt.plot(user_sorted_counts, linestyle='-')
    plt.xlabel("User Index")
    plt.ylabel("Number of Ratings")
    plt.title("Distribution of Ratings among Users")
    plt.grid()
    plt.show()

    # Answer of 1.F
    ratings_matrix = ratings_matrix.tocsc()
    variances = np.zeros(num_movies)
    for j in range(num_movies):
        movie_j_ratings = ratings_matrix[:, j].toarray()[:, 0]
        movie_j_mean = np.mean(movie_j_ratings)
        movie_j_var = float(np.mean((movie_j_ratings - movie_j_mean)**2))
        variances[j] = movie_j_var
    variance_bins = np.arange(0, 5.5, 0.5)
    
    counts, bin_edges = np.histogram(variances, bins=variance_bins)
    plt.bar(bin_edges[:-1], counts, width=0.5, edgecolor='black', align='edge')
    plt.xlabel("Variance of Rating Values over Movies")
    plt.ylabel("Count")
    plt.title("Histogram of Variance of Ratings over Movies")
    plt.xticks(variance_bins)
    plt.show()