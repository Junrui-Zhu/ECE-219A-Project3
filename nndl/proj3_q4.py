from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise import KNNBasic
from surprise import accuracy
from proj3_q1 import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    rating_df = load_rating_data()

    # Convert data to Surprise format
    reader = Reader(rating_scale=(0.5, 5))
    dataset = Dataset.load_from_df(rating_df[["userId", "movieId", "rating"]], reader)

    # Define kNN collaborative filtering model
    sim_options = {
        "name": "Pearson",
        "user_based": True 
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    k_list = []
    RMSE_list = []
    MAE_list = []
    for k in range(2, 102, 2):
        model = KNNBasic(k=k, sim_options=sim_options)
        average_rmse = 0
        average_mae = 0
        for fold, (trainset, testset) in enumerate(kf.split(dataset)):

            # Train the model
            model.fit(trainset)

            # Make predictions
            predictions = model.test(testset)

            # Evaluate RMSE
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)
            average_rmse += rmse
            average_mae += mae

        average_rmse /= 9
        average_mae /= 9

        k_list.append(k)
        RMSE_list.append(average_rmse)
        MAE_list.append(average_mae)

    plt.figure(figsize=(10, 5))
    plt.plot(k_list, RMSE_list, label='RMSE')
    plt.plot(k_list, MAE_list, label='MAE')
    plt.xlabel("k")
    plt.ylabel("Average Error")
    plt.title("Plot of Average Error over k")
    plt.legend()
    plt.grid()
    plt.show()

    for i in range(len(k_list)-1):
        decrement_rate = (MAE_list[i] - MAE_list[i+1] + RMSE_list[i] - RMSE_list[i+1]) / (MAE_list[i] + RMSE_list[i])
        if decrement_rate < 0.0001:
            break
    print(f"k_min: {i}")
    print(f"Stable RMSE: {RMSE_list[i]}")
    print(f"Stable MAE: {MAE_list[i]}")
