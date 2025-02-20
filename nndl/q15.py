import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def load_one_fold(data_path):
    """Load one fold of the dataset using sklearn's load_svmlight_file."""
    X_train, y_train, qid_train = load_svmlight_file(str(data_path + 'train.txt'), query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(str(data_path + 'test.txt'), query_id=True)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    _, group_train = np.unique(qid_train, return_counts=True)
    _, group_test = np.unique(qid_test, return_counts=True)
    return X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test

def evaluate_model(model, X_test, y_test, qid_test, group_test):
    """Evaluate the model using nDCG@3, nDCG@5, and nDCG@10, ensuring proper query grouping."""
    unique_qids = np.unique(qid_test)
    ndcg_3_list, ndcg_5_list, ndcg_10_list = [], [], []

    for qid in unique_qids:
        indices = qid_test == qid
        if np.sum(indices) < 2:
            continue
        y_true = y_test[indices]
        y_score = model.predict(X_test[indices], num_iteration=model.best_iteration)
        ndcg_3_list.append(ndcg_score([y_true], [y_score], k=3))
        ndcg_5_list.append(ndcg_score([y_true], [y_score], k=5))
        ndcg_10_list.append(ndcg_score([y_true], [y_score], k=10))

    return (
        np.mean(ndcg_3_list) if ndcg_3_list else 0,
        np.mean(ndcg_5_list) if ndcg_5_list else 0,
        np.mean(ndcg_10_list) if ndcg_10_list else 0
    )

def get_top_features(model, top_n=5):
    """Retrieve the top N most important features based on 'gain' importance."""
    importance_df = (
        pd.DataFrame({
            'feature_name': model.feature_name(),
            'importance_gain': model.feature_importance(importance_type='gain'),
            'importance_split': model.feature_importance(importance_type='split'),
        })
        .sort_values('importance_gain', ascending=False)
        .reset_index(drop=True)
    )
    return importance_df.head(top_n)

def train_and_evaluate(folds_folder):
    """Train and evaluate the optimized LightGBM model across all five folds and list top 5 features."""
    fold_names = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
    all_fold_features = []

    for fold in fold_names:
        print(f"\n==== Training {fold} ====")
        fold_path = os.path.join(folds_folder, fold + "/")

        X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test = load_one_fold(fold_path)
        X_train_split, X_val, y_train_split, y_val, qid_train_split, qid_val = train_test_split(
            X_train, y_train, qid_train, test_size=0.1, random_state=42
        )
        _, group_train_split = np.unique(qid_train_split, return_counts=True)
        _, group_val = np.unique(qid_val, return_counts=True)

        train_data = lgb.Dataset(X_train_split, label=y_train_split, group=group_train_split)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val)

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'device': 'gpu',  # Use GPU for faster training
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'learning_rate': 0.1,
            'num_leaves': 31,      
            'min_data_in_leaf': 20,
            'verbose': -1,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0, 

        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=800, 
            valid_sets=[train_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5),  # Early stopping for faster convergence
                lgb.log_evaluation(period=50)
            ]
        )

        # Evaluate performance
        ndcg_3, ndcg_5, ndcg_10 = evaluate_model(model, X_test, y_test, qid_test, group_test)
        print(f"Performance on {fold} test set:")
        print(f"nDCG@3:  {ndcg_3:.4f}")
        print(f"nDCG@5:  {ndcg_5:.4f}")
        print(f"nDCG@10: {ndcg_10:.4f}")

        # Get top 5 features
        top_features = get_top_features(model, top_n=5)
        print(f"Top 5 most important features for {fold}:")
        print(top_features)
        all_fold_features.append((fold, top_features))

    return all_fold_features

if __name__ == "__main__":
    dataset_folder = "C:/Users/AndyXing/Desktop/ece_219/MSLR-WEB10K"  # Update if needed
    top_features_per_fold = train_and_evaluate(dataset_folder)

    # Combine and display results
    for fold, features in top_features_per_fold:
        print(f"\n===== {fold} - Top 5 Features =====")
        print(features)

# ===== Fold1 - Top 5 Features =====
#   feature_name  importance_gain  importance_split
# 0   Column_133     22509.302441               257
# 1    Column_54     14422.273663               218
# 2   Column_129      6566.901497               814
# 3   Column_128      4828.602377               472
# 4    Column_14      4605.773697               406

# ===== Fold2 - Top 5 Features =====
#   feature_name  importance_gain  importance_split
# 0   Column_133     22282.439733               273
# 1    Column_54     15825.311511               255
# 2   Column_129      7057.400825               917
# 3   Column_130      5328.208504              1253
# 4   Column_128      4406.970275               524

# ===== Fold3 - Top 5 Features =====
#   feature_name  importance_gain  importance_split
# 0   Column_133     22420.718331               296
# 1    Column_54     13112.206020               221
# 2   Column_129      6768.141324               953
# 3   Column_128      5197.381736               460
# 4   Column_130      5021.586955              1221

# ===== Fold4 - Top 5 Features =====
#   feature_name  importance_gain  importance_split
# 0   Column_133     22891.101171               273
# 1    Column_54     15846.240428               256
# 2   Column_129      5640.414552               915
# 3   Column_128      4834.048760               476
# 4   Column_130      4510.946808              1140

# ===== Fold5 - Top 5 Features =====
#   feature_name  importance_gain  importance_split
# 0   Column_133     21986.197452               267
# 1    Column_54     15900.570613               234
# 2   Column_129      5494.489304               929
# 3   Column_130      5408.566897              1200
# 4   Column_128      5275.257707               529

