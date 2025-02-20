"""
Answers part2 of question 16
"""
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


def evaluate_model(model, X_test, y_test, qid_test):
    """Evaluate model performance using nDCG@3, nDCG@5, and nDCG@10."""
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

    return np.mean(ndcg_3_list), np.mean(ndcg_5_list), np.mean(ndcg_10_list)


def get_bottom_features(model, bot_n=60):
    """Retrieve the top N most important features based on 'gain'."""
    importance_df = (
        pd.DataFrame({
            'feature_index': range(len(model.feature_name())),
            'importance_gain': model.feature_importance(importance_type='gain'),
        })
        .sort_values('importance_gain', ascending=False)
        .reset_index(drop=True)
    )
    return importance_df.tail(bot_n)['feature_index'].tolist()


def remove_features(X, features_to_remove):
    """Remove specified features from the dataset (sparse matrix support)."""
    keep_features = [i for i in range(X.shape[1]) if i not in features_to_remove]
    return X[:, keep_features]


def train_and_evaluate_reduced(folds_folder):
    """Train LightGBM after removing bottom 60 features and evaluate performance."""
    fold_names = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]

    for fold in fold_names:
        print(f"\n==== Training {fold} - Reduced Feature Set ====")
        fold_path = os.path.join(folds_folder, fold + "/")

        X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test = load_one_fold(fold_path)
        X_train_split, X_val, y_train_split, y_val, qid_train_split, qid_val = train_test_split(
            X_train, y_train, qid_train, test_size=0.1, random_state=42
        )
        _, group_train_split = np.unique(qid_train_split, return_counts=True)
        _, group_val = np.unique(qid_val, return_counts=True)
        train_data_initial = lgb.Dataset(X_train_split, label=y_train_split, group=group_train_split)
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

        initial_model = lgb.train(
            params, train_data_initial, num_boost_round=800,
            valid_sets=[train_data_initial],
            callbacks=[lgb.early_stopping(stopping_rounds=5),lgb.log_evaluation(period=50)]
        )

        bot_60_features = get_bottom_features(initial_model, bot_n=60)
        print(f"Removed bottom 60 features for {fold}: {bot_60_features}")

        X_train_reduced = remove_features(X_train_split, bot_60_features)
        X_val_reduced = remove_features(X_val, bot_60_features)
        X_test_reduced = remove_features(X_test, bot_60_features)

        train_data_reduced = lgb.Dataset(X_train_reduced, label=y_train_split, group=group_train_split)
        val_data_reduced = lgb.Dataset(X_val_reduced, label=y_val, group=group_val)

        final_model = lgb.train(
            params,
            train_data_reduced,
            num_boost_round=800,
            valid_sets=[train_data_reduced, val_data_reduced],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5),
                lgb.log_evaluation(period=50)
            ]
        )

        ndcg_3, ndcg_5, ndcg_10 = evaluate_model(final_model, X_test_reduced, y_test, qid_test)
        print(f"\n===== Performance after removing bottom 60 features ({fold}) =====")
        print(f"nDCG@3:  {ndcg_3:.4f}")
        print(f"nDCG@5:  {ndcg_5:.4f}")
        print(f"nDCG@10: {ndcg_10:.4f}")


if __name__ == "__main__":
    dataset_folder = "C:/Users/AndyXing/Desktop/ece_219/MSLR-WEB10K"  # Update if needed
    train_and_evaluate_reduced(dataset_folder)

# Key Reasons:
# Low Feature Redundancy
# The "least important" features may still provide auxiliary information, especially for specific queries. Removing them doesn’t improve performance significantly.
    
# High Signal-to-Noise Ratio
# The dataset may have less noise, so even features with low global importance contribute marginally, and their removal doesn't boost performance.

# Locally Important Features
# Some features are essential for specific query groups. Global importance scores might overlook these context-dependent contributions.
    
# LightGBM Importance Metric Limitations
# The gain metric might undervalue features that impact deep tree splits or have high feature correlation, leading to underestimated importance.

# Query Distribution Discrepancy
# Differences in query distribution between training and testing sets may mean that globally unimportant features are locally important for certain test queries.
    
# nDCG Sensitivity
# Since nDCG emphasizes top-ranked documents, features affecting lower-ranked results may not show significant performance differences when removed.

# Conclusion
# The lack of improvement is likely due to the global importance bias, limited redundancy, and local significance of the removed features. To address this, consider incremental feature removal, SHAP-based importance, or query-specific feature evaluations for further exploration

# ===== Performance after removing bottom 60 features (Fold1) =====
# nDCG@3:  0.5220
# nDCG@5:  0.5213
# nDCG@10: 0.5255

# ===== Performance after removing bottom 60 features (Fold2) =====
# nDCG@3:  0.5180
# nDCG@5:  0.5151
# nDCG@10: 0.5194

# ===== Performance after removing bottom 60 features (Fold3) =====
# nDCG@3:  0.5113
# nDCG@5:  0.5103
# nDCG@10: 0.5162

# ===== Performance after removing bottom 60 features (Fold4) =====
# nDCG@3:  0.5157
# nDCG@5:  0.5157
# nDCG@10: 0.5237

# ===== Performance after removing bottom 60 features (Fold5) =====
# nDCG@3:  0.5348
# nDCG@5:  0.5307
# nDCG@10: 0.5337