"""
Answers question 14
"""

import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
import numpy as np
import os

def load_one_fold(data_path):
    """Load the dataset for one fold using sklearn's load_svmlight_file."""
    X_train, y_train, qid_train = load_svmlight_file(str(data_path + 'train.txt'), query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(str(data_path + 'test.txt'), query_id=True)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    _, group_train = np.unique(qid_train, return_counts=True)
    _, group_test = np.unique(qid_test, return_counts=True)
    return X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test

def evaluate_model(model, X_test, y_test, qid_test):
    """Evaluate the model's performance using nDCG@3, nDCG@5, and nDCG@10 with document count checks."""
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


def train_and_evaluate(folds_folder):
    """Train and evaluate LightGBM model with optimized settings for fast training."""
    fold_names = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]

    for fold in fold_names:
        print(f"\n==== Training {fold} ====")
        fold_path = os.path.join(folds_folder, fold + "/")

        X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test = load_one_fold(fold_path)

        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)

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

        ndcg_3, ndcg_5, ndcg_10 = evaluate_model(model, X_test, y_test, qid_test)
        print(f"Performance on {fold} test set:")
        print(f"nDCG@3:  {ndcg_3:.4f}")
        print(f"nDCG@5:  {ndcg_5:.4f}")
        print(f"nDCG@10: {ndcg_10:.4f}")

if __name__ == "__main__":
    dataset_folder = "C:/Users/AndyXing/Desktop/ece_219/MSLR-WEB10K"  # Update as needed
    train_and_evaluate(dataset_folder)
# Performance on Fold1 test set:
# nDCG@3:  0.5354
# nDCG@5:  0.5349
# nDCG@10: 0.5393
# Performance on Fold2 test set:
# nDCG@3:  0.5403
# nDCG@5:  0.5367
# nDCG@10: 0.5392
# Performance on Fold3 test set:
# nDCG@3:  0.5309
# nDCG@5:  0.5317
# nDCG@10: 0.5388
# Performance on Fold4 test set:
# nDCG@3:  0.5427
# nDCG@5:  0.5428
# nDCG@10: 0.5484
# Performance on Fold5 test set:
# nDCG@3:  0.5428
# nDCG@5:  0.5448
# nDCG@10: 0.5499