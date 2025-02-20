"""
This file answers question 13, loads the dataset and preprocesses it using sklearn's load_svmlight_file.
Additionally, it prints out the number of unique queries and shows the distribution of relevance labels.
"""

from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def load_one_fold(data_path):
    """
    Load dataset for one fold using sklearn's load_svmlight_file.
    Returns the features, labels, query IDs, and group information for training and testing sets.
    """
    X_train, y_train, qid_train = load_svmlight_file(str(data_path + 'train.txt'), query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(str(data_path + 'test.txt'), query_id=True)
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    _, group_train = np.unique(qid_train, return_counts=True)
    _, group_test = np.unique(qid_test, return_counts=True)

    return X_train, y_train, qid_train, group_train, X_test, y_test, qid_test, group_test

def analyze_dataset(folder_path):
    """
    Analyze the dataset to print out the total number of unique queries and display the distribution of relevance labels.
    """
    all_query_ids = []
    all_relevance_labels = []

    # Iterate through each fold and load data
    for fold in sorted(os.listdir(folder_path)):
        fold_path = os.path.join(folder_path, fold)
        if os.path.isdir(fold_path):
            print(f"\nLoading data from {fold}...")
            try:
                X_train, y_train, qid_train, _, X_test, y_test, qid_test, _ = load_one_fold(fold_path + "/")
                
                all_query_ids.extend(qid_train)
                all_query_ids.extend(qid_test)
                
                all_relevance_labels.extend(y_train)
                all_relevance_labels.extend(y_test)
                
                print(f"Fold {fold} loaded successfully.")
            except Exception as e:
                print(f"Error loading {fold}: {e}")

    # Calculate unique queries
    unique_queries = len(np.unique(all_query_ids))
    print(f"\nTotal number of unique queries: {unique_queries}")

    # Calculate relevance label distribution
    label_distribution = Counter(all_relevance_labels)
    print("\nDistribution of relevance labels:")
    for label, count in sorted(label_distribution.items()):
        print(f"Label {label}: {count} samples")

    # Plot the distribution of relevance labels
    plt.figure(figsize=(8, 5))
    plt.bar(label_distribution.keys(), label_distribution.values(), color='skyblue')
    plt.xlabel('Relevance Label')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Relevance Labels in MSLR-WEB10K Dataset')
    plt.xticks(list(label_distribution.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    # Specify dataset folder path containing Fold1 to Fold5
    dataset_folder = "C:/Users/AndyXing/Desktop/ece_219/MSLR-WEB10K"  # Update this path as needed
    analyze_dataset(dataset_folder)
