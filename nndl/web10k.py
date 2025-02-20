"""
This file answers question 13, loads the dataset and preprocesses it
"""
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

def load_mslr_web10k_data(file_path):
    relevance_labels = []
    query_ids = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            relevance_labels.append(int(parts[0]))  # Extract relevance label
            qid = int(parts[1].split(":")[1])      # Extract query ID
            query_ids.append(qid)
    return pd.DataFrame({
        'query_id': query_ids,
        'relevance_label': relevance_labels
    })

def analyze_dataset(folder_path):
    all_data = pd.DataFrame()

    # Iterate through Fold1 to Fold5
    for fold in sorted(os.listdir(folder_path)):
        fold_path = os.path.join(folder_path, fold)
        if os.path.isdir(fold_path):
            print(f"\nLoading data from {fold}...")
            for file_name in os.listdir(fold_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(fold_path, file_name)
                    print(f"- Loading file: {file_name}")
                    fold_data = load_mslr_web10k_data(file_path)
                    all_data = pd.concat([all_data, fold_data], ignore_index=True)

    # Calculate the number of unique queries
    unique_queries = all_data['query_id'].nunique()
    print(f"\n\033[92mNumber of unique queries: {unique_queries}\033[0m")

    # Calculate the distribution of relevance labels
    label_distribution = Counter(all_data['relevance_label'])
    print("\n\033[94mDistribution of relevance labels:\033[0m")
    #for label, count in sorted(label_distribution.items()):
     #   print(f"Label {label}: {count} samples")

    # Plot the distribution of relevance labels
    plt.bar(label_distribution.keys(), label_distribution.values(), color='skyblue')
    plt.xlabel('Relevance Label')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Relevance Labels in MSLR-WEB10K Dataset')
    plt.xticks(list(label_distribution.keys()))
    plt.show()

if __name__ == "__main__":
    # Assume the dataset folder path is "MSLR-WEB10K" in the current directory
    dataset_folder = "C:/Users/AndyXing/Desktop/ece_219/MSLR-WEB10K"  # Update to your actual dataset path
    analyze_dataset(dataset_folder)
