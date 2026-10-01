#!/usr/bin/env python3
"""
Generate a JSON file mapping all source IDs to their classifications.
Easy reference for looking up the label for any source ID.
"""

import json
import numpy as np
import pandas as pd
import os

print("="*70)
print("GENERATING SOURCE ID TO CLASSIFICATION MAPPING")
print("="*70)

# Load RADCAT
radcat_df = pd.read_csv("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv", index_col=0)
radcat_df = radcat_df[radcat_df["Type"].isin(["COMPACT", "FRI", "FRII"])]

# Load our datasets to get the combined list
names_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/names_train.npy", allow_pickle=True)
y_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/y_train.npy")

names_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/names_test.npy", allow_pickle=True)
y_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/y_test.npy")

# Create mappings
id_to_classification = {}
id_to_dataset = {}

print("\nMapping training set...")
for source_id, label in zip(names_train, y_train):
    source_id_str = str(source_id)
    id_to_classification[source_id_str] = str(label)
    id_to_dataset[source_id_str] = "train"

print(f"  Added {len(names_train)} training samples")

print("Mapping test set...")
for source_id, label in zip(names_test, y_test):
    source_id_str = str(source_id)
    id_to_classification[source_id_str] = str(label)
    id_to_dataset[source_id_str] = "test"

print(f"  Added {len(names_test)} test samples")

# Create comprehensive mapping with metadata
mapping_data = {
    "metadata": {
        "total_sources": len(id_to_classification),
        "classes": ["COMPACT", "FRI", "FRII"],
        "generated": "2026-04-22",
        "description": "Complete mapping of source IDs to their classifications"
    },
    "class_distribution": {},
    "id_to_classification": id_to_classification,
    "id_to_dataset": id_to_dataset
}

# Calculate class distribution
for class_name in ["COMPACT", "FRI", "FRII"]:
    count = sum(1 for label in id_to_classification.values() if label == class_name)
    mapping_data["class_distribution"][class_name] = {
        "count": count,
        "percentage": round(count / len(id_to_classification) * 100, 1)
    }

# Save to JSON
output_path = "/home/abigaildeklerk/Desktop/DeKlerk_Models/source_id_classification_mapping.json"
with open(output_path, 'w') as f:
    json.dump(mapping_data, f, indent=2)

print(f"\n✓ JSON mapping saved to: {output_path}")

# Display summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nTotal sources: {mapping_data['metadata']['total_sources']}")
print("\nClass Distribution:")
for class_name, info in mapping_data["class_distribution"].items():
    print(f"  {class_name}: {info['count']:4d} ({info['percentage']:5.1f}%)")

print("\nDataset Split:")
train_count = sum(1 for ds in id_to_dataset.values() if ds == "train")
test_count = sum(1 for ds in id_to_dataset.values() if ds == "test")
print(f"  Train: {train_count} ({train_count/len(id_to_classification)*100:.1f}%)")
print(f"  Test:  {test_count} ({test_count/len(id_to_classification)*100:.1f}%)")

print("\n" + "="*70)
print("USAGE EXAMPLES")
print("="*70)
print("""
import json

# Load the mapping
with open('source_id_classification_mapping.json', 'r') as f:
    mapping = json.load(f)

# Get classification for a specific source ID
source_id = "10000"
classification = mapping['id_to_classification'][source_id]
dataset = mapping['id_to_dataset'][source_id]

print(f"Source {source_id}: {classification} (in {dataset} set)")
# Output: Source 10000: FRI (in train set)

# Get all sources of a specific class
frii_sources = [sid for sid, label in mapping['id_to_classification'].items() 
                if label == "FRII"]
print(f"FRII sources: {frii_sources[:5]}...")  # first 5

# Access class distribution
print(mapping['class_distribution'])
""")

print("\n" + "="*70)
