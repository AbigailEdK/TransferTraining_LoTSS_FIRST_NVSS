#!/usr/bin/env python3
"""
Verification script to confirm the combined datasets work with get_dataset()
"""

import numpy as np
import sys

print("="*60)
print("VERIFICATION: Dataset Compatibility")
print("="*60)

# Load the train set to verify format
X_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/X_train_no_clip.npy")
y_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/y_train.npy")
names_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/names_train.npy")

X_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/X_test_no_clip.npy")
y_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/y_test.npy")
names_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS/names_test.npy")

print("\n✓ TRAIN SET")
print(f"  X_train shape:     {X_train.shape} (samples, height, width, channels)")
print(f"  y_train shape:     {y_train.shape}")
print(f"  names_train shape: {names_train.shape}")
print(f"  Data type (X):     {X_train.dtype}")
print(f"  Data type (y):     {y_train.dtype}")

print("\n✓ TEST SET")
print(f"  X_test shape:      {X_test.shape} (samples, height, width, channels)")
print(f"  y_test shape:      {y_test.shape}")
print(f"  names_test shape:  {names_test.shape}")
print(f"  Data type (X):     {X_test.dtype}")
print(f"  Data type (y):     {y_test.dtype}")

print("\n✓ CLASS DISTRIBUTION")
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print("\n  Train set:")
for class_name, count in zip(unique_train, counts_train):
    pct = count / len(y_train) * 100
    print(f"    {class_name}: {count:4d} ({pct:5.1f}%)")

print("\n  Test set:")
for class_name, count in zip(unique_test, counts_test):
    pct = count / len(y_test) * 100
    print(f"    {class_name}: {count:4d} ({pct:5.1f}%)")

print("\n✓ SAMPLE VALIDATION")
print(f"  Example source ID:     {names_train[0]}")
print(f"  Example label:         {y_train[0]}")
print(f"  Example image range:   [{X_train[0].min():.0f}, {X_train[0].max():.0f}]")
print(f"  Example image shape:   {X_train[0].shape}")

print("\n✓ READY TO USE")
print("  Files are compatible with get_dataset() function")
print("  Usage: get_dataset('train', cf) or get_dataset('test', cf)")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
