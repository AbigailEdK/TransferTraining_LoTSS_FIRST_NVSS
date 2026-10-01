#!/usr/bin/env python3
"""
Comprehensive verification script to confirm:
1. Source names are correctly assigned to images
2. Labels match RADCAT for each source
3. Image data dimensions match expectations
"""

import numpy as np
import pandas as pd
import os

print("="*70)
print("VERIFICATION: Names, Labels, and Image Correspondence")
print("="*70)

# Load datasets
X_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/ConstructData/results/same_pixels/X_train_no_clip.npy")
y_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/ConstructData/results/same_pixels/y_train.npy")
names_train = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/ConstructData/results/same_pixels/names_train.npy", allow_pickle=True)

X_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/ConstructData/results/same_pixels/X_test_no_clip.npy")
y_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/ConstructData/results/same_pixels/y_test.npy")
names_test = np.load("/home/abigaildeklerk/Desktop/DeKlerk_Models/ConstructData/results/same_pixels/names_test.npy", allow_pickle=True)

# Load RADCAT
radcat_df = pd.read_csv("/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv", index_col=0)
radcat_df = radcat_df[radcat_df["Type"].isin(["COMPACT", "FRI", "FRII"])]

# Path to combined arrays
combined_dir = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_COMBINED_NUMPY"

print("\n" + "="*70)
print("1. CHECKING TRAIN SET")
print("="*70)

train_errors = []
train_checked = 0

for i in range(min(len(X_train), len(y_train), len(names_train))):
    source_id = str(names_train[i])
    label = y_train[i]
    image = X_train[i]
    
    # Check 1: Source ID exists in RADCAT
    if int(source_id) not in radcat_df.index:
        train_errors.append(f"Train[{i}]: Source {source_id} not in RADCAT")
        continue
    
    # Check 2: Label matches RADCAT
    expected_label = radcat_df.loc[int(source_id), "Type"]
    if str(label) != str(expected_label):
        train_errors.append(
            f"Train[{i}]: Source {source_id} has label '{label}' but RADCAT says '{expected_label}'"
        )
        continue
    
    # Check 3: Image file exists
    image_path = os.path.join(combined_dir, f"{source_id}.npy")
    if not os.path.exists(image_path):
        train_errors.append(f"Train[{i}]: Combined image file not found: {image_path}")
        continue
    
    # Check 4: Image dimensions are correct
    if image.shape != (128, 128, 3):
        train_errors.append(
            f"Train[{i}]: Source {source_id} has incorrect shape {image.shape}, expected (128, 128, 3)"
        )
        continue
    
    # Check 5: Image data is in valid uint8 range
    if image.dtype != np.uint8 or image.min() < 0 or image.max() > 255:
        train_errors.append(
            f"Train[{i}]: Source {source_id} has invalid data type or range"
        )
        continue
    
    train_checked += 1

print(f"✓ Checked {train_checked}/{len(X_train)} training samples")
if train_errors:
    print(f"✗ Found {len(train_errors)} errors in training set:")
    for error in train_errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(train_errors) > 10:
        print(f"  ... and {len(train_errors) - 10} more errors")
else:
    print("✓ No errors found in training set!")

print("\n" + "="*70)
print("2. CHECKING TEST SET")
print("="*70)

test_errors = []
test_checked = 0

for i in range(min(len(X_test), len(y_test), len(names_test))):
    source_id = str(names_test[i])
    label = y_test[i]
    image = X_test[i]
    
    # Check 1: Source ID exists in RADCAT
    if int(source_id) not in radcat_df.index:
        test_errors.append(f"Test[{i}]: Source {source_id} not in RADCAT")
        continue
    
    # Check 2: Label matches RADCAT
    expected_label = radcat_df.loc[int(source_id), "Type"]
    if str(label) != str(expected_label):
        test_errors.append(
            f"Test[{i}]: Source {source_id} has label '{label}' but RADCAT says '{expected_label}'"
        )
        continue
    
    # Check 3: Image file exists
    image_path = os.path.join(combined_dir, f"{source_id}.npy")
    if not os.path.exists(image_path):
        test_errors.append(f"Test[{i}]: Combined image file not found: {image_path}")
        continue
    
    # Check 4: Image dimensions are correct
    if image.shape != (128, 128, 3):
        test_errors.append(
            f"Test[{i}]: Source {source_id} has incorrect shape {image.shape}, expected (128, 128, 3)"
        )
        continue
    
    # Check 5: Image data is in valid uint8 range
    if image.dtype != np.uint8 or image.min() < 0 or image.max() > 255:
        test_errors.append(
            f"Test[{i}]: Source {source_id} has invalid data type or range"
        )
        continue
    
    test_checked += 1

print(f"✓ Checked {test_checked}/{len(X_test)} test samples")
if test_errors:
    print(f"✗ Found {len(test_errors)} errors in test set:")
    for error in test_errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(test_errors) > 10:
        print(f"  ... and {len(test_errors) - 10} more errors")
else:
    print("✓ No errors found in test set!")

print("\n" + "="*70)
print("3. DETAILED SPOT CHECKS")
print("="*70)

# Show detailed examples from train and test
print("\nTrain Set Examples:")
for i in [0, len(X_train)//2, len(X_train)-1]:
    if i < len(X_train):
        source_id = str(names_train[i])
        label = y_train[i]
        radcat_label = radcat_df.loc[int(source_id), "Type"]
        image = X_train[i]
        print(f"\n  Index {i}:")
        print(f"    Source ID:       {source_id}")
        print(f"    Label in array:  {label}")
        print(f"    Label in RADCAT: {radcat_label}")
        print(f"    Match:          {'✓' if str(label) == str(radcat_label) else '✗'}")
        print(f"    Image shape:     {image.shape}")
        print(f"    Data range:      [{image.min()}, {image.max()}]")
        print(f"    Data type:       {image.dtype}")
        print(f"    Channel means:   R={image[:,:,0].mean():.1f}, G={image[:,:,1].mean():.1f}, B={image[:,:,2].mean():.1f}")

print("\n\nTest Set Examples:")
for i in [0, len(X_test)//2, len(X_test)-1]:
    if i < len(X_test):
        source_id = str(names_test[i])
        label = y_test[i]
        radcat_label = radcat_df.loc[int(source_id), "Type"]
        image = X_test[i]
        print(f"\n  Index {i}:")
        print(f"    Source ID:       {source_id}")
        print(f"    Label in array:  {label}")
        print(f"    Label in RADCAT: {radcat_label}")
        print(f"    Match:          {'✓' if str(label) == str(radcat_label) else '✗'}")
        print(f"    Image shape:     {image.shape}")
        print(f"    Data range:      [{image.min()}, {image.max()}]")
        print(f"    Data type:       {image.dtype}")
        print(f"    Channel means:   R={image[:,:,0].mean():.1f}, G={image[:,:,1].mean():.1f}, B={image[:,:,2].mean():.1f}")

print("\n" + "="*70)
print("4. FINAL SUMMARY")
print("="*70)

total_checked = train_checked + test_checked
total_samples = len(X_train) + len(X_test)
total_errors = len(train_errors) + len(test_errors)

print(f"\nTotal samples checked: {total_checked}/{total_samples}")
print(f"Total errors found: {total_errors}")

if total_errors == 0:
    print("\n✓✓✓ ALL VERIFICATIONS PASSED ✓✓✓")
    print("Names and labels are correctly assigned to all images!")
else:
    print(f"\n✗ {total_errors} verification(s) failed - see details above")

print("\n" + "="*70)
