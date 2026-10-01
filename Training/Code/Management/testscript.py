import os

# CONFIG
BASE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Results"
EXP = "Experiment_1"
PATH = os.path.join(BASE, EXP, "Testing")

print(f"Checking path: {PATH}")

if not os.path.exists(PATH):
    print("!!! PATH DOES NOT EXIST !!!")
else:
    folders = os.listdir(PATH)
    print(f"Found {len(folders)} items in directory.")
    for f in folders[:5]: # Show first 5
        print(f" - Item: '{f}' | IsDir: {os.path.isdir(os.path.join(PATH, f))} | HasSet: {'_set' in f}")