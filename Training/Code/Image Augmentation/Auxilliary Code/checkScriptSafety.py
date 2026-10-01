"""
Script to verify that all Image Augmentation scripts correctly load and parse RADCAT.csv
"""
import sys
sys.path.insert(0, '/home/abigaildeklerk/Desktop/DeKlerk_Models')
import os
import ast
import re

SCRIPTS_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation"
RADCAT_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"

# Known good patterns for RADCAT.csv loading
GOOD_PATTERNS = [
    r"pd\.read_csv.*engine=['\"]python['\"].*on_bad_lines=['\"]skip['\"]",
    r"pd\.read_csv.*RADCAT.*engine=['\"]python['\"]",
    r"\.iloc\[:,\s*0\].*Type",  # Using iloc to get first column
]

# Bad patterns to flag
BAD_PATTERNS = [
    r"pd\.read_csv\([^)]*index_col=0[^)]*RADCAT",  # index_col=0 with RADCAT
    r"\.index.*RADCAT",  # Using .index for lookups
    r"\.loc\[[^,]*,\s*['\"]Type['\"]",  # Using .loc with Type column
]

def check_file(filepath):
    """Check a Python file for RADCAT loading issues"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    issues = []
    
    # Check for bad patterns
    for pattern in BAD_PATTERNS:
        if re.search(pattern, content):
            issues.append(f"Found potentially problematic pattern: {pattern}")
    
    # Specifically check for the problematic index_col=0 with RADCAT
    if "RADCAT" in content and "index_col=0" in content:
        # Check if they're in the same context
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "RADCAT" in line and "read_csv" in line and "index_col=0" in line:
                issues.append(f"Line {i+1}: Using index_col=0 with RADCAT.csv read")
    
    # Check for correct pattern with Python engine and on_bad_lines
    if "RADCAT" in content and "read_csv" in content:
        has_good_pattern = False
        for pattern in GOOD_PATTERNS:
            if re.search(pattern, content):
                has_good_pattern = True
                break
        
        if not has_good_pattern:
            # Check if they're using iloc to get IDs
            if "iloc[:, 0]" in content or ".iloc[:, 0]" in content:
                # This is okay if they're not using index_col=0
                if "index_col=0" not in content.split("RADCAT")[0].split("read_csv")[-1][:200]:
                    has_good_pattern = True
        
        if not has_good_pattern and "index_col=0" not in content:
            issues.append("RADCAT loading found but pattern not optimal (should use engine='python' and on_bad_lines='skip')")
    
    return issues

def main():
    print("Checking Image Augmentation scripts for RADCAT.csv handling issues...\n")
    print("=" * 70)
    
    py_files = [f for f in os.listdir(SCRIPTS_PATH) if f.endswith('.py')]
    
    all_good = True
    for filename in sorted(py_files):
        filepath = os.path.join(SCRIPTS_PATH, filename)
        issues = check_file(filepath)
        
        if not issues:
            print(f"✓ {filename}")
        else:
            all_good = False
            print(f"✗ {filename}")
            for issue in issues:
                print(f"    - {issue}")
    
    print("=" * 70)
    if all_good:
        print("\n✓ All scripts look good!")
    else:
        print("\n✗ Some scripts need attention (see above)")

if __name__ == "__main__":
    main()
