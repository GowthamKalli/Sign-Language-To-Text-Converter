import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = r"E:\Sign Language to Text\st-gcn-sl\data\train"
OUTPUT_DIR = r"E:\Sign Language to Text\st-gcn-sl\data"

# Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all npy files
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
print(f"Found {len(files)} npy files.")

# Extract labels from filenames
# Assuming filenames like 'label_XXXX.npy' or '123.npy'
labels = []
for f in files:
    name = os.path.splitext(f)[0]
    try:
        # if filename is like 'hello_42.npy' → label = 'hello'
        label = name.split("_")[0]
    except:
        label = "unknown"
    labels.append(label)

# Encode string labels as integers
unique_labels = sorted(list(set(labels)))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
encoded_labels = np.array([label_to_idx[l] for l in labels])

print(f"Detected {len(unique_labels)} unique classes.")

# Split train/val
train_files, val_files, y_train, y_val = train_test_split(
    files, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

def load_npy_list(file_list):
    data = []
    for f in file_list:
        arr = np.load(os.path.join(DATA_DIR, f))
        data.append(arr)
    return np.array(data, dtype=object)  # variable-length skeletons

print("Loading train and validation data (this might take a minute)...")
X_train = load_npy_list(train_files)
X_val = load_npy_list(val_files)

# Save datasets
np.save(os.path.join(OUTPUT_DIR, "train_data.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "train_label.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "val_data.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "val_label.npy"), y_val)

print("✅ Dataset split complete!")
print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")
print(f"Files saved in {OUTPUT_DIR}")
