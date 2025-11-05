import os
import json
import random
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
SKELETON_DIR = r"E:\Sign Language to Text\data\skeletons_lsa64_final"
RGB_DIR = r"E:\Sign Language to Text\data\lsa64_rgb_features"
OUT_DIR = r"E:\Sign Language to Text\st-gcn-sl\data_splits"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
random.seed(42)

# ============================================================
# HELPERS
# ============================================================
def parse_filename(name):
    """Extract signer_id, sign_id, sample_id from filename."""
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None

def get_signer_map(files):
    """Return {signer_id: [file1, file2, ...]}"""
    signer_map = {}
    for f in files:
        signer, _, _ = parse_filename(f)
        if signer:
            signer_map.setdefault(signer, []).append(f)
    return signer_map

# ============================================================
# MAIN
# ============================================================
print("📁 Scanning dataset folders...")

skeleton_files = [f for f in os.listdir(SKELETON_DIR) if f.endswith(".npy")]
rgb_files = [f[:-8] for f in os.listdir(RGB_DIR) if f.endswith("_rgb.npy")]  # remove '_rgb.npy'

skeleton_set = set(os.path.splitext(f)[0] for f in skeleton_files)
rgb_set = set(rgb_files)

common = sorted(skeleton_set & rgb_set)

print(f"✅ Found {len(common)} samples with both skeletons + RGB features.")

# Build metadata
metadata = []
for base in tqdm(common, desc="Building metadata"):
    signer_id, sign_id, sample_id = parse_filename(base)
    skeleton_path = os.path.join(SKELETON_DIR, base + ".npy")
    rgb_path = os.path.join(RGB_DIR, base + "_rgb.npy")
    metadata.append({
        "id": base,
        "signer_id": signer_id,
        "sign_id": sign_id,
        "sample_id": sample_id,
        "skeleton_path": skeleton_path,
        "rgb_path": rgb_path
    })

# Group by signer
signer_map = get_signer_map([m["id"] for m in metadata])
signers = list(signer_map.keys())
random.shuffle(signers)

n = len(signers)
train_signers = set(signers[:int(n * TRAIN_RATIO)])
val_signers = set(signers[int(n * TRAIN_RATIO): int(n * (TRAIN_RATIO + VAL_RATIO))])
test_signers = set(signers[int(n * (TRAIN_RATIO + VAL_RATIO)):])

def assign_split(m):
    sid = m["signer_id"]
    if sid in train_signers:
        return "train"
    elif sid in val_signers:
        return "val"
    else:
        return "test"

for m in metadata:
    m["split"] = assign_split(m)

train = [m for m in metadata if m["split"] == "train"]
val = [m for m in metadata if m["split"] == "val"]
test = [m for m in metadata if m["split"] == "test"]

# Save splits
def save_json(data, name):
    out_path = os.path.join(OUT_DIR, f"{name}_split.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {name} split with {len(data)} samples → {out_path}")

save_json(train, "train")
save_json(val, "val")
save_json(test, "test")

# Summary
summary = {
    "total_samples": len(common),
    "total_signers": len(signers),
    "train_signers": len(train_signers),
    "val_signers": len(val_signers),
    "test_signers": len(test_signers),
}
with open(os.path.join(OUT_DIR, "metadata_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n🎉 Metadata and dataset splits generated successfully!")
print(f"📄 Metadata summary: {os.path.join(OUT_DIR, 'metadata_summary.json')}")
