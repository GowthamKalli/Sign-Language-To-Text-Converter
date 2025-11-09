# tools/fix_splits_paths.py
import json
import os
from pathlib import Path

OLD_PREFIX = r"E:\Sign Language to Text"
NEW_PREFIX = r"C:\Users\jeffl\OneDrive\Desktop\Sign Language to Text"

# directory containing split jsons (adjust if yours is different)
SPLIT_DIR = Path("data_splits")

def fix_path(p):
    if not p:
        return p
    if p.startswith(OLD_PREFIX):
        # convert E:\... to C:\Users\...
        return p.replace(OLD_PREFIX, NEW_PREFIX, 1)
    return p

def process_file(path: Path):
    print("Processing:", path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except Exception as e:
        print("  ✖ JSON parse error:", e)
        return

    changed = False
    # If root is a list of dicts (your split format), update skeleton_path
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "skeleton_path" in item:
                old = item["skeleton_path"]
                new = fix_path(old)
                if new != old:
                    item["skeleton_path"] = new
                    changed = True
    elif isinstance(data, dict):
        # just in case structure is different (defensive)
        for k, v in data.items():
            if isinstance(v, str) and v.startswith(OLD_PREFIX):
                data[k] = fix_path(v)
                changed = True

    if changed:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            path.replace(bak)  # move original to .bak
            # write updated JSON back to original path
            bak.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            # move bak back to original name (we wrote to bak, so now replace)
            Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print("  ✓ Updated and created backup:", bak)
        else:
            # if backup already exists, just overwrite file
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print("  ✓ Updated (backup existed):", bak)
    else:
        print("  - No changes needed.")

def main():
    if not SPLIT_DIR.exists():
        print("Split directory not found:", SPLIT_DIR)
        return
    for j in SPLIT_DIR.glob("*.json"):
        process_file(j)
    print("Done. Please run the validator next.")

if __name__ == "__main__":
    main()
