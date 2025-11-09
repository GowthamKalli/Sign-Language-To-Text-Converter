import argparse, json, random, shutil
from pathlib import Path
from collections import defaultdict
from math import floor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True, help='directory with .npy skeleton files')
    p.add_argument('--labels', required=True, help='labels.json file (map id->text)')
    p.add_argument('--out-dir', required=True, help='where to write train/val/test split jsons')
    p.add_argument('--train', type=float, default=0.8)
    p.add_argument('--val', type=float, default=0.1)
    p.add_argument('--test', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def load_labels(path):
    with open(path, 'r', encoding='utf8') as f:
        labels = json.load(f)
    # Normalize keys to zero-padded 3-digit strings (e.g. "1" -> "001")
    normalized = {}
    for k,v in labels.items():
        try:
            ik = int(k)
            nk = f"{ik:03d}"
        except:
            nk = str(k).zfill(3)
        normalized[nk] = v
    return normalized

def scan_samples(data_root: Path):
    files = sorted(data_root.rglob('*.npy'))
    samples = []
    for p in files:
        name = p.stem  # e.g. 002_001_003
        parts = name.split('_')
        if len(parts) < 3:
            # skip unexpected filenames
            continue
        sign_id, signer_id, sample_id = parts[0], parts[1], parts[2]
        entry = {
            'id': name,
            'sign_id': sign_id,
            'signer_id': signer_id,
            'sample_id': sample_id,
            'skeleton_path': str(p.resolve())
        }
        samples.append(entry)
    return samples

def stratified_split(samples, train_frac, val_frac, test_frac, seed=123):
    # Group by sign_id
    by_label = defaultdict(list)
    for s in samples:
        by_label[s['sign_id']].append(s)
    random.seed(seed)
    train, val, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(floor(n * train_frac))
        n_val = int(floor(n * val_frac))
        # ensure minimal coverage: at least 1 sample goes somewhere if available
        if n_train + n_val >= n:
            # ensure each split gets at least 1 until exhausted
            n_train = max(1, min(n-2, n_train)) if n>=3 else max(0, n-1)
            n_val = max(1, min(n-1-n_train, n_val)) if (n - n_train) >= 2 else max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        # fallback distribution if negative
        if n_test < 0:
            n_test = 0
            if n_val > 0:
                n_val -= 1
            elif n_train > 0:
                n_train -= 1
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train+n_val])
        test.extend(items[n_train+n_val:])
    # final shuffle the groups (not necessary but cleaner)
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test

def backup_if_exists(path):
    p = Path(path)
    if p.exists():
        backup = p.with_suffix(p.suffix + '.bak')
        shutil.copy2(p, backup)
        print(f"Backed up {p} -> {backup}")

def annotate_with_text(samples, labels_map):
    for s in samples:
        sign = s['sign_id']
        s['sign_text'] = labels_map.get(sign, labels_map.get(str(int(sign)) if sign.isdigit() else sign, "UNKNOWN"))
    return samples

def write_json(path, samples):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_map = load_labels(args.labels)
    print(f"Loaded {len(labels_map)} labels (normalized keys sample): {list(labels_map.keys())[:6]}")

    samples = scan_samples(data_root)
    print("Found sample files:", len(samples))
    if len(samples) == 0:
        print("No .npy files found under", data_root)
        return

    # Compute fractions normalized
    total = args.train + args.val + args.test
    train_frac = args.train / total
    val_frac = args.val / total
    test_frac = args.test / total

    train, val, test = stratified_split(samples, train_frac, val_frac, test_frac, seed=args.seed)

    print("Split counts: train=%d  val=%d  test=%d" % (len(train), len(val), len(test)))

    # annotate with human label text
    train = annotate_with_text(train, labels_map)
    val = annotate_with_text(val, labels_map)
    test = annotate_with_text(test, labels_map)

    # backup existing files
    for name in ['train_split.json', 'val_split.json', 'test_split.json']:
        p = out_dir / name
        if p.exists():
            backup_if_exists(p)

    # write new splits
    write_json(out_dir / 'train_split.json', train)
    write_json(out_dir / 'val_split.json', val)
    write_json(out_dir / 'test_split.json', test)

    # also write a consolidated "all_samples.json" for convenience
    write_json(out_dir / 'all_samples_with_labels.json', train + val + test)

    print("Wrote new split files to", out_dir)
    # show a small per-class distribution summary
    from collections import Counter
    def counts_by_label(lst):
        return Counter([s['sign_id'] for s in lst])
    print("Examples: train per-label counts (first 10):", list(counts_by_label(train).items())[:10])

if __name__=='__main__':
    main()
