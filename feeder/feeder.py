import os
import json
import numpy as np
from torch.utils.data import Dataset

class Feeder(Dataset):
    """
    Robust Feeder that accepts optional label_map/index_to_text and extra kwargs.
    Usage:
      Feeder(data_split="path/to/train_split.json", input_type='skeleton', debug=False, label_map=label_map_dict)
    """

    def __init__(self, data_split, input_type='skeleton', debug=False, label_map=None, index_to_text=None, **kwargs):
        """
        Accepts extra kw args and ignores them (so processor can pass label_map etc).
        label_map: dict mapping sign_key string (e.g. "001") -> index (0..N-1)
        index_to_text: optional mapping index->human text (not required)
        """
        self.data_split = data_split
        self.input_type = input_type
        self.debug = debug
        self.label_map = label_map  # may be None
        self.index_to_text = index_to_text

        # Load samples
        with open(data_split, 'r', encoding='utf8') as f:
            self.samples = json.load(f)

        # Preload file paths + labels
        self.data, self.label = self.load_data()

        if self.debug:
            print(f"✅ Loaded {len(self.data)} valid samples from {data_split}")

    def normalize_sign_key(self, sign_key):
        """Ensure sign_key is zero-padded 3-char string like '001'"""
        if sign_key is None:
            return None
        # if already string '001' or '1'
        s = str(sign_key)
        # if it looks like '002' keep; else pad
        if len(s) >= 3:
            return s.zfill(3)
        return s.zfill(3)

    def extract_label_from_filename(self, path):
        """Extracts the sign_id (label) from filename format <sign>_<signer>_<sample>.npy or <sign>_<signer>_<sample> (user uses <sign>_<signer>_<sample>)"""
        fname = os.path.basename(path)
        # remove extension
        base = fname.replace('.npy', '').replace('.npz', '')
        parts = base.split('_')
        # Expected formats:
        #   sign_singer_sample  or  sign_signer_sample (you have fields like 002_001_001)
        # Identify the part that corresponds to sign_id:
        # Many datasets use <SIGN>_<SIGNER>_<SAMPLE>, where sign is first
        if len(parts) >= 3:
            sign_str = parts[0]
        elif len(parts) == 1:
            # fallback if filename is just sign id
            sign_str = parts[0]
        else:
            raise ValueError(f"Invalid filename format: {fname}")
        # Normalize to '003' etc
        sign_norm = self.normalize_sign_key(sign_str)
        # If label_map present, map to index; else int-1
        if self.label_map:
            # label_map maps '001' -> 0
            if sign_norm in self.label_map:
                return int(self.label_map[sign_norm])
            else:
                # fallback: try int conversion
                try:
                    return int(sign_norm) - 1
                except:
                    raise ValueError(f"Sign key {sign_norm} not found in label_map and not int-convertible")
        else:
            # convert to 0-index int
            try:
                return int(sign_norm) - 1
            except:
                raise ValueError(f"Couldn't parse sign id from filename: {fname}")

    def load_data(self):
        data = []
        labels = []

        for item in self.samples:
            try:
                # Accept either dict entries (with skeleton_path, sign_id) or plain string path
                if isinstance(item, dict):
                    skeleton_path = item.get('skeleton_path') or item.get('path') or item.get('file') or None
                    sign_id_field = item.get('sign_id') or item.get('label') or item.get('class') or None
                else:
                    skeleton_path = item
                    sign_id_field = None

                if skeleton_path is None:
                    # skip bad entry
                    if self.debug:
                        print("⚠️ Skipping sample with no skeleton_path:", item)
                    continue

                if not os.path.exists(skeleton_path):
                    # Try relative path (sometimes splits stored relative to repo)
                    alt = os.path.join(os.path.dirname(self.data_split), os.path.basename(skeleton_path))
                    if os.path.exists(alt):
                        skeleton_path = alt
                    else:
                        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")

                # Determine label index
                label_idx = None
                if sign_id_field is not None:
                    # normalize sign_id and map
                    sign_key = self.normalize_sign_key(sign_id_field)
                    if self.label_map and sign_key in self.label_map:
                        label_idx = int(self.label_map[sign_key])
                    else:
                        # if sign_id_field is numeric string, convert to int-1
                        try:
                            label_idx = int(str(sign_id_field)) - 1
                        except:
                            # fallback to filename parsing
                            label_idx = None

                if label_idx is None:
                    # Extract from filename
                    label_idx = self.extract_label_from_filename(skeleton_path)

                # load skeleton array and preprocess to model expected format
                skel = np.load(skeleton_path, allow_pickle=True)
                skel = self.preprocess_skeleton(skel)

                data.append(skel)
                labels.append(int(label_idx))

            except Exception as e:
                # keep going, but log
                print(f"⚠️ Error loading {skeleton_path}: {e}")

        return data, labels

    def preprocess_skeleton(self, skel):
        """
        Normalize skeleton shapes to (3, T, V, 1) float32 (or (C, T, V) w/ last dim added later).
        Handles common shapes:
         - (T, V, C)  -> transpose to (C, T, V)
         - (V, T, C)  -> transpose to (C, T, V)
         - (3, T, V)  -> ok, will add channel dim
         - (C, T, V, 1) -> ok
         - (T, V*3) -> reshape
        """
        arr = np.asarray(skel)
        if arr.ndim == 3:
            # common case: (T, V, C) where C==3  -> (C, T, V)
            T, V, C = arr.shape
            if C == 3:
                out = arr.transpose(2, 0, 1)
            else:
                # maybe (C, T, V)
                if arr.shape[0] == 3:
                    out = arr
                else:
                    # fallback: try (V, T, C)
                    out = arr.transpose(2, 1, 0)
        elif arr.ndim == 2:
            # maybe flattened (T, V*3)
            T = arr.shape[0]
            if arr.shape[1] % 3 == 0:
                V = arr.shape[1] // 3
                out = arr.reshape(T, V, 3).transpose(2, 0, 1)
            elif arr.size == 3 * 100 * 75:
                # fallback reshape used previously
                out = arr.reshape(3, 100, 75)
            else:
                raise ValueError(f"Unexpected 2D skeleton shape: {arr.shape}")
        elif arr.ndim == 4:
            # maybe already (C, T, V, 1)
            if arr.shape[0] == 3:
                out = arr[..., 0] if arr.shape[-1] == 1 else arr
            else:
                raise ValueError(f"Unexpected 4D skeleton shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported skeleton numpy ndim={arr.ndim}, shape={arr.shape}")

        # Ensure final shape is (3, T, V). We'll not add batch dim here.
        out = out.astype(np.float32)
        # ensure last channel dimension for some data loaders expects (C,T,V,1)
        if out.ndim == 3:
            out = np.expand_dims(out, axis=-1)  # (C,T,V,1)

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return (skeleton, label) where skeleton is np array
        return self.data[index], self.label[index]
