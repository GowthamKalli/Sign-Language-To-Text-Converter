import os
import json
import numpy as np
from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_split, input_type='skeleton', debug=False):
        self.data_split = data_split
        self.input_type = input_type
        self.debug = debug

        with open(data_split, 'r') as f:
            self.samples = json.load(f)

        self.data, self.label = self.load_data()

        if self.debug:
            print(f"✅ Loaded {len(self.data)} valid samples from {data_split}")

    def extract_label_from_filename(self, path):
        """Extracts the sign_id (label) from filename format <signer>_<sign>_<sample>.npy"""
        fname = os.path.basename(path)
        parts = fname.replace('.npy', '').split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {fname}")
        return int(parts[1]) - 1  # convert sign_id (1–64) → 0–63

    def load_data(self):
        data = []
        labels = []

        for item in self.samples:
            try:
                # Handle both dict or plain string entries
                if isinstance(item, dict):
                    skeleton_path = item.get('skeleton_path', None)
                else:
                    skeleton_path = item

                if skeleton_path is None:
                    continue

                label = self.extract_label_from_filename(skeleton_path)
                skel = np.load(skeleton_path)
                skel = self.preprocess_skeleton(skel)

                data.append(skel)
                labels.append(label)

            except Exception as e:
                print(f"⚠️ Error loading {skeleton_path}: {e}")

        return data, labels

    def preprocess_skeleton(self, skel):
        # Ensure shape (3, T, V, 1)
        if skel.ndim == 3:
            if skel.shape[-1] == 3:
                skel = skel.transpose(2, 0, 1)
            elif skel.shape[1] == 3:
                skel = skel.transpose(1, 2, 0)
            skel = np.expand_dims(skel, axis=-1)
        elif skel.ndim == 2 and skel.shape[1] % 3 == 0:
            T = skel.shape[0]
            V = skel.shape[1] // 3
            skel = skel.reshape(T, V, 3).transpose(2, 0, 1)
            skel = np.expand_dims(skel, axis=-1)
        elif skel.ndim == 4 and skel.shape[0] == 3:
            pass
        else:
            raise ValueError(f"❌ Unsupported skeleton shape: {skel.shape}")

        return skel.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
