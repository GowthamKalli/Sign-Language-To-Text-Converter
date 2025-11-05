import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
SKELETONS_DIR = r"E:\Sign Language to Text\data\skeletons_lsa64_final"
OUTPUT_DIR = r"E:\Sign Language to Text\output_stick_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CONNECTIONS (75 joints: pose + hands)
# =========================
POSE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),
    (23,24),(23,25),(24,26),
    (25,27),(26,28),
    (27,29),(28,30),
    (29,31),(30,32)
]
LEFT_HAND_START, RIGHT_HAND_START = 33, 54
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
EDGES = POSE_EDGES + \
        [(a+LEFT_HAND_START, b+LEFT_HAND_START) for a,b in HAND_EDGES] + \
        [(a+RIGHT_HAND_START, b+RIGHT_HAND_START) for a,b in HAND_EDGES]

# =========================
# DRAW FUNCTIONS
# =========================
def draw_skeleton(frame, joints):
    """Draw stick figure lines + points."""
    for (i, j) in EDGES:
        if i < len(joints) and j < len(joints):
            p1 = tuple(np.int32(joints[i, :2]))
            p2 = tuple(np.int32(joints[j, :2]))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
    for idx, (x, y, _) in enumerate(joints):
        color = (255, 255, 255)
        if idx >= RIGHT_HAND_START:
            color = (0, 0, 255)  # right hand = red
        elif idx >= LEFT_HAND_START:
            color = (255, 0, 0)  # left hand = blue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

# =========================
# ANALYSIS FUNCTIONS
# =========================
def compute_stats(data):
    """Compute per-sample skeleton statistics."""
    valid_mask = np.all(data != 0, axis=-1)
    valid_joints_pct = np.mean(valid_mask) * 100

    motion = np.linalg.norm(np.diff(data[:, :, :2], axis=0), axis=-1)
    avg_motion = np.mean(motion)
    motion_std = np.std(motion)

    frozen_frames = np.sum(np.mean(motion, axis=1) < 1e-3)
    frozen_ratio = (frozen_frames / len(data)) * 100

    # Hand jitter detection
    left_hand = data[:, LEFT_HAND_START:RIGHT_HAND_START, :2]
    right_hand = data[:, RIGHT_HAND_START:, :2]
    lh_motion = np.linalg.norm(np.diff(left_hand, axis=0), axis=-1)
    rh_motion = np.linalg.norm(np.diff(right_hand, axis=0), axis=-1)
    jitter_frames = np.sum((np.mean(lh_motion, axis=1) > 15) | (np.mean(rh_motion, axis=1) > 15))
    jitter_ratio = (jitter_frames / (len(data) - 1)) * 100

    return {
        "valid_joints_%": valid_joints_pct,
        "avg_motion": avg_motion,
        "motion_std": motion_std,
        "frozen_ratio_%": frozen_ratio,
        "hand_jitter_%": jitter_ratio
    }

# =========================
# MAIN VISUALIZATION
# =========================
def visualize_sample(np_file):
    data = np.load(np_file)
    filename = os.path.basename(np_file).replace(".npy", "")
    out_path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")

    # Automatically reshape if 2D
    if data.ndim == 2 and data.shape[1] == 225:
        data = data.reshape((data.shape[0], 75, 3))
    elif data.ndim != 3:
        print(f"⚠️ Skipping {filename}: invalid shape {data.shape}")
        return

    stats = compute_stats(data)
    print(f"\n📊 Stats for {filename}:")
    for k, v in stats.items():
        print(f"  {k:<18}: {v:.2f}")

    T, V, C = data.shape
    H, W = 480, 640
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (W, H))

    # Normalize skeletons roughly to canvas
    x = data[:, :, 0]
    y = data[:, :, 1]
    if np.max(x) < 2 and np.max(y) < 2:
        data[:, :, 0] *= W
        data[:, :, 1] *= H
    else:
        data[:, :, 0] = (x - np.min(x)) / (np.max(x) - np.min(x)) * W
        data[:, :, 1] = (y - np.min(y)) / (np.max(y) - np.min(y)) * H

    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_skeleton(frame, data[t])
        video.write(frame)

    video.release()
    print(f"✅ Saved stick-figure: {out_path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    npy_files = [f for f in os.listdir(SKELETONS_DIR) if f.endswith(".npy")]
    np.random.shuffle(npy_files)
    npy_files = npy_files[:5]  # visualize 5 samples

    for f in tqdm(npy_files, desc="Generating skeleton animations"):
        visualize_sample(os.path.join(SKELETONS_DIR, f))

    print(f"\n🎬 Stick-figure videos + stats saved in: {OUTPUT_DIR}")
