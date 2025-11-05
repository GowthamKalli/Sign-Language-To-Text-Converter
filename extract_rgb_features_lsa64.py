import os
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing

# ============================================================
# CONFIG
# ============================================================
VIDEO_DIR = r"E:\Sign Language to Text\data\lsa64_raw\all"
SKELETON_DIR = r"E:\Sign Language to Text\data\skeletons_lsa64"
OUTPUT_DIR = r"E:\Sign Language to Text\data\lsa64_rgb_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================
def extract_rgb_mean(video_path):
    """Extract mean RGB values across frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    rgb_means = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(frame, axis=(0, 1))
        rgb_means.append(mean_rgb)
    cap.release()
    return np.mean(rgb_means, axis=0) if rgb_means else np.zeros(3)


def process_video(video_file):
    """Process one video and save its RGB features."""
    try:
        base_name = os.path.splitext(video_file)[0]
        skeleton_path = os.path.join(SKELETON_DIR, base_name + ".npy")
        output_path = os.path.join(OUTPUT_DIR, base_name + "_rgb.npy")

        # Skip if already processed or skeleton missing
        if not os.path.exists(skeleton_path) or os.path.exists(output_path):
            return None

        video_path = os.path.join(VIDEO_DIR, video_file)
        rgb_features = extract_rgb_mean(video_path)
        np.save(output_path, rgb_features)
        return True
    except Exception as e:
        print(f"⚠️ Error processing {video_file}: {e}")
        return None


def main():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4") or f.endswith(".avi")]
    matching_videos = [f for f in video_files if os.path.exists(os.path.join(SKELETON_DIR, os.path.splitext(f)[0] + ".npy"))]

    print(f"Found {len(matching_videos)} matching videos.")
    if not matching_videos:
        return

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} CPU workers...")

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, matching_videos), total=len(matching_videos), desc="Extracting RGB features"))

    print("✅ RGB feature extraction complete!")


if __name__ == "__main__":
    main()
