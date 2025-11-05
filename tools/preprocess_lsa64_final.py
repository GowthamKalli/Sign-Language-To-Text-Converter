import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
import multiprocessing
from scipy.interpolate import interp1d
import warnings

# ============================================================
# CONFIG
# ============================================================
VIDEO_DIR = r"E:\Sign Language to Text\data\lsa64_raw\all"
OUTPUT_DIR = r"E:\Sign Language to Text\data\skeletons_lsa64_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FRAMES = 100
SMOOTHING = True

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# FUNCTIONS
# ============================================================
def extract_keypoints_from_video(video_path):
    """Extract normalized skeleton (pose + hands) from a video."""
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        hands_results = hands.process(frame_rgb)

        keypoints = []

        # Pose keypoints (33 landmarks)
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0, 0, 0] * 33)

        # Hand keypoints (21 each hand)
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                label = handedness.classification[0].label.lower()
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                if label == "left":
                    left_hand = coords
                else:
                    right_hand = coords

        keypoints.extend(left_hand.flatten())
        keypoints.extend(right_hand.flatten())

        frames.append(keypoints)

    cap.release()
    pose.close()
    hands.close()

    if not frames:
        return np.zeros((TARGET_FRAMES, 225))

    keypoints = np.array(frames)

    if SMOOTHING and len(keypoints) > 5:
        keypoints = smooth_keypoints(keypoints)

    keypoints = resample_keypoints(keypoints, TARGET_FRAMES)
    return keypoints


def smooth_keypoints(kp_array, window=3):
    smoothed = np.copy(kp_array)
    for i in range(kp_array.shape[1]):
        smoothed[:, i] = np.convolve(kp_array[:, i], np.ones(window)/window, mode='same')
    return smoothed


def resample_keypoints(kp_array, num_frames):
    original_frames = np.arange(kp_array.shape[0])
    target_frames = np.linspace(0, kp_array.shape[0] - 1, num_frames)
    interpolator = interp1d(original_frames, kp_array, axis=0, fill_value="extrapolate")
    return interpolator(target_frames)


def process_video(video_file):
    """Process one video file to extract skeleton data."""
    try:
        base_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(OUTPUT_DIR, base_name + ".npy")

        if os.path.exists(output_path):
            return None  # already processed

        video_path = os.path.join(VIDEO_DIR, video_file)
        skeleton = extract_keypoints_from_video(video_path)
        np.save(output_path, skeleton)
        return True
    except Exception as e:
        print(f"⚠️ Error processing {video_file}: {e}")
        return None


# ============================================================
# MAIN
# ============================================================
def main():
    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi"))]
    processed = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")}
    remaining = [f for f in all_videos if os.path.splitext(f)[0] not in processed]

    print(f"Found {len(all_videos)} total videos.")
    print(f"✅ {len(processed)} already processed.")
    print(f"▶️ {len(remaining)} remaining to process.\n")

    if not remaining:
        print("🎉 All videos already processed!")
        return

    num_workers = min(15, max(1, multiprocessing.cpu_count() - 1))
    print(f"Using {num_workers} workers | Target frames: {TARGET_FRAMES} | Smoothing: {SMOOTHING}")
    print("Suppressing all MediaPipe logs; progress bar will show processing status.\n")

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, remaining), total=len(remaining), desc="Processing Skeletons"))

    print("✅ All skeletons extracted successfully!")


if __name__ == "__main__":
    main()
