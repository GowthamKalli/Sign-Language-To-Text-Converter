import os, sys, json, argparse, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
import mediapipe as mp

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser(description="Live demo - FIXED preprocessing")
parser.add_argument('--project-root', default=r"E:\SL Final")
parser.add_argument('--model-path', default=None)
parser.add_argument('--labels-path', default=None)
parser.add_argument('--device', default='cuda')
parser.add_argument('--capture-frames', type=int, default=60)
parser.add_argument('--target-frames', type=int, default=100)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.10)
parser.add_argument('--consensus-window', type=int, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

PR = Path(args.project_root).resolve()
STGCN = PR / 'st-gcn-sl'
if str(STGCN) not in sys.path:
    sys.path.insert(0, str(STGCN))

from net.st_gcn import Model

# Paths
DEFAULT_MODEL = STGCN / 'work_dir' / 'lsa64_stgat' / 'epoch46_model.pt'
DEFAULT_LABELS = STGCN / 'data_splits' / 'index_to_text.json'
MODEL_PATH = Path(args.model_path) if args.model_path else DEFAULT_MODEL
LABELS_PATH = Path(args.labels_path) if args.labels_path else DEFAULT_LABELS

DEVICE = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
CAPTURE_FRAMES = args.capture_frames
TARGET_FRAMES = args.target_frames
TOPK = args.topk
CONF_THRESHOLD = args.threshold
CONSENSUS_WINDOW = max(1, args.consensus_window)
DEBUG = args.debug

IN_CHANNELS = 3
NUM_JOINTS = 75
NUM_CLASSES = 64

# load labels
def load_labels(p):
    if not p.exists():
        return {}
    try:
        return json.load(open(p, 'r', encoding='utf8'))
    except:
        return {}

labels = load_labels(LABELS_PATH)

# -----------------------
# PREPROCESS helpers - MATCHING TRAINING EXACTLY
# -----------------------
def extract_keypoints_225_from_mediapipe_results(pose_res, hands_res):
    """Extract keypoints EXACTLY like training preprocessing script."""
    keypoints = []
    
    # Pose keypoints (33 landmarks × 3 = 99 values)
    if pose_res and getattr(pose_res, "pose_landmarks", None):
        for lm in pose_res.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0, 0.0, 0.0] * 33)

    # Hand keypoints (21 × 3 each = 63 values per hand)
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    
    if hands_res and getattr(hands_res, "multi_hand_landmarks", None) and getattr(hands_res, "multi_handedness", None):
        for hand_lm, hand_label in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = hand_label.classification[0].label.lower()
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
            if label == "left":
                left_hand = coords
            else:
                right_hand = coords
    
    keypoints.extend(left_hand.flatten().tolist())
    keypoints.extend(right_hand.flatten().tolist())
    
    return np.array(keypoints, dtype=np.float32)

def smooth_keypoints(kp_array, window=3):
    """Smooth keypoints - MATCHING TRAINING."""
    if kp_array.shape[0] <= 1:
        return kp_array
    smoothed = np.copy(kp_array)
    for i in range(kp_array.shape[1]):
        smoothed[:, i] = np.convolve(kp_array[:, i], np.ones(window)/window, mode='same')
    return smoothed

def resample_keypoints(kp_array, num_frames):
    """Resample to target frames - MATCHING TRAINING."""
    if kp_array.shape[0] == num_frames:
        return kp_array.copy()
    original_frames = np.arange(kp_array.shape[0])
    target_frames = np.linspace(0, kp_array.shape[0] - 1, num_frames)
    # Use same interpolation as training
    interp = interp1d(original_frames, kp_array, axis=0, fill_value="extrapolate")
    return interp(target_frames)

def prepare_model_input(frames_array):
    """
    Convert (T=100, 225) to model input (1, 3, 100, 75, 1)
    MATCHING the Feeder preprocessing.
    """
    assert frames_array.shape == (TARGET_FRAMES, 3 * NUM_JOINTS), f"Expected shape ({TARGET_FRAMES}, {3*NUM_JOINTS}), got {frames_array.shape}"
    
    # Reshape to (T, V, C) then transpose to (C, T, V)
    data = frames_array.reshape(TARGET_FRAMES, NUM_JOINTS, 3)  # (100, 75, 3)
    data = data.transpose(2, 0, 1)  # (3, 100, 75)
    data = np.expand_dims(data, axis=(0, -1))  # (1, 3, 100, 75, 1)
    
    return torch.tensor(data, dtype=torch.float32, device=DEVICE)

# -----------------------
# Model load
# -----------------------
def load_model(path):
    print("📄 Loading model ...", path)
    model = Model(
        in_channels=IN_CHANNELS, 
        num_class=NUM_CLASSES, 
        graph_args={'layout':'mediapipe','strategy':'spatial'}, 
        edge_importance_weighting=True
    )
    if not Path(path).exists():
        raise FileNotFoundError(path)
    ck = torch.load(str(path), map_location='cpu')
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    if isinstance(ck, dict):
        ck = {k.replace('module.', ''): v for k, v in ck.items()}
    model.load_state_dict(ck, strict=False)
    model.to(DEVICE).eval()
    print("✅ Model loaded on", DEVICE)
    return model

# -----------------------
# MAIN
# -----------------------
def main():
    model = load_model(MODEL_PATH)

    # mediapipe setup
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam not accessible")
    cap.set(cv2.CAP_PROP_FPS, 30)

    window = deque(maxlen=CAPTURE_FRAMES)
    pred_buffer = deque(maxlen=10)
    recent_preds = deque(maxlen=CONSENSUS_WINDOW)
    recording = False

    print("🎥 Live demo running. Controls: SPACE to start/stop, Q to quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)

            # Draw landmarks
            if getattr(pose_res, 'pose_landmarks', None):
                mp_drawing.draw_landmarks(
                    frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
            if getattr(hands_res, 'multi_hand_landmarks', None):
                for hlm in hands_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # Extract keypoints
            kp = extract_keypoints_225_from_mediapipe_results(pose_res, hands_res)
            detection_quality = np.count_nonzero(kp) / kp.size

            if recording:
                window.append(kp)
                # Draw progress bar
                progress = len(window) / CAPTURE_FRAMES
                w, h = 400, 28
                bx = (frame.shape[1] - w) // 2
                by = frame.shape[0] - 60
                cv2.rectangle(frame, (bx, by), (bx + w, by + h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(w * progress), by + h), (0, 255, 0), -1)
                cv2.putText(
                    frame, f"Recording: {len(window)}/{CAPTURE_FRAMES}", 
                    (bx + 10, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
                )
                cv2.circle(frame, (30, 30), 10, (0,0,255), -1)
                cv2.putText(frame, "REC", (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                if len(window) >= CAPTURE_FRAMES:
                    recording = False
                    
                    # PREPROCESSING - MATCHING TRAINING EXACTLY
                    frames_array = np.array(list(window), dtype=np.float32)
                    
                    if DEBUG:
                        print("\n>>> RAW CAPTURE:")
                        print(f"  Shape: {frames_array.shape}")
                        print(f"  Mean: {frames_array.mean():.6f}, Std: {frames_array.std():.6f}")
                        print(f"  Range: [{frames_array.min():.6f}, {frames_array.max():.6f}]")
                    
                    # Step 1: Smooth (if enough frames)
                    if frames_array.shape[0] > 5:
                        frames_array = smooth_keypoints(frames_array, window=3)
                    
                    # Step 2: Resample to TARGET_FRAMES
                    frames_array = resample_keypoints(frames_array, TARGET_FRAMES)
                    
                    # ⚠️ NO OTHER TRANSFORMS! Training data has none!
                    # No root subtraction, no hand swapping, no flip, no normalization, no clipping
                    
                    if DEBUG:
                        print(">>> AFTER PREPROCESSING:")
                        print(f"  Shape: {frames_array.shape}")
                        print(f"  Mean: {frames_array.mean():.6f}, Std: {frames_array.std():.6f}")
                        print(f"  Range: [{frames_array.min():.6f}, {frames_array.max():.6f}]")
                        print(f"  First 20 values: {frames_array.flatten()[:20]}")
                    
                    # Prepare model input
                    tensor = prepare_model_input(frames_array)
                    
                    # Inference
                    with torch.no_grad():
                        out = model(tensor)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
                        logits = out.cpu().numpy()[0]
                    
                    topk_idx = np.argsort(probs)[::-1][:TOPK]
                    topk_vals = probs[topk_idx]

                    print("\n🔮 Predictions:")
                    print("\n🔮 Predictions:")
                    for i, (idx, val) in enumerate(zip(topk_idx, topk_vals), 1):
                        idx = int(idx)
                        label_num = idx + 1  # human-friendly 1-based label number
                        # Handle both 0-indexed and 1-indexed label maps
                        name = labels.get(str(idx),
                                        labels.get(str(label_num).zfill(3),
                                                    f"Class_{idx}"))
                        print(f"  {i}. [{label_num}] {name} (idx={idx}) prob={val:.4f} logit={logits[idx]:.4f}")


                    # Consensus logic
                    accepted = False
                    top1_idx = int(topk_idx[0])
                    top1_conf = float(topk_vals[0])
                    
                    if top1_conf >= CONF_THRESHOLD:
                        recent_preds.append(top1_idx)
                        counts = {}
                        for r in recent_preds:
                            counts[r] = counts.get(r, 0) + 1
                        majority_idx, majority_count = max(counts.items(), key=lambda kv: kv[1])
                        if majority_count >= (CONSENSUS_WINDOW // 2) + 1 and majority_idx == top1_idx:
                            accepted = True
                    
                    if accepted:
                        label_name = labels.get(str(int(top1_idx)), 
                                            labels.get(str(int(top1_idx)+1).zfill(3), 
                                                        f"Class_{top1_idx}"))
                        pred_buffer.append((label_name, top1_conf))
                        print(f"✅ ACCEPTED: {label_name} | Confidence: {top1_conf:.4f} ({top1_conf*100:.2f}%)")
                    else:
                        label_name = labels.get(str(int(top1_idx)), 
                                            labels.get(str(int(top1_idx)+1).zfill(3), 
                                                        f"Class_{top1_idx}"))
                        print(f"❌ REJECTED: {label_name} | Confidence: {top1_conf:.4f} ({top1_conf*100:.2f}%) — below threshold ({CONF_THRESHOLD:.2f}) or no consensus")

                    window.clear()

            else:
                # Not recording - show status
                status_color = (0,255,0) if detection_quality > 0.12 else (0,165,255)
                status_text = "Ready - Press SPACE" if detection_quality > 0.12 else "Position hands in frame"
                cv2.putText(frame, status_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                
                if pred_buffer:
                    last_label, last_conf = pred_buffer[-1]

                    # Choose a bright color that stands out against white background
                    text_color = (0, 180, 255)   # Orange
                    shadow_color = (0, 0, 0)     # Black outline for contrast
                    text_pos = (30, 100)
                    text = f"Detected: {last_label} ({last_conf:.1%})"

                    # Draw shadow (outline)
                    cv2.putText(frame, text, (text_pos[0]+2, text_pos[1]+2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, shadow_color, 3)
                    # Draw main text
                    cv2.putText(frame, text, text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            # Draw detection quality indicator
            qc = (0,255,0) if detection_quality > 0.2 else (0,165,255) if detection_quality > 0.1 else (0,0,255)
            cv2.circle(frame, (frame.shape[1]-30, 30), 10, qc, -1)
            
            cv2.imshow("Sign Language Recognition - FIXED", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                recording = not recording
                window.clear()
                recent_preds.clear()

                if recording:
                    print("⏳ Get ready... recording will start in 3 seconds")
                    cv2.putText(frame, "Get ready...", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Sign Language Recognition - FIXED", frame)
                    cv2.waitKey(1)  # Refresh frame
                    time.sleep(3)  # ⏱️ Delay before starting capture
                    print("🔴 Recording started...")
                else:
                    print("⏸️ Recording stopped")

                    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()
        print("✅ Demo closed.")

if __name__ == "__main__":
    main()