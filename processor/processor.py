#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live Sign Language Recognition Demo (Feeder-consistent preprocessing + BN-batch fallback + warm-up calibration)
One-click updated script — Jefflin
"""

import os
import sys
import json
import time
from collections import deque

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
import mediapipe as mp

# -----------------------
# CONFIG (adjust paths if needed)
# -----------------------
PROJECT_ROOT = r"C:\Users\jeffl\OneDrive\Desktop\Sign Language to Text"
MODEL_PATH   = os.path.join(PROJECT_ROOT, "st-gcn-sl", "work_dir", "lsa64_stgat", "epoch45_manualsave_model.pt")
LABELS_PATH  = os.path.join(PROJECT_ROOT, "st-gcn-sl", "data_splits", "index_to_text.json")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FRAMES = 100      # model expected temporal length (same as Feeder)
NUM_JOINTS   = 75        # 225 values / 3 channels
IN_CHANNELS  = 3
NUM_CLASSES  = 64
CAPTURE_FRAMES = 60      # ~2 seconds capture @30fps (will be resampled to TARGET_FRAMES)
MIN_DETECTION_QUALITY = 0.1
TOPK = 3
EPS = 1e-6

# Warmup calibration settings (automatic)
WARMUP_ENABLED = True
WARMUP_SAMPLES = 6        # number of short clips to collect and feed to BN as warmup
WARMUP_GAP_SEC = 0.5      # seconds between sample captures (UI/prompt breathing room)

# BN-batch fallback during single-gesture forward (set to True recommended)
USE_BN_BATCH_FALLBACK = True

# -----------------------
# Add project module path
# -----------------------
STGCN_PATH = os.path.join(PROJECT_ROOT, "st-gcn-sl")
if STGCN_PATH not in sys.path:
    sys.path.insert(0, STGCN_PATH)

try:
    from net.st_gcn import Model
except Exception as e:
    raise ImportError(
        f"❌ Could not import Model from net.st_gcn.\nCheck your path: {STGCN_PATH}"
    ) from e

# -----------------------
# Utilities (match Feeder preprocessing)
# -----------------------
def extract_keypoints_225_from_mediapipe_results(pose_res, hands_res):
    """Extract 33 pose + 21 left hand + 21 right hand → (225,) float32"""
    keypoints = []

    # Pose landmarks
    if pose_res and pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks.landmark:
            # normalized coords in MediaPipe
            keypoints.extend([lm.x or 0.0, lm.y or 0.0, lm.z or 0.0])
    else:
        keypoints.extend([0.0, 0.0, 0.0] * 33)

    # Hands
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if hands_res and getattr(hands_res, "multi_hand_landmarks", None) and getattr(hands_res, "multi_handedness", None):
        for hand_lm, hand_label in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = hand_label.classification[0].label.lower()
            coords = np.array([[lm.x or 0.0, lm.y or 0.0, lm.z or 0.0] for lm in hand_lm.landmark], dtype=np.float32)
            if label == "left":
                left_hand = coords
            else:
                right_hand = coords

    keypoints.extend(left_hand.flatten().tolist())
    keypoints.extend(right_hand.flatten().tolist())
    return np.array(keypoints, dtype=np.float32)


def smooth_keypoints(kp_array, window=3):
    """3-frame moving average (Feeder). kp_array shape (T, 225)."""
    if kp_array.shape[0] <= 1:
        return kp_array
    smoothed = np.copy(kp_array)
    for i in range(kp_array.shape[1]):
        smoothed[:, i] = np.convolve(kp_array[:, i], np.ones(window)/window, mode='same')
    return smoothed


def resample_keypoints(kp_array, num_frames):
    """Temporal interpolation (Feeder)."""
    if kp_array.shape[0] == num_frames:
        return kp_array.copy()
    original_frames = np.arange(kp_array.shape[0])
    target_frames = np.linspace(0, kp_array.shape[0] - 1, num_frames)
    interp = interp1d(original_frames, kp_array, axis=0, fill_value="extrapolate")
    return interp(target_frames)


def prepare_model_input(frames_array):
    """Convert (T,225) → (1,3,T,75,1) tensor for model."""
    data = frames_array.T.reshape(IN_CHANNELS, TARGET_FRAMES, NUM_JOINTS)  # (3, T, V)
    data = np.expand_dims(data, axis=(0, -1))  # (1, 3, T, V, 1)
    return torch.tensor(data, dtype=torch.float32, device=DEVICE)


# -----------------------
# Label handling
# -----------------------
def load_labels(path):
    if not os.path.exists(path):
        print(f"⚠️ Labels file not found at {path}. Predictions will use indices.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_label(idx, labels_dict):
    s = str(int(idx))
    if s in labels_dict:
        return labels_dict[s]
    s2 = str(int(idx) + 1).zfill(3)
    if s2 in labels_dict:
        return labels_dict[s2]
    # fallback for index->text keyed with ints
    if int(idx) in labels_dict:
        return labels_dict[int(idx)]
    return f"Class_{idx}"


# -----------------------
# Model loading
# -----------------------
def load_model(path):
    print("📄 Loading model ...")
    model = Model(
        in_channels=IN_CHANNELS,
        num_class=NUM_CLASSES,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'},
        edge_importance_weighting=True
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found: {path}")
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE).eval()
    print(f"✅ Model loaded on {DEVICE} - torch {torch.__version__}")
    return model


# -----------------------
# BN helpers
# -----------------------
def print_bn_stats(model, n=12):
    if hasattr(model, 'data_bn'):
        rm = model.data_bn.running_mean.detach().cpu().numpy()
        rv = model.data_bn.running_var.detach().cpu().numpy()
        print(f">>> BN running_mean (first {n}): {np.round(rm[:n], 6).tolist()}")
        print(f">>> BN running_var  (first {n}): {np.round(rv[:n], 6).tolist()}")
    else:
        print(">>> Model has no data_bn attribute; cannot show BN stats.")


def bn_batch_forward(model, tensor):
    """
    Run model forward while forcing model.data_bn to use batch stats.
    Keeps model in eval() except data_bn temporarily in train().
    Returns model output tensor.
    """
    # keep track of original modes
    had_data_bn = hasattr(model, 'data_bn')
    was_model_training = model.training
    # Ensure model.eval() globally (we don't want dropout behaviour)
    model.eval()
    if had_data_bn:
        model.data_bn.train()  # use batch stats for this forward
    with torch.no_grad():
        out = model(tensor)
    # restore bn mode
    if had_data_bn:
        model.data_bn.eval()
    if was_model_training:
        model.train()
    return out


def warmup_calibration(model, warmup_tensors):
    """
    Run a small set of tensors through the model with data_bn.train() to update running stats.
    warmup_tensors: list of tensors shaped (1, C, T, V, 1)
    """
    if not hasattr(model, 'data_bn'):
        print(">>> Warmup skipped: model has no data_bn.")
        return

    print(f">>> Running warmup calibration on {len(warmup_tensors)} samples (data_bn.train()).")
    # set model to eval but data_bn to train to update running stats
    model.eval()
    model.data_bn.train()
    with torch.no_grad():
        for t in warmup_tensors:
            _ = model(t)  # forward will update running_mean/var due to BN in train mode
    model.data_bn.eval()
    print(">>> Warmup calibration finished.")


# -----------------------
# Warmup capture helper
# -----------------------
def collect_warmup_clips(cap, pose, hands, num_clips, clip_frames, show=True):
    """
    Automatically capture `num_clips` clips of `clip_frames` frames from webcam.
    Returns list of numpy arrays shaped (TARGET_FRAMES, 225) (already resampled).
    """
    collected = []
    print(f">>> Warmup capture: collecting {num_clips} clips of {clip_frames} frames each.")
    for ci in range(num_clips):
        print(f"  - Prepare for warmup clip {ci+1}/{num_clips} (starting in 0.8s)...")
        # small pause to let user adjust
        time.sleep(0.8)
        frames = []
        while len(frames) < clip_frames:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera read failed during warmup.")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)
            kp = extract_keypoints_225_from_mediapipe_results(pose_res, hands_res)
            frames.append(kp)
            if show:
                # simple overlay
                cv2.putText(frame, f"Warmup {ci+1}/{num_clips}  frames {len(frames)}/{clip_frames}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Warmup (press q to abort)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(">>> Warmup aborted by user.")
                    return collected
        arr = np.array(frames, dtype=np.float32)
        if arr.shape[0] > 5:
            arr = smooth_keypoints(arr)
        arr = resample_keypoints(arr, TARGET_FRAMES)
        collected.append(arr)
        print(f"    captured clip {ci+1} -> shape {arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")
        time.sleep(WARMUP_GAP_SEC)
    print(">>> Warmup capture done.")
    return collected


# -----------------------
# Main demo
# -----------------------
def main():
    labels = load_labels(LABELS_PATH)
    model = load_model(MODEL_PATH)

    print("\n>>> BN stats (before warmup):")
    print_bn_stats(model, n=12)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)

    # automatic warmup calibration (optional)
    if WARMUP_ENABLED and WARMUP_SAMPLES > 0:
        try:
            warmup_clips = collect_warmup_clips(cap, pose, hands, WARMUP_SAMPLES, CAPTURE_FRAMES, show=True)
            # convert to tensors (1,C,T,V,1)
            warmup_tensors = []
            for arr in warmup_clips:
                t = prepare_model_input(arr)
                warmup_tensors.append(t)
            if warmup_tensors:
                warmup_calibration(model, warmup_tensors)
                print("\n>>> BN stats (after warmup):")
                print_bn_stats(model, n=12)
            else:
                print(">>> No warmup clips collected — skipping BN warmup.")
        except Exception as e:
            print(">>> Warmup failed/aborted:", e)
            print(">>> Continuing without warmup...")

    window = deque(maxlen=CAPTURE_FRAMES)
    pred_buffer = deque(maxlen=5)
    recording = False

    print("\n🎥 Live Sign Language Recognition")
    print("=" * 60)
    print("Controls: SPACE start/stop recording, Q quit")
    print("Tip: Press SPACE, perform sign ~2s, press SPACE again (or auto-stop when buffer full).")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)

            # draw landmarks for UX
            if pose_res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            if hands_res.multi_hand_landmarks:
                for hand_lm in hands_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                                              mp_styles.get_default_hand_landmarks_style(),
                                              mp_styles.get_default_hand_connections_style())

            keypoints = extract_keypoints_225_from_mediapipe_results(pose_res, hands_res)
            detection_quality = np.count_nonzero(keypoints) / keypoints.size

            if recording:
                window.append(keypoints)
                progress = len(window) / CAPTURE_FRAMES
                bar_w, bar_h = 400, 28
                bx = (frame.shape[1] - bar_w) // 2
                by = frame.shape[0] - 60
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(bar_w * progress), by + bar_h), (0, 255, 0), -1)
                cv2.putText(frame, f"Recording: {len(window)}/{CAPTURE_FRAMES}", (bx + 10, by + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if len(window) >= CAPTURE_FRAMES:
                    recording = False
                    frames_array = np.array(list(window), dtype=np.float32)
                    print(f"\n>>> DEBUG capture: frames={frames_array.shape[0]}, detection_quality(last)={detection_quality:.4f}, nonzeros={np.count_nonzero(frames_array)}/{frames_array.size}")

                    if frames_array.shape[0] > 5:
                        frames_array = smooth_keypoints(frames_array)
                    frames_array = resample_keypoints(frames_array, TARGET_FRAMES)

                    print(">>> DEBUG: frames_array shape:", frames_array.shape)
                    print(f">>> DEBUG: frames_array stats: mean={frames_array.mean():.6f}, std={frames_array.std():.6f}, min={frames_array.min():.6f}, max={frames_array.max():.6f}")
                    print(">>> DEBUG: sample values (first 20):", np.around(frames_array.flatten()[:20], 6).tolist())

                    # prepare input
                    tensor = prepare_model_input(frames_array)

                    # do forward: either BN-batch-fallback or normal eval
                    if USE_BN_BATCH_FALLBACK and hasattr(model, 'data_bn'):
                        out = bn_batch_forward(model, tensor)
                        used_bn_fallback = True
                    else:
                        model.eval()
                        with torch.no_grad():
                            out = model(tensor)
                        used_bn_fallback = False

                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    logits = out.cpu().numpy()[0].astype(np.float64)
                    probs_torch = torch.nn.functional.softmax(out, dim=1)
                    topk_vals, topk_idx = torch.topk(probs_torch, TOPK, dim=1)
                    topk_vals = topk_vals.cpu().numpy()[0]
                    topk_idx = topk_idx.cpu().numpy()[0]

                    # debug: show top logits
                    top_logits_idx = np.argsort(logits)[-8:][::-1]
                    print(">>> DEBUG: logits (top entries):")
                    for ii in top_logits_idx:
                        print(f"    idx={ii:02d}  logit={logits[ii]:.6f}")

                    print("\n" + "=" * 50)
                    print(f"Predictions (BN-batch-fallback used: {used_bn_fallback}):")
                    for i, (idx, conf) in enumerate(zip(topk_idx, topk_vals), 1):
                        label = resolve_label(idx, labels)
                        print(f"  {i}. {label} (idx={idx}) prob={conf:.6f}")
                    print("=" * 50 + "\n")

                    # accept if confident
                    top1_conf = float(topk_vals[0])
                    if top1_conf > 0.25:
                        pred_buffer.append((resolve_label(int(topk_idx[0]), labels), top1_conf))
                    else:
                        print(">>> Prediction not accepted (low confidence).")

                    window.clear()

            else:
                status_color = (0, 255, 0) if detection_quality > MIN_DETECTION_QUALITY else (0, 165, 255)
                status_text = "Ready - Press SPACE" if detection_quality > MIN_DETECTION_QUALITY else "Position hands in frame"
                cv2.putText(frame, status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                if pred_buffer:
                    last_label, last_conf = pred_buffer[-1]
                    cv2.putText(frame, f"Last: {last_label} ({last_conf:.1%})", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            quality_color = (0, 255, 0) if detection_quality > 0.2 else (0, 165, 255) if detection_quality > 0.1 else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, quality_color, -1)

            cv2.imshow("Sign Language Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                recording = not recording
                window.clear()
                if recording:
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
