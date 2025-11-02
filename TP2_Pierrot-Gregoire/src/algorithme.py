import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque
from create_model import load_model, save_model

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Pour éviter le spam des logs GPU

#==================================================================#

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#==================================================================#

def norm(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

#==================================================================#

def angle_beetween(u, v):
    u = norm(u); v = norm(v)
    dp = np.clip(np.dot(u, v), -1.0, 1.0)
    return math.degrees(math.acos(dp))

#==================================================================#

def detect_hand_mediapipe(frame):
    frame_copy = frame.copy()
    rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame_copy.shape
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        margin_percent = 0.2
        margin_x = int((x_max - x_min) * margin_percent)
        margin_y = int((y_max - y_min) * margin_percent)
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)

        return x_min, y_min, x_max, y_max
    return None

#==================================================================#
'''
def count_fingers(hand_frame):
    gray = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(cnt, returnPoints=False)

        defects = cv2.convexityDefects(cnt, hull)

        finger_count = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, depth = defects[i, 0]

                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                if depth > 1500:  
                    finger_count += 1

                    cv2.circle(hand_frame, far, 5, (0, 0, 255), -1)
                    cv2.line(hand_frame, start, end, (255, 0, 0), 2)

        finger_count = min(finger_count + 1, 5)
        return finger_count
    return 0
'''
#==================================================================#
'''
def hand_edges(hand_frame, show_debug=False):
    H, W = hand_frame.shape[:2]
    scale = 1.0
    if max(H, W) > 400:
        scale = 400.0 / max(H, W)
        hand_small = cv2.resize(hand_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        hand_small = hand_frame.copy()

    gray = cv2.cvtColor(hand_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    low_thresh = 10
    high_thresh = 150
    edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=3, L2gradient=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    thin_done = False
    try:
        thin = cv2.ximgproc.thinning(edges)
        edges = thin
        thin_done = True
    except Exception:
        try:
            img = edges.copy()
            img[img > 0] = 1
            prev = np.zeros_like(img)
            skel = np.zeros_like(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(img, kernel)
                temp = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                if cv2.countNonZero(img) == 0:
                    break
            edges = (skel * 255).astype('uint8')
            thin_done = True
        except Exception:
            thin_done = False

    if scale != 1.0:
        edges = cv2.resize(edges, (W, H), interpolation=cv2.INTER_LINEAR)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if show_debug:
        info = f"edges shape:{edges.shape}, thinning:{thin_done}"
        print(info)

    return edges_bgr
'''
#==================================================================#

def count_fingers_from_landmarks(landmarks, handedness_label=None):
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    lm = [(p.x, p.y, p.z) for p in landmarks]
    
    fingers = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20]
    }

    states = {name: False for name in fingers.keys()}
    count = 0
    debug = {finger: {} for finger in fingers.keys()}

    wrist = lm[0]

    for finger_name, idxs in fingers.items():
        mcp = lm[idxs[0]]
        pip = lm[idxs[1]]
        dip = lm[idxs[2]]
        tip = lm[idxs[3]]

        d_mcp = distance(mcp, wrist)
        d_pip = distance(pip, wrist)
        d_dip = distance(dip, wrist)
        d_tip = distance(tip, wrist)

        angle1 = angle_beetween(np.array(mcp) - np.array(wrist), np.array(pip) - np.array(mcp))
        angle2 = angle_beetween(np.array(pip) - np.array(mcp), np.array(dip) - np.array(pip))
        angle3 = angle_beetween(np.array(dip) - np.array(pip), np.array(tip) - np.array(dip))
        total_angle = angle1 + angle2 + angle3

        cond_dist = (d_tip > d_dip > d_pip > d_mcp)
        cond_angle = total_angle < 90
        is_extended = cond_dist and cond_angle

        if is_extended:
            states[finger_name] = True
            count += 1
        
        debug[finger_name] = {
            "distances": {
                "mcp": d_mcp,
                "pip": d_pip,
                "dip": d_dip,
                "tip": d_tip
            },
            "angles": {
                "angle_mcp_pip": angle1,
                "angle_pip_dip": angle2,
                "angle_dip_tip": angle3,
                "total_angle": total_angle
            },
            "conditions": {
                "distance_order_ok": cond_dist,
                "angle_ok": cond_angle
            },
            "extended": is_extended
        }

    return count, states, debug

#==================================================================#

def draw_landmarks_and_vectors_on_roi(roi, landmarks, origin, full_shape, states=None):
    full_w, full_h = full_shape[0], full_shape[1]
    x_off, y_off = origin

    lm_px = []
    for p in landmarks:
        x = int(p.x * full_w) - x_off
        y = int(p.y * full_h) - y_off
        lm_px.append((x, y))

    for i, (x, y) in enumerate(lm_px):
        if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
            cv2.circle(roi, (x, y), 3, (200, 200, 200), -1)

    connections = [
        (0, 1),(1, 2),(2, 3),(3, 4),        # thumb
        (0, 5),(5, 6),(6, 7),(7, 8),        # index
        (0, 9),(9, 10),(10, 11),(11, 12),   # middle
        (0, 13),(13, 14),(14, 15),(15, 16), # ring
        (0, 17),(17, 18),(18, 19),(19, 20)  # pinky
    ]
    for (a,b) in connections:
        xa, ya = lm_px[a]; xb, yb = lm_px[b]
        if (0 <= xa < roi.shape[1] and 0 <= ya < roi.shape[0]) and (0 <= xb < roi.shape[1] and 0 <= yb < roi.shape[0]):
            cv2.line(roi, (xa, ya), (xb, yb), (90, 90, 90), 2)

    finger_tips_idx = [4, 8, 12, 16, 20]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for i, idx in enumerate(finger_tips_idx):
        x, y = lm_px[idx]
        if not (0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]):
            continue
        if states is None:
            col = (0, 255, 255)
        else:
            finger_name = finger_names[i]
            col = (0, 220, 0) if states[finger_name] else (0, 60, 255)
        cv2.circle(roi, (x, y), 6, col, -1)
        cv2.putText(roi, str(i), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#==================================================================#

def bbox_from_landmarks(landmarks, image_shape, margin_percent=0.2):
    h, w = image_shape[:2]
    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x_min, x_max = max(0, min(xs)), min(w, max(xs))
    y_min, y_max = max(0, min(ys)), min(h, max(ys))
    margin_x = int((x_max - x_min) * margin_percent)
    margin_y = int((y_max - y_min) * margin_percent)
    return max(0, x_min - margin_x), max(0, y_min - margin_y), min(w, x_max + margin_x), min(h, y_max + margin_y)

#==================================================================#

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    if not ret:
        break
    
    show_frame = frame.copy()
    hand_frame = detect_hand_mediapipe(show_frame)
    if hand_frame is not None:
        x_min, y_min, x_max, y_max = hand_frame
        hand_frame = show_frame[y_min:y_max, x_min:x_max]

        try:
            rgb = cv2.cvtColor(show_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_label = None
                try:
                    handedness_label = results.multi_handedness[0].classification[0].label
                except Exception:
                    handedness_label = None

                count, states, debug = count_fingers_from_landmarks(hand_landmarks.landmark, handedness_label)
                draw_landmarks_and_vectors_on_roi(hand_frame, hand_landmarks.landmark, (x_min, y_min), (w, h), states)

                cv2.putText(hand_frame, f"Doigts: {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0), 2)
                for i, (finger, info) in enumerate(debug.items()):
                    ext = "E" if info["extended"] else "F"
                    cv2.putText(show_frame, f"{finger[0].upper()}:{ext}", (10, 50 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        except Exception as e:
            pass
        
        hand_frame = cv2.resize(hand_frame, (h, w))

    else:
        cv2.putText(frame, "Aucune main detectee", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        hand_frame = frame
        
    cv2.imshow('Image', show_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()