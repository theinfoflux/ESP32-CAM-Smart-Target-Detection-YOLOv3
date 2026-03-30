"""
YOLOv3 Object Detection — ESP32-CAM HTTP Stream
TheInfoFlux | github.com/theInfoFlux

Behaviour:
  - Every detected object  →  GREEN bounding box  +  "ClassName  XX.X%" label
  - Target object (person) →  RED  bounding box   +  "ClassName  XX.X%" label
                               + corner accents + alert banner + pulsing border

Requirements:
    pip install opencv-python numpy

Model files — run  python download_model.py  once first:
    model/yolov3.weights
    model/yolov3.cfg
    model/coco.names
"""

import cv2
import numpy as np
import urllib.request
import time
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — only edit this section
# ─────────────────────────────────────────────────────────────────────────────
ESP32_CAM_URL  = "http://192.168.43.144/capture"   # your ESP32-CAM IP
STREAM_MODE    = False       # False = snapshot polling | True = MJPEG stream

TARGET_CLASS   = "remote"    # detected in RED  — must match a name in coco.names

CONFIDENCE_MIN = 0.5         # detection threshold  (0.0 – 1.0)
NMS_THRESHOLD  = 0.4         # non-max suppression
POLL_INTERVAL  = 0.1         # seconds between requests (snapshot mode only)
INPUT_SIZE     = (416, 416)  # (320,320) faster | (608,608) more accurate

MODEL_DIR    = "model"
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3.weights")
CONFIG_FILE  = os.path.join(MODEL_DIR, "yolov3.cfg")
NAMES_FILE   = os.path.join(MODEL_DIR, "coco.names")

WINDOW_NAME  = "TheInfoFlux — Object Detection | ESP32-CAM"

# ── BGR colour constants ──────────────────────────────────────────────────────
GREEN         = (0,  210,  50)   # all non-target objects
GREEN_DARK    = (0,  130,  25)   # label background for non-target
RED           = (0,   30, 255)   # target object
RED_DARK      = (0,   15, 170)   # label background for target
WHITE         = (255, 255, 255)
ALERT_BG      = (0,    0, 160)
BORDER_COLOR  = (0,    0, 255)

ALERT_TEXT    = "!! PERSON DETECTED !!"
BORDER_THICK  = 6
# ─────────────────────────────────────────────────────────────────────────────


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    for f in [WEIGHTS_FILE, CONFIG_FILE, NAMES_FILE]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing file: {f}")
            print("        Run:  python download_model.py")
            sys.exit(1)

    print("[INFO] Loading YOLOv3 ...")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(NAMES_FILE) as f:
        classes = [line.strip() for line in f.readlines()]

    if TARGET_CLASS not in classes:
        print(f"[ERROR] '{TARGET_CLASS}' is not in coco.names")
        print(f"        Valid names: {classes}")
        sys.exit(1)

    target_id     = classes.index(TARGET_CLASS)
    layer_names   = net.getLayerNames()
    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers().flatten()]

    print(f"[INFO] Model ready — {len(classes)} classes")
    print(f"[INFO] Target '{TARGET_CLASS}' (id {target_id}) → RED box")
    print(f"[INFO] All other objects                        → GREEN box")
    return net, classes, target_id, output_layers


# ── frame fetching ────────────────────────────────────────────────────────────

def fetch_snapshot(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            arr = np.frombuffer(resp.read(), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[WARN] Snapshot failed: {e}")
        return None


def open_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {url}")
        sys.exit(1)
    print(f"[INFO] MJPEG stream opened: {url}")
    return cap


# ── detection ─────────────────────────────────────────────────────────────────

def detect_all(net, output_layers, frame):
    """Run YOLOv3 on frame, return all detections that pass CONFIDENCE_MIN."""
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT_SIZE,
                                  swapRB=True, crop=False)
    net.setInput(blob)

    t0     = time.perf_counter()
    outs   = net.forward(output_layers)
    inf_ms = (time.perf_counter() - t0) * 1000

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for det in out:
            scores     = det[5:]
            class_id   = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < CONFIDENCE_MIN:
                continue

            cx, cy = int(det[0] * w), int(det[1] * h)
            bw, bh = int(det[2] * w), int(det[3] * h)
            x1, y1 = cx - bw // 2, cy - bh // 2

            boxes.append([x1, y1, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_MIN, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            cid  = class_ids[i]
            results.append({
                "box":        boxes[i],
                "confidence": confidences[i],
                "class_id":   cid,
            })

    return results, inf_ms


# ── drawing helpers ───────────────────────────────────────────────────────────

def _draw_label(frame, x, y, text, box_color, bg_color, font_scale=0.52):
    """Draw a filled label pill just above the box corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad  = 5
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

    # keep label inside frame vertically
    top = y - th - pad * 2
    if top < 0:
        top = y + 2           # place it just below the top edge of the box

    # filled pill background
    cv2.rectangle(frame, (x, top), (x + tw + pad * 2, top + th + pad * 2),
                  bg_color, -1)
    # thin border matching box colour
    cv2.rectangle(frame, (x, top), (x + tw + pad * 2, top + th + pad * 2),
                  box_color, 1)
    # text
    cv2.putText(frame, text,
                (x + pad, top + th + pad - 1),
                font, font_scale, WHITE, 1, cv2.LINE_AA)


def draw_green_box(frame, x, y, bw, bh, label):
    """Standard green bounding box with label — for all non-target objects."""
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), GREEN, 2)
    _draw_label(frame, x, y, label, GREEN, GREEN_DARK)


def draw_red_box(frame, x, y, bw, bh, label):
    """Red bounding box with corner accents — for the target object."""
    # main box — slightly thicker
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), RED, 3)

    # L-shaped corner accents  (20 px long, 4 px thick)
    c, t = 20, 4
    corners = [
        (x,      y,       1,  1),   # top-left
        (x + bw, y,      -1,  1),   # top-right
        (x,      y + bh,  1, -1),   # bottom-left
        (x + bw, y + bh, -1, -1),   # bottom-right
    ]
    for (cx, cy, sx, sy) in corners:
        cv2.line(frame, (cx, cy), (cx + sx * c, cy),     RED, t)
        cv2.line(frame, (cx, cy), (cx,           cy + sy * c), RED, t)

    # label — slightly larger font to distinguish from green labels
    _draw_label(frame, x, y, label, RED, RED_DARK, font_scale=0.60)


def draw_detections(frame, results, classes, target_id):
    """
    Two-pass render so red boxes always appear on top of green ones.

    Pass 1: all non-target objects  → green box + 'ClassName  XX.X%'
    Pass 2: target objects          → red  box + 'ClassName  XX.X%'
    """
    # ── pass 1: green ─────────────────────────────────────────────
    for r in results:
        if r["class_id"] == target_id:
            continue
        x, y, bw, bh = r["box"]
        name  = classes[r["class_id"]]
        label = f"{name}  {r['confidence'] * 100:.1f}%"
        draw_green_box(frame, x, y, bw, bh, label)

    # ── pass 2: red (drawn last → always on top) ──────────────────
    for r in results:
        if r["class_id"] != target_id:
            continue
        x, y, bw, bh = r["box"]
        name  = classes[r["class_id"]]
        label = f"{name}  {r['confidence'] * 100:.1f}%"
        draw_red_box(frame, x, y, bw, bh, label)

    return frame


def draw_alert_banner(frame, n_targets):
    """Semi-transparent red banner + pulsing red frame border."""
    h, w = frame.shape[:2]

    # semi-transparent banner at bottom
    banner_h = 44
    y0       = h - banner_h
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), ALERT_BG, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    count     = f"{n_targets} {TARGET_CLASS}{'s' if n_targets > 1 else ''}"
    full_text = f"{ALERT_TEXT}  [{count}]"
    (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    tx = (w - tw) // 2
    ty = y0 + (banner_h + th) // 2 - 2

    # shadow + text
    cv2.putText(frame, full_text, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, full_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, WHITE, 2, cv2.LINE_AA)

    # pulsing red border — alternates every ~0.5 s
    if int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), BORDER_COLOR, BORDER_THICK)

    return frame


def overlay_stats(frame, fps, inf_ms, n_total, n_target):
    """Top-left stats bar + top-right legend + bottom-centre watermark."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── stats bar ─────────────────────────────────────────────────
    if n_target > 0:
        status  = f"TARGET x{n_target} DETECTED"
        s_color = (0, 60, 255)      # red tint
    else:
        status  = f"Scanning — {n_total} object{'s' if n_total != 1 else ''}"
        s_color = (0, 200, 80)      # green tint

    stat_line = f"FPS {fps:.1f}  |  Inf {inf_ms:.0f}ms  |  {status}"
    (tw, th), _ = cv2.getTextSize(stat_line, font, 0.48, 1)
    cv2.rectangle(frame, (8, 8),       (16 + tw, 22 + th), (0, 0, 0),  -1)
    cv2.rectangle(frame, (8, 8),       (16 + tw, 22 + th), s_color,      1)
    cv2.putText(frame, stat_line, (12, 8 + th), font, 0.48, s_color, 1, cv2.LINE_AA)

    # ── legend (top-right) ────────────────────────────────────────
    legend_x = w - 155
    cv2.rectangle(frame, (legend_x, 8), (w - 8, 60), (0, 0, 0), -1)

    # red swatch + label
    cv2.rectangle(frame, (legend_x + 104, 16), (w - 14, 27), RED,   -1)
    cv2.putText(frame, TARGET_CLASS.capitalize(),
                (legend_x + 6, 26), font, 0.40, RED,   1, cv2.LINE_AA)

    # green swatch + label
    cv2.rectangle(frame, (legend_x + 104, 38), (w - 14, 49), GREEN, -1)
    cv2.putText(frame, "Other objects",
                (legend_x + 6, 48), font, 0.40, GREEN, 1, cv2.LINE_AA)

    # ── watermark ────────────────────────────────────────────────
    wm = "TheInfoFlux"
    (ww, _), _ = cv2.getTextSize(wm, font, 0.38, 1)
    cv2.putText(frame, wm, ((w - ww) // 2, h - 8),
                font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


# ── main loop ─────────────────────────────────────────────────────────────────

def run():
    net, classes, target_id, output_layers = load_model()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    cap = None
    if STREAM_MODE:
        cap = open_stream(ESP32_CAM_URL)
    else:
        print(f"[INFO] Polling: {ESP32_CAM_URL}  every {POLL_INTERVAL}s")

    print("[INFO] Press  Q  or  ESC  to quit.\n")

    fps_counter, fps_t0, fps = 0, time.perf_counter(), 0.0
    last_print = 0.0

    while True:
        # ── grab frame ──────────────────────────────────────────────────
        if STREAM_MODE:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Stream failed — retrying ...")
                time.sleep(0.5)
                continue
        else:
            frame = fetch_snapshot(ESP32_CAM_URL)
            if frame is None:
                time.sleep(POLL_INTERVAL)
                continue

        # ── run YOLO (all 80 classes) ────────────────────────────────────
        results, inf_ms = detect_all(net, output_layers, frame)

        target_hits = [r for r in results if r["class_id"] == target_id]
        n_target    = len(target_hits)
        n_total     = len(results)

        # ── draw boxes: green for all, red for target ────────────────────
        frame = draw_detections(frame, results, classes, target_id)

        # ── alert banner (only when target is in frame) ──────────────────
        if n_target > 0:
            frame = draw_alert_banner(frame, n_target)

        # ── fps + stats overlay ──────────────────────────────────────────
        fps_counter += 1
        if time.perf_counter() - fps_t0 >= 1.0:
            fps = fps_counter / (time.perf_counter() - fps_t0)
            fps_counter, fps_t0 = 0, time.perf_counter()

        frame = overlay_stats(frame, fps, inf_ms, n_total, n_target)

        cv2.imshow(WINDOW_NAME, frame)

        # ── throttled terminal log ───────────────────────────────────────
        now = time.time()
        if results and now - last_print >= 1.0:
            others = [classes[r["class_id"]] for r in results
                      if r["class_id"] != target_id]
            if n_target:
                confs = [f"{r['confidence']*100:.1f}%" for r in target_hits]
                print(f"[ALERT] {TARGET_CLASS.upper()} x{n_target} "
                      f"conf={', '.join(confs)} "
                      f"| other={', '.join(others) or 'none'}")
            else:
                print(f"[INFO]  Detected: {', '.join(others)}")
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

        if not STREAM_MODE:
            time.sleep(POLL_INTERVAL)

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    run()
