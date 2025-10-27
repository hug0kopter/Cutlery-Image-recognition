import time
import json
import cv2 as cv
import numpy as np
from pathlib import Path
from imutils.video import VideoStream
import joblib
from collections import deque

# ---------- PATHS (relative & portable) ----------
# This file lives in: Cutlery_Image_Recognition/Scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts_shared"
MODEL_FILE = ARTIFACTS / "cutlery_knn.joblib"
META_FILE  = ARTIFACTS / "dataset_meta.json"

# ---------- CAMERA ----------
FRAME_SIZE = (2200, 1080)  # (width, height)
CAM_SRC = 1

# ---------- SAME SEGMENTATION YOU USED ----------
USE_ADAPTIVE = False
ADAPTIVE_BLOCK = 31
ADAPTIVE_C = 2
GAUSS_BLUR_K = 3
MORPH_OPEN = False
MORPH_K = 3

# ---------- CONTOUR GUARDS ----------
BORDER_TOL = 2
MAX_AREA_FRAC = 0.60
MIN_AREA_ABS = 20.0

# ---------- “IGNORE” PRIOR ----------
MIN_OBJECT_FRAC = 0.008  # object present if area >= 0.8% of frame
IGNORE_LABEL = "ignore"

# ---------- STABILIZATION ----------
USE_EMA = True
EMA_ALPHA = 0.85  # higher -> smoother probs
GAMMA = 2.0       # >1 sharpens probabilities (softmax temperature)
STABLE_FRAMES = 60  # require this many consecutive wins to switch label
DISPLAY_MIN = 0.40  # lower than before to avoid "unsure"
MARGIN_MIN = 0.05   # top1 must beat top2 by at least this (after smoothing+sharpening)

TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)
BOX_COLOR = (0, 255, 255)


# --------------------------------------------------------
def load_model_and_scaler():
    pack = joblib.load(MODEL_FILE)
    model = pack["model"]
    class_names = list(pack["class_names"])

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model has no predict_proba().")

    meta = json.loads(Path(META_FILE).read_text(encoding="utf-8"))
    mean = np.asarray(meta["scaler_mean_"], dtype=np.float64)
    scale = np.asarray(meta["scaler_scale_"], dtype=np.float64)

    if mean.shape[0] != 3 or scale.shape[0] != 3:
        raise RuntimeError("Expected 3 features (aspect_ratio, area_perimeter, extent).")

    # Align proba columns to names
    label_map = {enc: name for enc, name in zip(model.classes_, class_names)}
    proba_names = [label_map[c] for c in model.classes_]
    return model, mean, scale, proba_names


def make_mask(img_bgr):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    if GAUSS_BLUR_K and GAUSS_BLUR_K >= 3:
        gray = cv.GaussianBlur(gray, (GAUSS_BLUR_K, GAUSS_BLUR_K), 0)

    if USE_ADAPTIVE:
        mask = cv.adaptiveThreshold(
            gray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            ADAPTIVE_BLOCK,
            ADAPTIVE_C,
        )
    else:
        _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if MORPH_OPEN:
        k = cv.getStructuringElement(cv.MORPH_RECT, (MORPH_K, MORPH_K))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)

    if (mask > 0).mean() > 0.5:
        mask = cv.bitwise_not(mask)

    return (mask > 0).astype(np.uint8) * 255


def touches_border(bbox, img_w, img_h, tol=BORDER_TOL):
    x, y, w, h = bbox
    return (x <= tol) or (y <= tol) or (x + w >= img_w - tol) or (y + h >= img_h - tol)


def pick_object_contour(mask):
    """Pick a reasonable contour; fallback to center ROI."""
    h, w = mask.shape
    img_area = float(h * w)

    def choose(cnts):
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        for c in cnts:
            area = float(cv.contourArea(c))
            if area < MIN_AREA_ABS or area > MAX_AREA_FRAC * img_area:
                continue
            x, y, bw, bh = cv.boundingRect(c)
            if touches_border((x, y, bw, bh), w, h):
                continue
            return c, area, (x, y, bw, bh)
        return None, None, None

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c, area, bbox = choose(cnts)
    if c is not None:
        return c, area, bbox, img_area

    # center ROI fallback (60%)
    cx0, cy0 = int(0.2 * w), int(0.2 * h)
    cx1, cy1 = int(0.8 * w), int(0.8 * h)
    roi = mask[cy0:cy1, cx0:cx1]
    cnts, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = [cnt + np.array([[cx0, cy0]]) for cnt in cnts]
    c, area, bbox = choose(cnts)
    if c is not None:
        return c, area, bbox, img_area

    return None, None, None, img_area


def features_from_contour(c):
    area = float(cv.contourArea(c))
    perim = float(cv.arcLength(c, True))
    x, y, w, h = cv.boundingRect(c)
    aspect_ratio = (w / h) if h > 0 else 0.0
    area_perim = (area / perim) if perim > 1e-6 else 0.0
    extent = (area / (w * h)) if (w * h) > 0 else 0.0
    return np.array([aspect_ratio, area_perim, extent], dtype=np.float64), (x, y, w, h), area


def scale_features(x, mean, scale):
    return (x - mean) / scale


def sharpen_probs(p, gamma=GAMMA):
    """Raise probs to a power (>1 -> sharper), then renormalize."""
    p = np.clip(p, 1e-9, 1.0)
    p = p ** gamma
    return p / p.sum()


def draw_overlay(frame, label_text, top2_text, bbox):
    if bbox is not None:
        x, y, bw, bh = bbox
        cv.rectangle(frame, (x, y), (x + bw, y + bh), BOX_COLOR, 2)

    (tw, th), _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv.rectangle(frame, (10, 10), (10 + max(620, tw + 20), 70), TEXT_BG, -1)
    cv.putText(frame, label_text, (20, 55), cv.FONT_HERSHEY_SIMPLEX, 1.4, TEXT_FG, 3, cv.LINE_AA)

    if top2_text:
        cv.putText(frame, top2_text, (20, 95), cv.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_FG, 2, cv.LINE_AA)


def main():
    model, mean, scale, proba_names = load_model_and_scaler()
    n_classes = len(proba_names)
    ema = np.ones(n_classes, dtype=np.float64) / n_classes
    ignore_idx = proba_names.index(IGNORE_LABEL) if IGNORE_LABEL in proba_names else None

    # hysteresis state
    stable_label = "unsure"
    last_winner = None
    win_streak = 0

    vs = VideoStream(src=CAM_SRC, usePiCamera=False, resolution=FRAME_SIZE).start()
    time.sleep(1.0)

    print("[INFO] classes:", proba_names)

    try:
        last_t = time.time()

        while True:
            frame = vs.read()
            if frame is None:
                continue

            mask = make_mask(frame)
            c, area, bbox, img_area = pick_object_contour(mask)

            if c is None:
                stable_label = "no object"
                draw_overlay(frame, stable_label, "", None)
            else:
                feats, bbox, area = features_from_contour(c)
                area_frac = area / img_area

                if (ignore_idx is not None) and (area_frac < MIN_OBJECT_FRAC):
                    # empty scene -> force ignore
                    probs = np.zeros(n_classes, dtype=np.float64)
                    probs[ignore_idx] = 1.0
                else:
                    x_scaled = scale_features(feats, mean, scale).reshape(1, -1)
                    probs = model.predict_proba(x_scaled)[0]
                    if (ignore_idx is not None) and (area_frac >= MIN_OBJECT_FRAC):
                        probs[ignore_idx] = 0.0
                    s = probs.sum()
                    probs = probs / s if s > 0 else probs

                if USE_EMA:
                    ema = EMA_ALPHA * ema + (1.0 - EMA_ALPHA) * probs
                    probs_use = ema
                else:
                    probs_use = probs

                # sharpen to encourage a clear winner
                probs_use = sharpen_probs(probs_use, GAMMA)
                order = np.argsort(-probs_use)
                i1, i2 = int(order[0]), int(order[1])
                p1, p2 = float(probs_use[i1]), float(probs_use[i2])
                label1, label2 = proba_names[i1], proba_names[i2]

                # hysteresis: only switch label after STABLE_FRAMES consecutive wins
                if last_winner == label1:
                    win_streak += 1
                else:
                    last_winner = label1
                    win_streak = 1

                if (p1 >= DISPLAY_MIN and (p1 - p2) >= MARGIN_MIN and win_streak >= STABLE_FRAMES):
                    stable_label = label1

                label_text = f"{stable_label} (p1={p1:.2f}, area={area_frac*100:.1f}%)"
                top2_text = f"top2: {label1} {p1:.2f} | {label2} {p2:.2f}"
                draw_overlay(frame, label_text, top2_text, bbox)

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_t))
            last_t = now

            cv.putText(
                frame,
                f"FPS: {fps:.1f}",
                (FRAME_SIZE[0] - 180, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )

            cv.imshow("Cutlery detection (kNN)", frame)
            cv.imshow("Mask (preview)", cv.resize(mask, (0, 0), fx=0.5, fy=0.5))

            k = cv.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break

    finally:
        try:
            vs.stop()
        except Exception:
            pass
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
