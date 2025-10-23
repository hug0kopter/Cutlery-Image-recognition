import cv2
import numpy as np
from pathlib import Path

# ---------------- PORTABLE PATHS ----------------
REPO_ROOT = Path(__file__).resolve().parents[2]
IN_DIR  = REPO_ROOT / "data_and_features" / "Data_original"
OUT_DIR = REPO_ROOT / "data_and_features" / "Data_proccesed"
# ------------------------------------------------

# ------------- LIGHT TOUCH CONFIG ----------
USE_ADAPTIVE = False
ADAPTIVE_BLOCK = 31
ADAPTIVE_C = 2
GAUSS_BLUR_K = 3
MORPH_OPEN = False
MORPH_K = 3

def make_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if GAUSS_BLUR_K and GAUSS_BLUR_K >= 3:
        gray = cv2.GaussianBlur(gray, (GAUSS_BLUR_K, GAUSS_BLUR_K), 0)

    if USE_ADAPTIVE:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK, ADAPTIVE_C
        )
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if MORPH_OPEN:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    if (mask > 0).mean() > 0.5:
        mask = cv2.bitwise_not(mask)

    return (mask > 0).astype(np.uint8) * 255

IN_DIR = Path(IN_DIR)
OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for src_path in IN_DIR.rglob("*"):
    if src_path.is_file() and src_path.suffix.lower() in exts:
        rel = src_path.relative_to(IN_DIR)
        out_subdir = OUT_DIR / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / f"{src_path.stem}_mask.png"

        image = cv2.imread(str(src_path))
        if image is None:
            print(f"[skip] Could not read: {src_path}")
            continue

        mask = make_mask(image)
        cv2.imwrite(str(out_path), mask)
        print(f"[ok] {src_path} -> {out_path}")
