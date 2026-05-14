import cv2
import numpy as np

def calculate_tss(heatmap, original_img):
    """
    Calculates Tumor Severity Score (TSS)
    heatmap  : Grad-CAM output (0–1)
    original_img : Original BGR image (cv2 format)
    """

    # ---------------- SAFETY CHECK ----------------
    if heatmap is None or original_img is None:
        raise ValueError("❌ Heatmap or original image is None")

    # Ensure heatmap is float
    heatmap = heatmap.astype("float32")

    # ---------------- RESIZE HEATMAP ----------------
    h, w = original_img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))

    # Normalize properly
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_uint8 = np.uint8(255 * heatmap)

    # ---------------- TUMOR MASK ----------------
    thresh_val = int(0.6 * 255)  # 60% threshold
    _, tumor_mask = cv2.threshold(
        heatmap_uint8,
        thresh_val,
        255,
        cv2.THRESH_BINARY
    )

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)

    tumor_area = np.sum(tumor_mask > 0)

    # ---------------- BRAIN MASK (OTSU) ----------------
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, brain_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    brain_area = np.sum(brain_mask > 0)

    # ---------------- TSS CALCULATION ----------------
    if brain_area == 0:
        return 0, 0, 0, tumor_mask

    tumor_area = min(tumor_area, brain_area)
    tss = (tumor_area / brain_area) * 100
    tss = np.clip(tss, 0, 100)

    return tumor_area, brain_area, round(tss, 2), tumor_mask