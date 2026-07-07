import cv2, numpy as np

img = cv2.imread("/mnt/d/Images_/SIBV/A26/260615_0/tesst/Image__2026-06-15__11-23-31_obj_0.bmp")
B, G, R = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
green = ((G > 140) & (R < 110) & (B < 110)).astype(np.uint8)
red   = ((R > 150) & (G < 90) & (B < 90)).astype(np.uint8)
mask = cv2.dilate((green | red), np.ones((5, 5), np.uint8))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clean = cv2.inpaint(gray, mask, 5, cv2.INPAINT_TELEA)


def find_edge(sub, col_lo, col_hi, polarity):
    """tim cot co |gradient ngang| cuc dai theo phan cuc, noi suy sub-pixel."""
    gx = cv2.Sobel(cv2.GaussianBlur(sub, (3, 3), 0).astype(np.float32),
                   cv2.CV_32F, 1, 0, ksize=3)
    prof = gx.mean(axis=0) * polarity            # huong can tim thanh duong
    col_lo, col_hi = max(1, col_lo), min(len(prof) - 2, col_hi)
    win = prof[col_lo:col_hi + 1]
    k = int(np.argmax(win)) + col_lo
    a, b, c = prof[k - 1], prof[k], prof[k + 1]   # sub-pixel parabol
    den = (a - 2 * b + c)
    shift = 0.5 * (a - c) / den if den != 0 else 0.0
    return k + shift

# moi zone: ROI + 2 caliper (cot tuong doi trong ROI, phan cuc) | ky vong
ZONES = [
    dict(name="tab_trai", roi=(85, 1289, 92, 391),
         eA=(22, 38, -1), eB=(39, 55, +1), expect=15),
    dict(name="day_phai", roi=(1846, 1911, 82, 171),
         eA=(25, 40, +1), eB=(41, 58, +1), expect=13),
    dict(name="day_trai", roi=(285, 1938, 69, 160),
         eA=(18, 33, -1), eB=(34, 50, -1), expect=12),
]
for z in ZONES:
    x, y, w, h = z["roi"]
    sub = clean[y:y+h, x:x+w]
    xa = find_edge(sub, z["eA"][0], z["eA"][1], z["eA"][2])
    xb = find_edge(sub, z["eB"][0], z["eB"][1], z["eB"][2])
    dist = xb - xa
    print(f"{z['name']:9s}: edgeA(global x)={x+xa:7.2f}  edgeB={x+xb:7.2f}  "
          f"distance={dist:6.2f}px  (ky vong ~{z['expect']})")
