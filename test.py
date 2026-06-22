
import glob
nc = 2  # so class trong data.yaml (bien_dang, thieu_nhua)
bad = []
for f in glob.glob("train/*.txt") + glob.glob("val/*.txt"):
    for ln, line in enumerate(open(f), 1):
        s = line.split()
        if not s:
            continue
        cid = int(float(s[0]))
        coords = list(map(float, s[1:]))
        if cid < 0 or cid >= nc:
            bad.append((f, ln, "class id sai", cid))
        if coords and (min(coords) < 0 or max(coords) > 1):
            bad.append((f, ln, "toa do ngoai [0,1]", None))
        if len(coords) % 2 != 0:
            bad.append((f, ln, "so toa do le", len(coords)))
        if len(coords) < 6:
            bad.append((f, ln, "polygon < 3 diem", len(coords)))
print("Tong dong loi:", len(bad))
for b in bad[:30]:
        if cid < 0 or cid >= nc:
            bad.append((f, ln, "class id sai", cid))
        if coords and (min(coords) < 0 or max(coords) > 1):
            bad.append((f, ln, "toa do ngoai [0,1]", None))
        if len(coords) % 2 != 0:
            bad.append((f, ln, "so toa do le", len(coords)))
        if len(coords) < 6:
            bad.append((f, ln, "polygon < 3 diem", len(coords)))
print("Tong dong loi:", len(bad))
for b in bad[:30]:
    print(b)