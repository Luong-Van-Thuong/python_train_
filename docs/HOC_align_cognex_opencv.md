# 🎓 Ghi chú: Tự dựng "VisionPro mini" bằng OpenCV (Cognex → OpenCV)

> Buổi học cùng Claude (2026-07-07). Bối cảnh: mình từ web/WinForm CRUD chuyển sang AI/vision.
> Đã quen Cognex VisionPro (kéo-thả tool), giờ muốn TỰ DỰNG bằng OpenCV để không phụ thuộc license.
> Gap thật của mình KHÔNG phải khái niệm — mà là **bản đồ dịch Cognex → OpenCV** và **tư duy hệ tọa độ**.

---

## 1. Định vị lại vấn đề

Mình **đã sở hữu tư duy Cognex**: `Align → Fixture → đặt tool → đo`. Cái thiếu chỉ là:
1. Bảng dịch "tool Cognex = kỹ thuật OpenCV nào".
2. Hiểu chất keo nối các tool = **hệ tọa độ** (Cognex giấu đi qua Fixture nên chưa từng phải nghĩ).

→ Đây là **dịch từ ngôn ngữ đã giỏi sang OpenCV**, không phải học lại từ đầu.

## 2. Bảng dịch Cognex → OpenCV

| Tool Cognex | Làm gì | Tương đương OpenCV |
|---|---|---|
| **CogPMAlign / PatMax** | Tìm mẫu → pose (x,y,θ,scale) | `matchTemplate` (chỉ tịnh tiến) / feature ORB-SIFT + `estimateAffinePartial2D(RANSAC)` / geometric matching |
| **CogFixture** (đặt gốc) | Tool dính theo con hàng khi xê dịch | **Ma trận affine M** từ Align → `warpAffine` nắn ảnh chuẩn, hoặc nhân ROI với M |
| **CogCaliper** | Dò 1 cạnh sub-pixel | ROI → chiếu profile 1D → `Sobel`/`np.gradient` → đỉnh + nội suy parabol |
| **CogFindLine** | Fit đường từ nhiều caliper | điểm cạnh → `cv2.fitLine` (RANSAC) |
| **CogFindCircle** | Fit vòng tròn | fit circle least-square / `HoughCircles` |
| **CogBlob** | Đếm/đo vùng | `connectedComponentsWithStats` / `findContours` |
| **CogCalibCheckerboard** | pixel → mm | `findChessboardCorners` + `calibrateCamera`, hoặc tỉ lệ vật chuẩn |
| **CogDistance** | Đo khoảng cách | toán numpy trên đường/điểm đã fit |

> `src/measure/test_a26.py` của mình ĐÃ là một Caliper-pair — mình không ở vạch xuất phát.

## 3. TƯ DUY CỐT LÕI: đo đạc = 90% quản lý HỆ TỌA ĐỘ, 10% xử lý ảnh

3 hệ tọa độ, mọi việc là đi lại giữa chúng:

| Hệ | Là gì |
|---|---|
| **Pixel space** | tọa độ thô trên ảnh |
| **Part space** | tọa độ *gắn theo con hàng* — con hàng xê dịch thì KHÔNG đổi |
| **World (mm)** | tọa độ đời thực |

```
Align    : con hàng ở đâu trong PIXEL? → pose (x,y,θ)
Fixture  : pose → ma trận M (PIXEL ↔ PART) → đặt tool theo PART space
Caliper  : trong ROI, cạnh ở đâu (sub-pixel)? → điểm cạnh
Measure  : khoảng cách/góc giữa các hình → toán hình học
Calibrate: × tỉ lệ → ra mm (WORLD)
```

**Vì sao có bước Align:** không Align → caliper đặt ở pixel cố định → con hàng lệch vài mm → cạnh trôi khỏi ROI → sai. Có Align → đặt caliper trong part space → con hàng lệch thì part space trôi theo → vẫn trúng. **Fixture = ma trận affine M** chính là cây cầu pixel ↔ part.

**Hỏi 3 câu khi nhận bài đo mới:** (1) Cái gì cố định làm GỐC? (2) Đo gì, ở đâu SO VỚI gốc? (3) 1 pixel = mấy mm?

## 4. ĐÀO SÂU khâu ALIGN (khó nhất, "bí kíp" Cognex)

**PatMax thực chất = geometric matching trên EDGE:** mô hình mẫu = tập điểm cạnh + **hướng gradient**; khớp pose theo cạnh+hướng. Dùng hướng gradient (không dùng độ sáng) nên trơ ánh sáng, chịu che, chịu đảo tương phản.

### Thang thuật toán (yếu → mạnh)
1. **Template Matching** `cv2.matchTemplate` (NCC): đơn giản, sub-pixel bằng parabol. ❌ chỉ tịnh tiến, nhạy sáng.
2. **Geometric/Edge = PatMax:** `cv2.createGeneralizedHoughGuil()` (ra x,y,θ,scale, chậm), linemod (contrib), hoặc TỰ DỰNG edge-model coarse-to-fine. ✅ trơ sáng, chịu che, ra góc+scale. ❌ chết khi cạnh yếu / pattern lặp.
3. **Feature-based:** ORB/SIFT/AKAZE + `estimateAffinePartial2D(RANSAC)`. ✅ xoay/scale/che. ❌ **cần texture** — con hàng nhẵn/bóng → ít keypoint → chết (điểm chết hay gặp ở đồ nhà máy).
4. **ECC** `cv2.findTransformECC`: tối ưu lặp, sub-pixel — dùng làm **KHÂU TINH CHỈNH** sau pose thô (local optimizer, cần điểm khởi đầu tốt).
5. **Phase correlation** `cv2.phaseCorrelate`: FFT ra tịnh tiến (log-polar → xoay+scale), trơ sáng, nhanh — làm pre-align thô.

### Bảng "Align chết → cứu"
| Chết vì | Nguyên nhân | Cứu |
|---|---|---|
| Part xoay, matchTemplate trượt | NCC không bất biến xoay | bank mẫu theo góc / geometric / feature+RANSAC |
| Part nhẵn, feature ít điểm | ORB cần texture | edge/geometric, SIFT/AKAZE, dùng contour làm model |
| Ánh sáng đổi | dùng độ sáng thô | match trên ảnh gradient/edge |
| Pattern lặp/đối xứng | mẫu mơ hồ | thêm feature bất đối xứng, thu ROI, ràng góc |
| Bị che | occlusion | geometric, hoặc multi-anchor (khớp 3/5 mẫu nhỏ) |
| Chậm | quét full-res/full-angle | **pyramid coarse-to-fine**, thu ROI, bước góc thô→refine |
| Thô, thiếu chính xác | pose thô | **ECC** / nội suy sub-pixel |

### 2 tối ưu "phải biết"
- **A. Coarse-to-fine trên image pyramid:** tìm ở ảnh nhỏ → refine ở ảnh gốc. Nhanh 10–100×. (PatMax làm đúng thế.)
- **B. Sub-pixel/sub-degree refine:** fit parabol quanh đỉnh score theo x,y,θ; hoặc ECC. Đạt 0.01mm không cần zoom.

### Công thức mặc định (đồ nhà máy)
- Biên/contour rõ (đa số): **geometric/edge coarse-to-fine → ECC refine**.
- Có texture: **SIFT/AKAZE + RANSAC affine → ECC**.
- Chỉ tịnh tiến, sáng ổn định: **matchTemplate → sub-pixel**.
- LUÔN kết bằng khâu refine.

> Thẳng thắn: đạt chất lượng PatMax bằng OpenCV thuần phải **tự dựng geometric matcher** (edge-model + coarse-to-fine + score theo hướng gradient). Generalized Hough Guil gần nhất nhưng chậm. Vì khó nên mới "kiếm cơm" được.

## 5. Còn nợ (buổi sau đào tiếp — mình đang tìm hiểu, CHƯA code)
- [ ] Geometric edge-model hoạt động từng bước (lõi PatMax).
- [ ] Coarse-to-fine pyramid vận hành ra sao.
- [ ] ECC refine bản chất tối ưu cái gì.
- [ ] Khi sẵn sàng: dựng pipeline Align→Fixture→Caliper→đo mm end-to-end trên 1 ảnh thật.
