# 🎓 Ghi chú học: Bài toán segmentation lỗi nhỏ 6×6 px (UNet)

> File tự học, ghi lại buổi học cùng Claude. Bài 1-8 đã ✅ xong — phần dưới đã RÚT GỌN còn công thức/số
> liệu/kết luận cốt lõi, bỏ phần hỏi-đáp kiểm tra (đã thuộc, không cần ôn lại; lịch sử đầy đủ nằm trong
> `git log` nếu cần tra lại nguyên văn). Bài 9 đang làm — giữ nguyên chi tiết vì còn đang dùng.
> Có file bản đồ riêng: `HOC_vision.md`. Trạng thái tổng các mảng học: `TIEN_DO.md`.

---

## 🗺️ Bản đồ học (9 bài) — ✅ xong · 🟡 đang làm

**Phần A — Nền móng**
- ✅ Bài 1: Segmentation là gì?
- ✅ Bài 2: UNet chạy thế nào? (Encoder – Bottleneck – Decoder – Skip)

**Phần B — Vì sao lỗi 6×6 px khó**
- ✅ Bài 3: Mất cân bằng lớp (class imbalance)
- ✅ Bài 4: Vì sao nén ảnh (downsample) làm lỗi "bốc hơi"

**Phần C — 3 vũ khí giải quyết**
- ✅ Bài 5: Hàm Loss (CE → Dice → Tversky → Focal → FocalTversky)
- ✅ Bài 6: Thước đo (IoU vs Recall/F1, đếm pixel vs đếm "cục lỗi")
- ✅ Bài 7: Kiến trúc (Unet vs Unet++/MAnet)

**Phần D — Thực chiến**
- ✅ Bài 8: Đọc `train_unet.py` từng dòng (đã tự bấm tay ra loss từ 2 bảng số)
- 🟡 Bài 9: Chạy thí nghiệm so sánh (ablation), đọc kết quả — ĐANG LÀM

---

## 🔑 Con số vàng phải nhớ

| | Số pixel |
|---|---|
| Cả tile | 512 × 512 = **262.144** |
| Một con lỗi 6×6 | 6 × 6 = **36** |
| Tỉ lệ | **0,014%** |
| Lỗi ở đáy chữ U | 6 / 32 ≈ **0,19 pixel** (chưa tới 1 pixel → biến mất) |

**Ví von kim cương 💎:** lỗi là thứ *đắt giá nhất* nhưng *nhẹ nhất*. Accuracy cân theo trọng lượng (số pixel) chứ không theo giá trị → nên nó làm ngơ kim cương.

---

## 📚 Recap Bài 1-8 (rút gọn — công thức/kết luận cốt lõi, đã thuộc)

**Bài 1 — Segmentation:** 3 mức, từ thô đến tinh: Classification ("có lỗi không?") → Detection ("lỗi ở
đâu?", khung box) → **Segmentation** ("pixel nào là lỗi?"). Dùng Segmentation vì cần đếm pixel để đo
kích thước lỗi, và box không giữ được hình dạng thật của lỗi nhỏ.

**Bài 2 — UNet chạy thế nào:** Encoder nén để *hiểu* (resnet34, mỗi bước /2: 512→256→128→64→32→16,
tức 2⁵=32×) → Bottleneck (đáy U, 16×16, hiểu tổng thể nhưng mất vật nhỏ) → Decoder bung để *vẽ lại* mask
→ **Skip connection** = đường tắt tuồn chi tiết còn nét từ tầng nông sang Decoder, cực quan trọng cho
lỗi nhỏ. Lỗi 6px ở đáy chỉ còn 6/32 ≈ 0,19px → biến mất.

**Bài 3 — Class imbalance (kẻ thù số 1):** đoán toàn nền vẫn đạt (262.144−36)/262.144 = 99,986% accuracy
nhưng bỏ sót 100% lỗi — vì loss bị biển nền nhấn chìm 36px lỗi. Chống bằng: (1) class weights
`[0.2, 2.0, 2.0, 1.5, 2.0]` (nền phạt nhẹ ×0.2, lỗi phạt nặng ×2.0), (2) đổi sang loss đo vùng (Dice/Tversky).

**Bài 4 — Vì sao downsample làm lỗi "bốc hơi":** 36px → ~9 → ~1.5 → <1 → 0,19px qua các tầng; dưới 1px,
giá trị lỗi bị trung bình hoá với nền vây quanh → ở bottleneck không còn thấy lỗi. 3 hướng cứu: skip
connection, kiến trúc nhiều skip hơn (Unet++/MAnet — Bài 7), loss ép giữ lỗi nhỏ (FocalTversky — Bài 5).

**Bài 5 — Hàm Loss (trái tim bài toán):** 3 chữ nền tảng — **TP** đúng ✅, **FP** báo nhầm 🚨,
**FN** bỏ sót ❌ (nguy hiểm nhất, hàng lỗi lọt ra khách). `Dice = 2·TP/(2·TP+FP+FN)` (đo vùng, miễn nhiễm
biển nền, nhưng phạt FP=FN ngang nhau). `Tversky = TP/(TP + α·FP + β·FN)`: α phạt báo nhầm, β phạt bỏ sót
→ β cao = recall cao; α=β=0.5 → chính là Dice. `FocalTversky = (1−Tversky)^γ`, γ>1 dồn sức vào lỗi nhỏ
còn tô trượt. Đề dùng: `ftl_loss(α=0.3, β=0.7, γ=1.333) + focal_loss(γ=2.0)` (`--loss ftl_focal`).

| Loss | Trường phái | Chống nền nhấn chìm? | Núm chỉnh recall? |
|------|-------------|:---:|:---:|
| CrossEntropy | đếm pixel | ❌ (chỉ nhờ weights) | ❌ |
| Focal | đếm pixel | 一 phần | ❌ |
| Dice | đo vùng | ✅ | ❌ (FP=FN) |
| Tversky | đo vùng | ✅ | ✅ α, β |
| FocalTversky | đo vùng | ✅ | ✅ α,β + γ |

**Bài 6 — Metric & chọn best.pt:** Loss = máy đọc lúc học (mượt/đạo hàm); Metric = người đọc để chọn
`best.pt` (chỉ cần đếm). IoU per-pixel BẤT CÔNG với vật nhỏ: lệch 2px trên lỗi 36px → IoU ~0,38, cùng
lệch đó trên vật 200×200 → IoU ~0,98; đồng thời nền 99,98% "hoàn hảo" kéo điểm trung bình lên dù bỏ sót
cả cục lỗi. → best.pt theo IoU dễ chọn nhầm model "chơi an toàn, né lỗi nhỏ". Giải pháp: chấm theo
**blob/object-level** — cục lỗi chạm ≥1px = TP object, không chạm = FN, bịa ở nền = FP.
**Recall = bắt được / tổng cục thật** là chỉ số quan trọng nhất cho bài toán này.

**Bài 7 — Kiến trúc:** Skip = cứu tinh lỗi nhỏ → càng nhiều/dày càng tốt. Unet (1 skip/tầng) < Unet++
(skip lồng nhau, nhạy vật nhỏ) < MAnet (+attention dồn nhìn vào cụm lỗi) — đánh đổi bằng nặng/chậm/tốn
RAM hơn. backbone = encoder = resnet34 = nửa trái chữ U (3 tên, cùng 1 thứ). Đổi arch không chắc thắng →
phải giữ baseline + chạy ablation, đo bằng Recall object-level, không tin "nghe nói xịn".

**Bài 8 — Đọc `train_unet.py`:** tự bấm tay ra loss=0.30 đúng từ 2 bảng số → hết "mù" đọc code train.

---

# 🟡 Bài 9 — Chạy ablation (ĐANG HỌC — đã chạy thật, 2026-07-07)

**Ablation = đổi ĐÚNG 1 thứ mỗi lần, giữ nguyên phần còn lại, rồi đo** (như bác sĩ thử thuốc). Đổi nhiều thứ cùng lúc = không biết nhờ cái nào.

## 🚨 BÀI HỌC LỚN NHẤT hôm nay: SOI DATA TRƯỚC KHI ĐỔ LỖI MODEL

Chạy thật exp_A (dice_ce+Unet) → `loss` tụt đẹp 1.73→0.18 NHƯNG per-class IoU lộ ra model chỉ học được 1 lớp, còn lại **0.000**. Viết `count_classes.py` đếm pixel/tile mỗi lớp → phát hiện **data thủng**:

- **`data_imgs_unet` (169/74 tile — CŨ, NHỎ):** `sut_ne` có **0 tile train** (model chưa từng thấy → không học nổi); `bavia` có **0 tile val** (thước đo mù → không chấm được). Ba số 0.000 có 3 nguyên nhân khác nhau, KHÔNG phải model dốt.
- **`data_imgs_unet_1` (8374/2067 tile — MỚI, TO):** mọi lớp (`ban, burrs, defo, phong`) đều có mặt ở CẢ train lẫn val → ablation mới có nghĩa. ⚠️ Nhưng `dataset.yaml` của nó có bug: `path:` trỏ nhầm về folder cũ — đã sửa thành `.../data_imgs_unet_1`.

**Chốt Bài 9:** *Trước khi đổ lỗi model/loss/arch → đếm phân bố lớp trong train VÀ val.* Nửa số "thất bại" là data thiếu lớp, không phải model. Chỉ nhìn `loss` tụt mà mừng là bẫy (Bài 3 + Bài 6 hiện ra bằng số thật). Ablation chỉ đo được lớp có data thật ở cả 2 phía.

**Cách chạy đã dùng (LƯU Ý):** viết `run_exp_*.sh` rồi `wsl -e bash /mnt/.../run.sh` **QUA POWERSHELL** — Bash tool (Git Bash) DỊCH NHẦM path `/mnt/d`→`C:/Program Files/Git/mnt/d` làm vỡ. Log `tee` ra `logs/*.log`; lọc dòng tổng kết bằng `grep -- "-> loss="`. Data to → ~2.5 phút/epoch, 150 epoch ~6h → khi học chạy rút gọn ~12 epoch.

## Kế hoạch ablation gốc (3 lần chạy)

3 lệnh (chạy trong WSL env vision_ai — xem cách chạy ở memory python-env-wsl-vision-ai):
```
python train_unet.py --loss dice_ce  --arch Unet         --name exp_A_baseline
python train_unet.py --loss ftl_focal --arch Unet         --name exp_B_ftl
python train_unet.py --loss ftl_focal --arch UnetPlusPlus --name exp_C_unetpp
```
A vs B = loss có ăn không; B vs C = arch có ăn thêm không.

**Báo cáo cho sếp = bảng "cỡ mm → recall"**: gom ảnh test theo cỡ lỗi (1/2/3mm), đếm bỏ sót mỗi cỡ → chỗ recall tụt mạnh = **giới hạn phát hiện** ("bắt tốt từ X mm trở lên").

✅ Cập nhật 2026-07-19: đã kiểm tra lại `train_unet.py` — việc nâng best.pt sang Recall/F1 object-level (Bài 6)
ĐÃ LÀM XONG, không còn nợ nữa. Code có cờ `--best-metric` (`iou_pixel` | `recall_object` | `f1_object`,
mặc định `f1_object`). Ghi chú cũ ở đây (dòng 266 chọn IoU) đã lỗi thời, đã xoá.

### 📚 Tài liệu tham khảo cho gói ablation này
(gộp từ `NOTES_unet_small_defect.md` cũ — đã xoá vì nội dung phân tích của nó chính là kế hoạch exp_A/B/C ở trên, không cần giữ 2 bản)
- **Loss cho imbalance:** Tversky Loss (2017), Focal Tversky Loss (2018), Focal Loss / RetinaNet (2017).
- **Vì sao object nhỏ khó:** downsampling stride làm mất phân giải không gian, vật < stride tầng sâu thì
  mất tín hiệu. Tra cứu thêm: "small object segmentation", "receptive field vs object size",
  "output stride / dilated convolution".
- **Metric object-level:** connected-components / blob matching (đúng thứ Bài 6 đang dùng).
- **Kiến trúc giữ vật thể nhỏ ngoài Unet++/MAnet:** HRNet (giữ high-resolution suốt mạng), feature
  pyramid, giảm `encoder_depth`.
- Điểm phụ đã chốt (không cần thí nghiệm lại): augmentation flip/rot90/brightness AN TOÀN cho lỗi nhỏ
  (rot90 lossless); KHÔNG dùng scale-down/elastic/rotate góc lẻ (phá lỗi 6px); cân nhắc
  `WeightedRandomSampler` nếu `bg_ratio` cao làm loãng tín hiệu dương.

---

## 🔧 Ghi chú kỹ thuật
- Đã sửa lỗi cú pháp dòng 129 `train_unet.py` (xóa chữ `k` thừa sau `default=0.7,`). File giờ chạy được.
- Công tắc trong code: `--loss dice_ce` (baseline) vs `--loss ftl_focal` (đề mới cho lỗi nhỏ). Các núm: `--tv-alpha --tv-beta --tv-gamma --focal-gamma`.
