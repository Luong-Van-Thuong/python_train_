# 🎓 Ghi chú học: Bài toán segmentation lỗi nhỏ 6×6 px (UNet)

> File tự học, ghi lại buổi học cùng Claude. Mỗi bài có phần lý thuyết + câu hỏi kiểm tra.
> **Đang học tới:** Bài 8 XONG (đã tự bấm tay ra loss=0.30 đúng — hết "mù"). Đang vào Bài 9 (chạy ablation).
> Ghi chú: user từng thấy "trả lời đúng lý thuyết mà chưa hiểu" → đã đổi cách dạy sang BẢNG SỐ CỤ THỂ + tự bấm máy. Cách này hiệu quả, tiếp tục dùng.
> Có file bản đồ riêng: HOC_vision.md (2 nhánh OpenCV/DL + Playbook 3 ngày).

---

## 🗺️ Bản đồ học (9 bài)

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
- ⬜ Bài 9: Chạy thí nghiệm so sánh (ablation), đọc kết quả

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

# 📘 Bài 1 — Segmentation là gì?

Máy "nhìn" ảnh có 3 mức, từ thô đến tinh:

| Mức | Trả lời câu hỏi | Kết quả |
|-----|-----------------|---------|
| Classification | "Ảnh *có* lỗi không?" | 1 nhãn có/không |
| Detection | "Lỗi *nằm đâu*?" | 1 khung box |
| **Segmentation** | "*Pixel nào* là lỗi?" | bản đồ tô màu từng pixel |

Bài của mình dùng **Segmentation** (mức tinh nhất).

**3 từ khóa:**
- **Tile** = miếng ảnh vuông cắt ra để đưa vào model = **512×512 px**.
- **Mask** = "đáp án tô màu" cùng kích thước tile, mỗi pixel mang **mã lớp**: `0`=nền, `1,2,3,4`=các loại lỗi.
- **Class (lớp)** = loại lỗi. Code có 5 lớp (1 nền + 4 lỗi) → `weights_list` có 5 số.

**Việc của model:** nhận 1 tile (512×512×3) → đoán ra 1 mask (512×512, mỗi pixel điền số lớp).

### Câu hỏi & đáp án của mình
1. *Lỗi 6×6 px trong mask trông thế nào, bao nhiêu pixel khác 0?* → **36 pixel** mang số khác 0 (nền=0, mỗi loại lỗi=1 số khác 0).
2. *Vì sao dùng Segmentation chứ không Detection?* → vì khung box **không chính xác**: không giữ được hình dạng thật, vật 6px dễ bị model box bỏ qua, và cần **đếm pixel để đo kích thước** lỗi (quyết định đạt/không đạt).

> Chốt: **Detection = "quanh đâu có lỗi"; Segmentation = "chính xác pixel nào là lỗi".**

---

# 📗 Bài 2 — UNet chạy thế nào?

UNet hình chữ **U**, 3 bộ phận + skip:

```
   Ảnh vào 512²                          Mask ra 512²
        │                                     ▲
   ┌────▼────┐  ENCODER (nén)   DECODER (bung) ┌──┴──┐
   │  256²   │ ─────skip────────────────────► │ 256²│
   │   128²  │ ───────skip──────────────────► │128² │
   │    64²  │ ─────────skip────────────────► │ 64² │
   │     32² │ ───────────skip──────────────► │32²  │
   └─────16²─┘        BOTTLENECK (đáy U)   └─16²──┘
```

- **Encoder** = nén để *hiểu*. Mỗi bước thu nhỏ ảnh /2: 512→256→128→64→32→16 (5 bước). Việc thu nhỏ = **downsample**. `resnet34` chính là encoder này.
- **Bottleneck** = đáy chữ U (16×16), hiểu tổng thể nhất — nhưng vật nhỏ dễ mất nhất.
- **Decoder** = bung để *vẽ ra đáp án*. Phóng ảnh /2 ngược lại về 512, tô màu từng pixel.
- **Skip connection** = đường tắt tuồn chi tiết còn nét từ Encoder (tầng nông 256²,128²) thẳng sang Decoder. **Cực quan trọng cho lỗi nhỏ.**

### Câu hỏi & đáp án của mình
1. *Encoder/Decoder làm gì?* → **Encoder nén để hiểu, Decoder bung để dựng lại mask 512×512.**
2. *"Downsample 32×" số 32 ở đâu?* → **2⁵ = 32** (5 lần chia đôi).
3. *Lỗi 6×6 ở đáy còn bao nhiêu?* → **6/32 ≈ 0,19 pixel** → chưa tới 1 pixel → biến mất.

---

# 📙 Bài 3 — Mất cân bằng lớp (class imbalance)

**Kẻ thù số 1**, sinh từ con số 36/262.144.

**Vấn đề "lười mà điểm cao":** model gian lận đoán *toàn bộ là nền* vẫn đạt:
```
(262.144 − 36) / 262.144 = 99,986% pixel đúng
```
nhưng **bỏ sót 100% lỗi** → vô dụng. Vì hàm loss bị **biển nền nhấn chìm** 36 pixel lỗi.

**2 cách chống (code dùng cả 2):**
1. **Class weights** — `weights_list = [0.2, 2.0, 2.0, 1.5, 2.0]` (nền phạt nhẹ ×0.2, lỗi phạt nặng ×2.0). Trọng số gắn với LỚP, nhân vào sai sót tại pixel thuộc lớp đó → **bỏ sót lỗi đau gấp ~10 lần tô lố nền** → model "thà báo nhầm hơn bỏ sót" (= **recall**).
2. **Đổi loại loss** — dùng Dice/Tversky đo *độ chồng lấn vùng*, không quan tâm nền to (Bài 5).

### Câu hỏi & đáp án của mình
1. *Vì sao đoán toàn nền vẫn 99,99% mà vô dụng?* → vì lỗi (kim cương) tuy quan trọng nhất nhưng chiếm tỉ lệ pixel cực nhỏ; accuracy cân theo *số lượng* nên bỏ qua nó.
2. *Vì sao số đầu (0.2) nhỏ hơn?* → đó là trọng số phạt sai trên **nền** (để nhẹ), các số sau phạt sai trên **lỗi** (để nặng), mục tiêu ép model không lười bỏ sót lỗi.

---

# 📕 Bài 4 — Vì sao downsample làm lỗi "bốc hơi"

**Downsample = gộp ô rồi lấy trung bình.** Theo dõi lỗi 6×6 qua các bước:
```
512²: 36 pixel sáng rõ
 256²: ~9 pixel, mờ đi
 128²: ~1.5px, lẫn nền
  64²: <1px, hòa tan vào nền
  16²: 0,19px → không còn pixel riêng
```

**Vì sao "hòa tan" là chết:** khi lỗi <1 pixel, giá trị nó bị **lấy trung bình chung với nền vây quanh** → ra con số gần y hệt nền → model không phân biệt nổi → ở bottleneck **không còn thấy lỗi** → không tô lại được.

> Nghịch lý UNet: Encoder càng nén sâu càng hiểu tốt vật to, nhưng càng bóp chết vật nhỏ.

**3 hướng cứu (= 3 vũ khí Phần C):**
1. **Skip connection** giữ chi tiết từ tầng nông (lỗi chưa bốc hơi) → Decoder.
2. **Đổi kiến trúc** Unet++ / MAnet có nhiều đường skip dày hơn (Bài 7).
3. **Loss** FocalTversky ép model cố giữ lỗi nhỏ (Bài 5).

### Câu hỏi & đáp án của mình
1. *Vì sao <1px = mất tín hiệu?* → giá trị lỗi bị trung bình chung với nền vây quanh → thành ~giá trị nền → không phân biệt được.
2. *Bộ phận nào là "vị cứu tinh"?* → **Decoder + Skip**, vì skip tuồn thẳng chi tiết từ tầng nông còn nét (256²,128²) sang Decoder, né được cái đáy U nơi lỗi bị bóp chết.

---

# 📘 Bài 5 — Hàm Loss (trái tim của việc dạy lỗi nhỏ)

**Loss = thước đo model sai bao nhiêu. Chọn loss = chọn model quan tâm điều gì.** Có 2 trường phái.

## Trường phái 1 — "Đếm từng pixel"
- **CrossEntropy (CE):** duyệt từng pixel, sai thì phạt. 👎 bị nền nhấn chìm (chỉ đỡ nhờ class weights). Đây là nửa của đề cũ `dice_ce`.
- **Focal (nâng cấp CE):** vẫn đếm pixel nhưng **pixel dễ giảm phạt, dồn sức pixel khó**. Núm `--focal-gamma` (=2.0): càng cao càng bỏ pixel dễ. Lỗi nhỏ là pixel khó → được ưu ái.

## Trường phái 2 — "Đo độ chồng lấn vùng" (miễn nhiễm biển nền)
**3 chữ nền tảng:**
- **TP** = pixel lỗi tô **đúng** ✅
- **FP** = nền bị tô nhầm thành lỗi = **báo nhầm** 🚨
- **FN** = lỗi bị **bỏ sót** (tô thành nền) ❌ ← kẻ thù tệ nhất

**Dice Loss:** `Dice = 2·TP / (2·TP + FP + FN)` — càng trùng càng gần 1, nền to mấy cũng kệ. Nửa còn lại của đề cũ (`classes=defect_classes` bỏ nền).
👎 Nhược: **phạt FP và FN ngang nhau**, nhưng ta cần phạt bỏ sót (FN) nặng hơn.

**Tversky (= Dice có núm α, β):** `Tversky = TP / (TP + α·FP + β·FN)`

| Núm | Phạt | Tăng lên thì… |
|-----|------|---------------|
| **α** (`--tv-alpha`) | FP = báo nhầm | bớt báo nhầm → precision ↑ |
| **β** (`--tv-beta`) | FN = bỏ sót | bớt bỏ sót → **recall ↑** |

- α = β = 0.5 → **chính là Dice**.
- Code: α=0.3, β=0.7 → β>α → **nghiêng về recall** (thà báo nhầm hơn bỏ sót).

**FocalTversky:** `(1 − Tversky)^γ` — núm **γ** (`--tv-gamma`=1.333): γ>1 → **dồn sức vào cục lỗi bé còn tô trượt** (Focal ở cấp độ vùng).

## Ghép lại — "đề mới" `--loss ftl_focal`
```python
ftl_loss   = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.333)  # đo vùng, nghiêng recall
focal_loss = FocalLoss(gamma=2.0)                            # đếm pixel, dồn ca khó
criterion  = ftl_loss + focal_loss
```

### Bảng tóm tắt cả bài
| Loss | Trường phái | Chống nền nhấn chìm? | Núm chỉnh recall? |
|------|-------------|:---:|:---:|
| CrossEntropy | đếm pixel | ❌ (chỉ nhờ weights) | ❌ |
| Focal | đếm pixel | 一 phần | ❌ (chỉ dồn ca khó) |
| Dice | đo vùng | ✅ | ❌ (FP=FN) |
| **Tversky** | đo vùng | ✅ | ✅ α, β |
| **FocalTversky** | đo vùng | ✅ | ✅ α,β + γ |

### ✅ Câu hỏi Bài 5 — ĐÃ CHẤM
1. **FP vs FN** — đáp án: **FN = False Negative = BỎ SÓT** (có lỗi thật mà báo "không") ← *nguy hiểm nhất* (hàng lỗi lọt ra khách). **FP = False Positive = BÁO NHẦM** (không lỗi mà báo "có") ← đỡ hơn, chỉ tốn công soi lại.
   - *Mẹo nhớ:* nhìn chữ cuối — **N**egative = model nói "Không"; False-Negative = nói "Không" mà sai = thực ra CÓ = bỏ sót. **P**ositive = model nói "Có"; False-Positive = nói "Có" mà sai = báo nhầm.
   - *Lần này mình lỡ dán nhầm nhãn FP↔FN nhưng hiểu đúng bản chất (bỏ sót nguy hiểm nhất ✅).*
2. **Vặn β LÊN cao** ✅ — β phạt FN (bỏ sót); phạt càng đau → recall ↑. (Trả lời đúng.)
3. **α = β = 0,5 → Dice** (thế số vào Tversky, nhân 2 tử-mẫu ra đúng công thức Dice). Tversky = "Dice có thêm 2 núm α,β"; vặn về giữa thì thành Dice. (*Mình đoán "anomaly detection" — chưa đúng, đó là hướng bài toán khác.*)

---

# ✅ Bài 6 — Thước đo (metric) & chọn best.pt

**Loss vs Metric:** Loss = thầy la học sinh *trong lúc học* (MÁY đọc để tự chỉnh, cần mượt/đạo hàm). Metric = bảng điểm *cuối kỳ* (NGƯỜI đọc, dùng để quyết lưu `best.pt` của epoch nào; chỉ cần đếm).

**IoU per-pixel lừa mình 2 cách với lỗi 6px:**
1. *Lệch nhẹ rớt điểm thảm:* lỗi 36px lệch 2px → IoU ~0,38; cùng lệch đó trên vật 200×200 → IoU ~0,98. Vật càng bé IoU phạt càng nặng (vì vài px lệch = % lớn). Người kiểm hàng chỉ hỏi "có trúng cục lỗi không" → thấy bất công.
2. *Trung bình dìm điểm:* nền 99,98% → điểm nền hoàn hảo kéo trung bình lên cao dù bỏ sót cả cục lỗi (biển nền nhấn chìm — lặp lại Bài 3, lần này ở khâu chấm).
→ best.pt theo IoU có xu hướng lưu model "chơi an toàn, tô ít, né lỗi nhỏ" — thứ KHÔNG muốn.

**Lời giải — chấm theo "cục lỗi" (blob / object-level):**
- Blob = 1 cục pixel lỗi dính liền (1 con lỗi 6×6 = 1 blob).
- Model chạm ≥1px vào cục thật → TP object; cục thật không chạm → FN (bỏ sót); bịa cục ở nền → FP (báo nhầm).
- **Recall = bắt được / tổng cục thật** ("bỏ sót bao nhiêu?") ← QUAN TRỌNG NHẤT với bài này (bỏ sót = hàng lỗi ra khách).
- Precision = báo đúng / tổng báo ("báo nhầm nhiều không?"). F1 = cân bằng 2 cái.
→ Nên chọn best.pt theo **Recall/F1 object-level**, không phải IoU per-pixel.

**Nối Bài 5:** núm β (`--tv-beta`) trong Loss DẠY model đừng bỏ sót → Metric Recall CHẤM xem nó có thật sự không bỏ sót. Cùng nhắm 1 mục tiêu.

### ✅ Câu hỏi Bài 6 — ĐÃ CHẤM
1. Loss = MÁY đọc tự chỉnh (mượt); Metric = NGƯỜI đọc, chọn best.pt (đếm). *(User ví von thầy-la/bảng-điểm — đúng, bổ sung "ai đọc".)*
2. IoU chấm THẤP; người kiểm hàng thấy bất công — *vì cục lỗi quá NHỎ nên vài px lệch = % lớn; 2 bên hỏi 2 câu khác nhau (trùng khít % vs có trúng cục không).* ✅
3. **Recall** quan trọng nhất = bắt được / tổng lỗi thật → recall thấp = bỏ sót nhiều = thảm họa. ✅

---

# ✅ Bài 7 — Kiến trúc (Unet vs Unet++/MAnet)
- Skip là vị cứu tinh lỗi nhỏ → càng nhiều/dày skip càng cứu tốt. Unet = 1 đường skip thô/tầng; Unet++ = skip lồng nhau nhiều trạm (nhạy vật nhỏ); MAnet = thêm attention (dồn ánh nhìn vào cụm lỗi, dẹp nền). Giá: nặng/chậm/tốn RAM hơn.
- **backbone = encoder = resnet34 = nửa trái chữ U** (3 chữ cùng 1 thứ — user từng gợn chữ này, đã gỡ).
- Đổi arch KHÔNG chắc thắng → phải giữ Unet baseline + chạy ablation, đo bằng Recall object-level. "Nghe nói xịn" không phải bằng chứng. (Trong code chỉ là cờ `--arch`.)

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

**2 câu HỎI USER còn treo (hỏi đầu buổi sau):**
1. Data đã sẵn chưa? (`dataset.yaml` dòng 26 train_unet.py — ảnh + mask thật đã có chưa, hay phải lo khâu cắt tile/gán nhãn trước?)
2. Buổi sau muốn CHẠY THẬT (kick off exp_A trong WSL, đọc log cùng nhau) hay chỉ cần hiểu cách chạy?

⚠️ Nhắc lại từ Bài 6: code hiện chọn best.pt theo IoU per-pixel (dòng 266) — CHƯA nâng cấp sang Recall object-level. Đây là việc code còn nợ.

---

## 🔧 Ghi chú kỹ thuật
- Đã sửa lỗi cú pháp dòng 129 `train_unet.py` (xóa chữ `k` thừa sau `default=0.7,`). File giờ chạy được.
- Công tắc trong code: `--loss dice_ce` (baseline) vs `--loss ftl_focal` (đề mới cho lỗi nhỏ). Các núm: `--tv-alpha --tv-beta --tv-gamma --focal-gamma`.
