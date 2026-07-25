# 📍 Tiến độ & Mục tiêu — đọc file này TRƯỚC, mỗi lần quay lại

> Đây là bảng điều khiển duy nhất cần đọc để biết: đang ở đâu, đích đến là gì, việc tiếp theo là gì.
> Các file `HOC_*.md` khác là NỘI DUNG CHI TIẾT — chỉ mở khi cần tra/học lại, không cần đọc hết mỗi lần.
> Cập nhật: 2026-07-25.

---

## 🎯 Mục tiêu lớn
Từ web/WinForm CRUD → làm chủ machine vision công nghiệp: vừa làm dự án thật ở công ty (JeaYoung/MLCC,
SIBV A26/A27), vừa tự học nền tảng qua paper/benchmark công khai để không phụ thuộc "biết dùng tool"
mà hiểu được BẢN CHẤT (tự dựng được khi cần, không chỉ kéo-thả Cognex).

Bản đồ chọn nhánh (OpenCV luật vs Deep Learning) + playbook 3 ngày cho dự án mới: xem `HOC_vision.md`
(file đó là bản đồ để TRA, không phải để đọc lại từ đầu).

---

## 📏 ĐÍCH ĐẾN xuyên suốt mọi bài toán (chốt 2026-07-25) — không cần biết làm cách nào, kết quả phải đạt

Vai trò: lập trình viên machine vision cho 1 vendor công nghiệp (khách kiểu Samsung). Đây là mục tiêu
CUỐI của MỌI bài toán từ nay — bất kể phương pháp (OpenCV hay Deep Learning), kết quả phải đạt đủ 4 số:

| Trục | Mục tiêu | Đo bằng | Đơn vị đếm |
|---|---|---|---|
| Bắt đúng hàng lỗi | **Recall ≥ 99%** (FN ≤ 1%) | `test_kolektorsdd2.py` khối "CON HÀNG" | 1 SẢN PHẨM (con hàng), KHÔNG phải 1 cục lỗi |
| Không báo nhầm hàng tốt | **FP rate ≤ 1%** | như trên | 1 sản phẩm |
| Repeatability | **flip rate ≤ 1%** sau 10 lần chạy/sản phẩm | CHƯA có script đo — bug đang debug dở, mục 🔥 Ưu tiên số 1 | 1 sản phẩm |
| Tốc độ | tuỳ bài (VD ≤100ms/ảnh cho 1 mặt/4 con hàng) | đo thời gian inference thật | 1 ảnh |

**Lưu ý đơn vị đếm quan trọng:** Recall/FN/FP ở đây tính theo **CON HÀNG** (cả sản phẩm pass/fail), khác
"Recall cục lỗi" mà `object_stats()` tính bấy lâu (đếm theo blob). 2 con số đo 2 câu hỏi khác nhau — xem
`HOC_paper_kolektorsdd2_mixed_supervision.md` mục 2 để ôn lại phân biệt image-level vs object-level.

**Đã code xong (2026-07-25):** `test_kolektorsdd2.py` giờ tính thêm khối "CON HÀNG" (TP/FP/FN/TN ở mức
1 ảnh = 1 sản phẩm: ảnh có lỗi thật không, model có báo lỗi không) + xuất `hang_sai_capdoconhang.txt`
liệt kê tên từng hàng bị bỏ sót/báo nhầm.

**Hiện trạng so với đích (đo thật 2026-07-25, toàn bộ 1004 ảnh test):**

| Bản | Recall con hàng | FN rate | FP rate |
|---|---|---|---|
| Mục tiêu | ≥99% | ≤1% | ≤1% |
| `results_260721` (tốt nhất hiện có) | 88.2% | 11.8% | 0.67% ✅ đạt |
| `results_260723_2` (mặc định trong script) | 80.0% | 20.0% | 0.45% ✅ đạt |

→ **FP rate đã đạt mục tiêu ở cả 2 bản.** Khoảng cách còn lại nằm gần hết ở Recall/FN (thiếu ~11 điểm %
ngay cả ở bản tốt nhất) — đây là trục cần ưu tiên train tiếp, không cần vặn cả 2 phía.

---

## ✅ Đã xong — không cần học lại, chỉ tra khi cần

- **UNet cho lỗi nhỏ, nền tảng (Bài 1-8):** segmentation là gì, kiến trúc UNet, vì sao lỗi 6px khó
  (class imbalance + downsample), loss (CE→Dice→Tversky→FocalTversky), metric (IoU per-pixel vs
  Recall/F1 object-level), kiến trúc (Unet/Unet++/MAnet), đọc hiểu `train_unet.py`.
  → Chi tiết (đã rút gọn): `HOC_unet_loi_nho_6px.md`.
- **Cognex → OpenCV, tư duy hệ tọa độ (mục 1-4):** bảng dịch tool Cognex sang kỹ thuật OpenCV, 3 hệ tọa độ
  (pixel/part/world), thang thuật toán Align (template → geometric → feature → ECC refine).
  → Chi tiết: `HOC_align_cognex_opencv.md` mục 1-4.

---

## 🔥 Ưu tiên số 1 — bug sản xuất thật (không phải tự học)

- **Mất tính lặp lại (repeatability), dung sai ±0.03mm:** cùng 1 ảnh chạy lại nhiều lần → NG/OK khác
  nhau, hoặc đo ra 0.05mm rồi 0.08mm (lệch giữa các lần chạy CÒN LỚN HƠN cả dung sai chấp nhận). Đang ở
  **Round 1 — Q1 đã trả lời một phần (2026-07-21) bằng quan sát thật, Q2-5 CHƯA trả lời**:
  - Tách được: cùng 1 **file ảnh cố định** chạy lại nhiều lần → **NG/OK ổn định, không đổi**. Chạy
    **hàng thật** (ảnh chụp lại mỗi lần, rung/ánh sáng nhẹ) → **NG/OK có đổi**. → gợi ý nguyên nhân
    nghiêng về ảnh chụp khác nhau + ngưỡng quyết định nhạy biên (Q4), nhưng CHƯA làm thí nghiệm chứng
    minh, mới là suy luận.
  - Riêng phép **đo mm**: cùng 1 file ảnh cố định, chạy lại → **vẫn ra sai số** (dù nhỏ). Đây là tín
    hiệu bất ổn phần mềm thật (không do ảnh khác nhau) — việc tiếp theo: lưu mask model xuất ra ở 2-3
    lần chạy trên CÙNG 1 ảnh, so pixel-by-pixel xem có bit-identical không (Q5, CHƯA làm).
  - "Ăn vào background" (mask lem ra ngoài vùng lỗi thật) được nhắc tới trong buổi 2026-07-21 nhưng
    CHƯA xác định là (a) lỗi hệ thống độc lập hay (b) một phần của repeatability (lem nhiều/ít khác
    nhau giữa các lần chạy) — PARKED, cần quay lại xác định trước khi debug tiếp.
  → Chi tiết: `HOC_repeatability_debug.md`.

## 🔥 Ưu tiên số 2 (2026-07-21, cập nhật 2026-07-25) — Generalization: model không tổng quát hóa tốt ra ảnh chưa từng thấy

- Model công ty train trên 500 ảnh (train/val nội bộ), giữ riêng 100 ảnh chưa từng đưa vào train/val
  để test → có cả **false positive** (báo lỗi sai chỗ bình thường) và **false negative** (bỏ sót lỗi
  thật). Checkpoint + log của lần train đó đã MẤT (đổi qua lại máy công ty/nhà, máy nhà mất code) —
  không truy lại được train/val gap lúc train.
- Khái niệm: **overfitting / generalization gap** — bình thường, gần như ai học deep learning cũng gặp,
  không phải lỗi làm sai gì đặc biệt.
- **Phát hiện phụ áp dụng chéo sang KolektorSDD2, ĐÃ SỬA (2026-07-21):**
  `tu_hoc_deep/chia_data_kolektorsdd2.py` từng map thẳng bộ `test/` gốc (bộ đánh giá chính thức) thành
  `val` — mà `val` lại dùng để **chọn `best.pt`** trong lúc train, làm bẩn bộ test, không còn holdout
  thật. Đã sửa: `train/` gốc tự tách 90/10 thành train+val (seed cố định `--seed 42`, chỉnh được qua
  `--val-ratio`), `test/` gốc giữ nguyên thành 1 split `test` riêng — `train_kolektorsdd2.py` không đọc
  key này nên không đụng tới lúc train.
  → Chi tiết: `HOC_kolektorsdd2_data_prep.md` mục 4.
- **Vòng 1 (2026-07-23):** `tu_hoc_deep/test_kolektorsdd2.py` (mới viết) chạy trên `test/` holdout thật
  (1004 ảnh, kết quả ở `results_260721`) → **Recall cấp cục lỗi 87.4%** (sót 15/119 lỗi thật = 12.6%,
  Precision 91.2%, F1 0.893) — xác nhận đúng mẫu hình FP+FN đã thấy ở model công ty (cùng nguyên nhân gốc).
- **Vòng 2 (train 2026-07-23, đọc lại số liệu 2026-07-25) — CẢ 2 LẦN THỬ TĂNG RECALL ĐỀU RA KẾT QUẢ TỆ
  HƠN, không tốt hơn:**

  | Run | Recall | Precision | sót (FN) | nhầm (FP) |
  |---|---|---|---|---|
  | `results_260721` (Vòng 1) | **87.4%** | 91.2% | 15/119 | 10 |
  | `results_260723_1` | 77.3% | 92.9% | 27/119 | 7 |
  | `results_260723_2` | 79.8% | 81.9% | 24/119 | 21 |

  **Nguyên nhân đã xác nhận (2026-07-25, tự đọc code):** `--conf` trong `test_kolektorsdd2.py` KHÔNG hề
  áp vào Recall/Precision (chỉ ảnh hưởng ảnh overlay — `object_stats()` luôn nhận `pred` từ `argmax`,
  tương đương ngưỡng 0.5 cứng) → đòn bẩy "hạ ngưỡng" trong 3 đòn bẩy dự định CHƯA từng được đo thật ở
  Vòng 2. Recall tụt phải đến từ 2 nguyên nhân còn lại (đổi loss/`tv-beta` cao hơn cả mức đề nghị, hoặc
  val set ~233 ảnh quá nhỏ làm `best.pt` chọn nhầm epoch may rủi — xem log dao động mạnh giữa epoch).
  → Chi tiết + kế hoạch làm lại đúng cách: `HOC_kolektorsdd2_data_prep.md` mục 8.

**Phương pháp học đang áp dụng (áp dụng cho MỌI buổi học sau, không chỉ 1 file) — gộp lại 2026-07-25:**
- **Socratic (từ 2026-07-19):** trước khi giải thích, đặt câu hỏi chẩn đoán bám vào code/số liệu cụ thể
  đang có trong tay, để tự trả lời bằng quan sát/thí nghiệm thật — không lộ đáp án trước. Lý do: gap thật
  không phải thiếu kiến thức lẻ, mà chưa chắc biết **chẻ vấn đề đúng lớp nguyên nhân** và chưa biết
  **tự kiểm tra hiểu thật hay chỉ tưởng hiểu**.
  - Ngoại lệ: hỏi thẳng "X là gì, tôi chưa hiểu" về khái niệm không có cách tự suy ra → giải thích trực
    tiếp luôn, không hỏi ngược.
- **1 chủ đề / 1 lúc (từ 2026-07-21):** nhiều vấn đề trộn 1 buổi → chọn 1 cái đào sâu, các cái khác PARK
  (ghi lại rõ ràng, không im lặng bỏ qua), không dồn cùng lúc (dễ "không load được").
- **Chia nhỏ khi giải thích (từ tối 2026-07-25):** kể cả giải thích trực tiếp cũng không dồn nhiều định
  dạng (diagram + bảng + kế hoạch) trong 1 tin — nói 1 ý, dừng, rồi mới sang ý kế.
- **Neo vào công thức toán + hình dung cụ thể (từ 2026-07-25):** mục tiêu học là hiểu **bản chất toán**
  của mỗi khái niệm/metric mới (không chỉ định nghĩa bằng lời) để sau này tự biết tối ưu tham số — vì học
  kiểu chỉ nghe mô tả xong dễ "nhớ nhớ quên quên". Mỗi khi giới thiệu 1 công thức/metric mới, phải có đủ 3
  phần: (1) công thức thật (tử số/mẫu số hoặc phương trình), (2) 1 ví dụ số cụ thể nhỏ, tính tay ra được,
  (3) một câu "hình dung" (ảnh/ẩn dụ) để neo trí nhớ — không dừng ở mô tả bằng lời suông.
- **Đọc code/pipeline: lần theo 1 GIÁ TRỊ THẬT, không lần theo Ý NGHĨA dòng code (mới, 2026-07-26):** gap
  đã tự chẩn đoán khi sửa bug `--conf` trong `test_kolektorsdd2.py`: hiểu được "dòng này làm gì" (cấp độ
  dòng) nhưng KHÔNG ghép được thành "dữ liệu chảy từ đâu tới đâu" (cấp độ pipeline) — không phải yếu code,
  chỉ là 2 kỹ năng đọc khác nhau. Đây là kỹ năng CHUNG cho mọi loại code (deep learning, app, game, web),
  không riêng project này. Kỹ thuật: khi đọc pipeline/hàm mới, chọn **1 giá trị cụ thể** (VD 1 tham số
  CLI, 1 pixel, 1 request) rồi TỰ viết ra giá trị của nó ở TỪNG bước nó đi qua từ đầu vào tới lúc dùng ở
  cuối — thay vì nghe/đọc mô tả "hàm này làm gì". Việc tự điền số vào từng bước mới là bước thật sự ghép
  "bức tranh tổng" vào đầu; nghe giải thích suông không làm được việc này.

---

## 🟡 Đang làm (tự học, ưu tiên sau mục 🔥 ở trên)

- **Bài 9 — Ablation UNet (exp_A/B/C: loss dice_ce vs ftl_focal, arch Unet vs UnetPlusPlus).**
  Bài học lớn đã rút ra: soi phân bố lớp train/val TRƯỚC khi đổ lỗi model.
  → Chi tiết + lệnh chạy: `HOC_unet_loi_nho_6px.md` § Bài 9.
- **KolektorSDD2 (tự học qua benchmark công khai):** adapter + train + script đánh giá riêng đã xong và
  ĐÃ CHẠY THẬT với split đúng (không còn leakage) — `tu_hoc_deep/chia_data_kolektorsdd2.py` +
  `tu_hoc_deep/train_kolektorsdd2.py` + `tu_hoc_deep/test_kolektorsdd2.py`. **Đang làm (2026-07-23):**
  Recall cấp cục lỗi trên test holdout chỉ 87.4% — đang thử đòn bẩy tăng Recall (threshold/loss/
  best-metric), xem mục 🔥 Ưu tiên số 2. Round 2 (câu hỏi về optimizer/scheduler/AMP/checkpoint trong
  `train_kolektorsdd2.py`) vẫn đang HOÃN, chờ xong Round 1 ở mục 🔥 Ưu tiên số 1. → Chi tiết:
  `HOC_kolektorsdd2_data_prep.md`, cách chạy: `tu_hoc_deep/README.md`, câu hỏi Round 2:
  `HOC_repeatability_debug.md` § 4.

---

## ⬜ Còn nợ / sắp tới

- Align/Cognex: tự dựng geometric edge-model (lõi PatMax), coarse-to-fine pyramid, hiểu ECC refine,
  rồi ráp pipeline Align→Fixture→Caliper→đo mm end-to-end trên 1 ảnh thật.
  → `HOC_align_cognex_opencv.md` mục 5 ("còn nợ").
- **KolektorSDD2 (nâng ưu tiên — đang chặn việc biết đòn bẩy nào thật sự ăn), phân công 2026-07-25:**
  1. **(người học tự vá)** `--conf` trong `test_kolektorsdd2.py` áp thẳng vào `object_stats()`, không chỉ
     `render()` — bug xác nhận 2026-07-25, xem `HOC_kolektorsdd2_data_prep.md` mục 8.
  2. Lưu hyperparameter mỗi lần train (`--loss --tv-beta --tv-gamma --focal-gamma --best-metric`) ra 1
     file cạnh `model_cfg.yaml` — hiện KHÔNG lưu, `results_260723_1`/`_2` phải suy ngược từ mặc định
     trong code mới biết đã chạy gì.
  3. Sau khi (1)(2) xong: chạy lại đúng 1 biến mỗi lần (thêm nấc `tv-beta 0.85` chưa từng thử), quét
     `--conf` thật trên CÙNG 1 `best.pt` để tách ảnh hưởng threshold khỏi ảnh hưởng loss.
  → chi tiết + bảng số liệu Vòng 1 vs Vòng 2: `HOC_kolektorsdd2_data_prep.md` mục 6-8.
- **Hướng tiếp theo (xác nhận với người học 2026-07-25):** sau khi KolektorSDD2 (1 lớp lỗi + nền) ổn
  định, mục tiêu kế tiếp là bài toán NHIỀU lớp lỗi nhỏ cùng lúc (đúng dạng sẽ gặp nhiều ở công ty) — xem
  playbook mới `HOC_danh_gia_model_va_multi_class.md`.
- Data công ty (500 train/val + 100 test): kiểm tra lại xem 500 ảnh có đang tách train/val đúng chuẩn
  không (val có bị lẫn với bộ dùng để test cuối không, giống lỗi vừa sửa ở KolektorSDD2) — cần làm khi
  có model/log train mới. → `HOC_generalization_overfitting.md` mục 7.
- **(nặng, làm SAU khi có baseline UNet ở trên)** Tái hiện paper gốc KolektorSDD2 (mixed supervision,
  2 mạng segmentation+decision, dùng đúng `split_weakly_*.pyb`) — bài tập đọc-paper-chủ-động, trả lời
  câu hỏi trước khi code. → `HOC_paper_kolektorsdd2_mixed_supervision.md`.

## ✅ (mới dời từ "còn nợ", đã xong)
- `train_unet.py` chọn `best.pt` theo Recall/F1 object-level thay IoU per-pixel — code đã có cờ
  `--best-metric` (mặc định `f1_object`), xác nhận lại 2026-07-19.

---

## Cách dùng file này
Mỗi lần quay lại sau vài ngày: đọc **Đang làm** + **Còn nợ** trước. Chỉ mở lại phần **Đã xong** khi cần
tra công thức/số liệu cụ thể — không cần đọc lại để "ôn", vì đã học xong rồi (dấu hiệu ✅ trong từng file
chi tiết). Khi 1 việc trong "Đang làm"/"Còn nợ" xong → dời xuống "Đã xong" và rút gọn ghi chú chi tiết
tương ứng (bỏ phần hỏi-đáp/nháp, giữ công thức/kết luận cốt lõi) — như đã làm với Bài 1-8.
