# 📍 Tiến độ & Mục tiêu — đọc file này TRƯỚC, mỗi lần quay lại

> Đây là bảng điều khiển duy nhất cần đọc để biết: đang ở đâu, đích đến là gì, việc tiếp theo là gì.
> Các file `HOC_*.md` khác là NỘI DUNG CHI TIẾT — chỉ mở khi cần tra/học lại, không cần đọc hết mỗi lần.
> Cập nhật: 2026-07-19.

---

## 🎯 Mục tiêu lớn
Từ web/WinForm CRUD → làm chủ machine vision công nghiệp: vừa làm dự án thật ở công ty (JeaYoung/MLCC,
SIBV A26/A27), vừa tự học nền tảng qua paper/benchmark công khai để không phụ thuộc "biết dùng tool"
mà hiểu được BẢN CHẤT (tự dựng được khi cần, không chỉ kéo-thả Cognex).

Bản đồ chọn nhánh (OpenCV luật vs Deep Learning) + playbook 3 ngày cho dự án mới: xem `HOC_vision.md`
(file đó là bản đồ để TRA, không phải để đọc lại từ đầu).

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
  **Round 1 — 5 câu hỏi chẩn đoán CHƯA trả lời** (khoanh vùng: chế độ model / GPU non-determinism /
  ngưỡng quyết định nhạy biên / tầng đo kế thừa bất ổn từ mask). Việc tiếp theo khi quay lại: TRẢ LỜI
  5 câu đó bằng quan sát thật trên pipeline của mình, rồi làm thí nghiệm chứng minh bằng số.
  → Chi tiết: `HOC_repeatability_debug.md`.

**Phương pháp học đang áp dụng (mới, 2026-07-19):** Socratic — Claude đặt câu hỏi chẩn đoán trước, tự
trả lời bằng quan sát/thí nghiệm thật rồi mới nghe giải thích, KHÔNG đọc đáp án trước. Lý do: gap thật
không phải thiếu kiến thức lẻ mà là chưa chắc biết **chẻ vấn đề đúng lớp nguyên nhân** và chưa biết
**tự kiểm tra hiểu thật hay chỉ tưởng hiểu**. Áp dụng cho mọi buổi học sau, không chỉ file này.

---

## 🟡 Đang làm (tự học, ưu tiên sau mục 🔥 ở trên)

- **Bài 9 — Ablation UNet (exp_A/B/C: loss dice_ce vs ftl_focal, arch Unet vs UnetPlusPlus).**
  Bài học lớn đã rút ra: soi phân bố lớp train/val TRƯỚC khi đổ lỗi model.
  → Chi tiết + lệnh chạy: `HOC_unet_loi_nho_6px.md` § Bài 9.
- **KolektorSDD2 (tự học qua benchmark công khai):** đã viết xong adapter + train script riêng, TÁCH
  KHỎI `src/` để không đụng code sản xuất — `tu_hoc_deep/chia_data_kolektorsdd2.py` +
  `tu_hoc_deep/train_kolektorsdd2.py`. Còn thiếu bước CHẠY THỬ THẬT + đọc log. Round 2 (câu hỏi về
  optimizer/scheduler/AMP/checkpoint trong `train_kolektorsdd2.py`) đang HOÃN, chờ xong Round 1 ở mục
  🔥 trên. → Chi tiết: `HOC_kolektorsdd2_data_prep.md`, cách chạy: `tu_hoc_deep/README.md`, câu hỏi
  Round 2: `HOC_repeatability_debug.md` § 4.

---

## ⬜ Còn nợ / sắp tới

- Align/Cognex: tự dựng geometric edge-model (lõi PatMax), coarse-to-fine pyramid, hiểu ECC refine,
  rồi ráp pipeline Align→Fixture→Caliper→đo mm end-to-end trên 1 ảnh thật.
  → `HOC_align_cognex_opencv.md` mục 5 ("còn nợ").
- KolektorSDD2: chạy thử thật `tu_hoc_deep/train_kolektorsdd2.py` (code đã xong — xem "Đang làm" ở trên),
  đọc log, so kết quả. → chi tiết: `HOC_kolektorsdd2_data_prep.md` mục 5.
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
