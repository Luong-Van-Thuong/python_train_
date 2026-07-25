# 🎓 Ghi chú: Model không tổng quát hóa tốt — overfitting trên 100 ảnh test rời (buổi 2026-07-21)

> Vấn đề sản xuất thật (công ty), phát hiện khi tách riêng 100 ảnh chưa từng đưa vào train/val để test.
> Trạng thái tổng: `TIEN_DO.md` (🔥 Ưu tiên số 2). Phương pháp: Socratic, xem `HOC_repeatability_debug.md`
> mục đầu file cho quy ước chung.

---

## 1. Hiện tượng

- Train model trên 500 ảnh (chia train/val nội bộ), giữ riêng thêm 100 ảnh KHÔNG đưa vào train hay val.
- Chạy inference 100 ảnh đó: có ảnh bình thường bị báo thành lỗi (**false positive**), có lỗi thật bị
  bỏ sót (**false negative**).
- Checkpoint + log của lần train đó hiện đã MẤT (đổi qua lại máy công ty/nhà, máy nhà bị mất code) —
  không còn cách xem lại train vs val metric lúc train để biết đã có dấu hiệu overfit từ trong lúc
  train hay chưa.

## 2. Đã loại được 1 khả năng

100 ảnh test lấy **ngẫu nhiên cùng nguồn** với 500 ảnh train/val (cùng đợt chụp, cùng máy, cùng điều
kiện) — không phải khác lô/khác điều kiện chụp. Nên nguyên nhân không phải do lệch **phân bố nguồn dữ
liệu** một cách hệ thống (khác ánh sáng/camera/lô hàng); nếu còn lệch, nhiều khả năng chỉ là may rủi
lấy mẫu ngẫu nhiên (chưa đo tỉ lệ ảnh lỗi/không-lỗi giữa 2 tập để xác nhận — còn nợ).

## 3. Khái niệm: Overfitting / generalization gap

Model không học "lỗi là gì" ở mức tổng quát — nó khớp với đúng những gì thấy trong tập train. Với 500
ảnh (ít, đặc biệt nếu 1 dạng lỗi/nền chỉ xuất hiện vài lần), model dễ học thuộc các mẫu cụ thể thay vì
đặc trưng chung:

- **False positive**: vùng bình thường có texture/ánh sáng giống 1 mẫu lỗi đã học thuộc.
- **False negative**: dạng lỗi khác với những gì đã thấy (khác size/hướng/độ tương phản).

Đây là lỗi gần như ai học deep learning cũng gặp — không phải làm sai gì đặc biệt, mà là hệ quả tự
nhiên của data ít + chưa kiểm chứng đúng cách.

## 4. Phát hiện phụ quan trọng — áp dụng chéo sang KolektorSDD2

Khi bàn cách đánh giá đúng (không test trên ảnh model đã thấy), phát hiện
`tu_hoc_deep/chia_data_kolektorsdd2.py` mắc đúng lỗi phương pháp luận liên quan: map thẳng `test/` gốc
(bộ đánh giá chính thức của KolektorSDD2) thành `val` — mà `val` lại dùng để **chọn `best.pt`** trong
`train_kolektorsdd2.py` (cơ chế ở mục 5 dưới). Nghĩa là bộ test "sạch" đã bị dùng để chọn model, không
còn ý nghĩa test mù nữa.

**Đã sửa (2026-07-21):** `train/` gốc (2332 ảnh) tự tách 90/10 thành train+val (seed cố định
`--seed 42`, tỉ lệ chỉnh qua `--val-ratio`). `test/` gốc (1004 ảnh) giữ nguyên thành 1 split `test`
riêng, ghi vào `dataset.yaml` nhưng `train_kolektorsdd2.py` không đọc key này — không đụng tới lúc
train/chọn best.pt. → chi tiết code: `HOC_kolektorsdd2_data_prep.md` mục 4.

## 5. Cơ chế "val dùng để chọn best.pt" (giải thích đã học buổi này)

Mỗi epoch trong `train_kolektorsdd2.py` (dòng 284-320): (1) học trên ảnh **train** (update trọng số),
(2) chấm điểm trên ảnh **val** (không update trọng số) — nếu điểm val cao hơn điểm tốt nhất từng thấy,
**lưu đè** trọng số hiện tại thành `best.pt`. Val không dạy model, nhưng **quyết định giữ lại phiên bản
nào** trong số các epoch đã train — nên nếu 1 bộ ảnh nào đó (như test) bị dùng làm val, model cuối cùng
bị "chọn để hợp" với đúng bộ đó, đo lại trên nó sau này sẽ lạc quan hơn thực tế (không phải test mù).

## 6. KolektorSDD2 rep hoàn tất vòng 1 (2026-07-23) — xác nhận đúng hiện tượng ở mục 1

`chia_data_kolektorsdd2.py` (split đúng) + `train_kolektorsdd2.py` + script đánh giá riêng
(`tu_hoc_deep/test_kolektorsdd2.py`, mới viết) đã chạy xong trên bộ `test/` holdout thật (1004 ảnh,
chưa từng đụng lúc train/chọn best.pt). Kết quả: Recall cấp cục lỗi chỉ **87.4%** (sót 15/119 lỗi thật =
12.6%), Precision 91.2% — nghĩa là **có cả sót lẫn nhầm cùng lúc**, đúng mẫu hình đã thấy ở model công
ty trong mục 1 (không phải trùng hợp riêng của KolektorSDD2 — cùng nguyên nhân gốc: model chưa tổng quát
hóa đủ tốt). Số liệu đầy đủ + đòn bẩy đang thử để tăng Recall: `HOC_kolektorsdd2_data_prep.md` mục 6-7.

## 7. Còn nợ / bước tiếp theo

- [ ] Áp dụng lại đúng nguyên tắc 3-way split (train/val/test) cho **data công ty thật** (500/100 ảnh)
      — hiện chưa rõ 500 ảnh có tách train/val đúng cách không (val có bị dùng để chọn best.pt rồi sau
      đó lại đánh giá "tổng quát hóa" trên chính val không?) — CẦN KIỂM TRA khi có model/log mới.
- [ ] Thiết lập quy trình lưu checkpoint + log train (tránh lặp lại việc mất dữ liệu do đổi qua lại máy
      công ty/nhà).
- [ ] Hướng xử lý overfitting/Recall thấp (data augmentation, thêm data đa dạng, regularization, đổi
      loss/threshold) — đang thử trước trên KolektorSDD2 (rẻ, nhanh lặp) ở `HOC_kolektorsdd2_data_prep.md`
      mục 6-7, đòn bẩy nào ăn sẽ áp dụng ngược lại cho data công ty.
