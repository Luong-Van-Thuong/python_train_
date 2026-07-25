# 🎓 Ghi chú: Đưa KolektorSDD2 vào `train_unet.py`

> Tự học qua benchmark công khai. Trạng thái tổng: xem `TIEN_DO.md`.
> Mục tiêu: tự đi từ data thô lạ (không phải data công ty, không qua `chia_data_unet.py` có sẵn) đến
> model train được bằng `train_unet.py` — để chứng minh hiểu THẬT pipeline, không chỉ chạy được lệnh có sẵn.

---

## 1. Vì sao KHÔNG đi qua `chia_data_seg_obd.py` / `chia_data_unet.py`

Cả 2 script đó đọc nhãn từ file `.json` labelme (polygon vẽ tay). KolektorSDD2 không có `.json` — nó cho
sẵn **mask pixel** (`{id}_GT.png`), tức bước "vẽ polygon → tô mask" mà `chia_data_unet.py` làm đã có kết
quả cuối rồi. Đi qua JSON là vòng thừa. Cách ngắn nhất: viết 1 script nhỏ đưa thẳng KolektorSDD2 vào
format mà `train_unet.py` đọc (`images/`, `masks/`, `dataset.yaml`) — không cần `chia_data_unet.py`.

## 2. Format `train_unet.py` thực sự cần (đọc từ code, không đoán)

- `dataset.yaml`: khoá `path`, `images: {train, val}`, `masks: {train, val}`, `num_classes`, `names`.
- Với mỗi ảnh `images/<split>/X.ext`, mask phải nằm ở `masks/<split>/X.png` — **tên phải trùng stem**.
- Mask đọc bằng `cv2.IMREAD_GRAYSCALE` rồi ép thẳng thành **class id** (`mask.astype(np.int64)`) — nghĩa
  là pixel phải là `0,1,2,...` (số lớp), KHÔNG phải 0/255.
- `weights_list = [0.2, 2.0, 2.0, 1.5, 2.0]` hard-code trong `main()` (dòng ~234), có `assert
  len(weights_list) == num_classes`. Đây là trọng số cho bài toán 4-lỗi công ty — SẼ CRASH nếu chạy
  bài 1-lớp mà không sửa.

## 3. Số liệu thật đã đo trên KolektorSDD2 (2026-07-19)

- Ảnh KHÔNG cùng kích thước: `10000.png`=229×645, `10001.png`=228×633, `20000.png` (test)=231×636 —
  lệch vài pixel/ảnh, không chia hết 32 → phải resize/pad về 1 size cố định chia hết 32 trước khi train
  (UNet encoder nén 5 lần /2 — xem Bài 2 `HOC_unet_loi_nho_6px.md`).
- `train/`: 2332 ảnh (không tính `_GT`). `test/`: 1004 ảnh. Đa số mask toàn đen (không lỗi) — mất cân
  bằng cực đoan, đúng bài đã học ở Bài 3.
- 2 lớp: nền (0) + defect (1) → `num_classes=2`.

## 4. ✅ Đã viết — nằm ở `tu_hoc_deep/` (TÁCH RIÊNG khỏi `src/`, không đụng code sản xuất)

Không sửa tạm vào `chia_data_unet.py`/`train_unet.py` sản xuất (dễ quên đổi lại, rủi ro cho data 4-lớp
công ty) — thay vào đó 2 file riêng:

- `tu_hoc_deep/chia_data_kolektorsdd2.py` — đổi tên mask theo stem ảnh, resize cả ảnh lẫn mask về kích
  thước cố định chia hết 32 (mặc định 224×640 — `INTER_LINEAR` cho ảnh, **`INTER_NEAREST` cho mask**),
  remap 0/255 → 0/1, ghi `images/`, `masks/`, `dataset.yaml` (`num_classes: 2`,
  `names: {0: background, 1: defect}`).
  **Sửa 2026-07-21 (bản đầu SAI):** bản đầu map thẳng `test/` gốc (bộ đánh giá chính thức) thành `val`
  — mà `val` lại dùng để chọn `best.pt` trong lúc train, làm bẩn bộ test (không còn holdout thật). Bản
  sửa: chỉ tách `train/` gốc (2332 ảnh) thành train+val (mặc định 90/10, `--val-ratio`, `--seed 42` để
  tách lại y hệt), còn `test/` gốc (1004 ảnh) giữ nguyên thành 1 split `test` riêng, ghi vào
  `dataset.yaml` nhưng `train_kolektorsdd2.py` không đọc key này — không đụng tới lúc train. Chi tiết
  lý do + hệ quả: `HOC_generalization_overfitting.md` mục 4-5.
- `tu_hoc_deep/train_kolektorsdd2.py` — bản riêng của `train_unet.py`, giống hệt kiến trúc/loss/metric/
  resume đã học ở Bài 1-9, chỉ khác: `weights_list` SINH TỰ ĐỘNG theo `num_classes`
  (`[0.2] + [2.0]*(num_classes-1)`) thay vì hard-code 5 phần tử — không vỡ assert với data 2 lớp, không
  cần sửa tay/nhớ đổi lại gì.
- `tu_hoc_deep/README.md` — cách chạy 2 lệnh.

Chạy:
```bash
cd tu_hoc_deep
python chia_data_kolektorsdd2.py
python train_kolektorsdd2.py --loss ftl_focal --best-metric f1_object
```

## 5. Sau khi bài này chạy được: bước nâng cao hơn
File này chỉ train 1 UNet đơn giản trên mask có sẵn (giống hệt Bài 1-9). Bản THẬT của paper công bố
KolektorSDD2 dùng kiến trúc khác hẳn (2 mạng, mixed supervision) và chính là chủ nhân của mấy file
`split_weakly_*.pyb`. Bài tập đọc-paper-chủ-động + tái hiện lại → `HOC_paper_kolektorsdd2_mixed_supervision.md`.

## 6. Đánh giá trên bộ test holdout thật (2026-07-23)

Đã viết `tu_hoc_deep/test_kolektorsdd2.py` — chạy `best.pt` trên `C:/THUONG/Images/KolektorSDD2/test`
(1004 ảnh gốc, chưa từng đụng lúc train/chọn best.pt, xem mục 4), so với `{id}_GT.png` có sẵn để tính
đúng metric mà `evaluate()` trong `train_kolektorsdd2.py` dùng (IoU pixel lớp lỗi, Recall/Precision/F1
**cấp cục lỗi** — Bài 6), lưu overlay (đỏ=dự đoán, vàng=viền GT) + log riêng các ảnh sai
(`loi_can_kiem_tra.txt`) để soi bằng mắt thay vì lướt cả nghìn ảnh.

**Kết quả thật (best.pt hiện tại, loss mặc định `dice_ce`, `--best-metric f1_object`):**
```
IoU pixel (lớp lỗi) = 0.479
Recall=0.874  Precision=0.912  F1=0.893   (bắt=104  nhầm=10  sót=15)
```
119 cục lỗi thật trong 1004 ảnh test (104 bắt được + 15 sót) → **Recall cấp cục lỗi = 87.4%**, tức
~1/8 lỗi thật bị lọt.

**Bài học đo lường quan trọng (dễ tính sai khi báo cáo cho khách):** sót phải tính theo % **trên tổng
số cục lỗi thật** (15/119 = 12.6%), KHÔNG phải theo % trên tổng số ảnh test (15/1004 = 1.5%) — cách
tính sau pha loãng vấn đề rất nhiều vì đa số ảnh vốn không có lỗi để mà sót. Khách hàng AOI quan tâm "%
lỗi thật bị lọt", không phải "% tổng ảnh có ít nhất 1 lỗi bị lọt".

**Verdict so với paper gốc (đã học ở `HOC_paper_kolektorsdd2_mixed_supervision.md` mục 2):** kiến trúc
paper (2 mạng segmentation+decision, đo AP **cấp ảnh** "có lỗi hay không") KHÔNG phải đòn bẩy nhanh cho
đúng vấn đề Recall **cấp cục lỗi** đang gặp — 2 câu hỏi khác nhau, không so/copy hyperparameter trực
tiếp được. Đòn bẩy thật nằm ở chính U-Net hiện có, xếp theo tốc độ thấy kết quả:

1. **Không cần train lại:** hạ ngưỡng quyết định lớp lỗi. `test_kolektorsdd2.py` hiện dùng argmax = ngưỡng
   0.5 để tính metric; cờ `--conf` chỉ ảnh hưởng ảnh overlay, CHƯA áp vào pred dùng tính Recall/Precision
   — cần vá trước khi quét ngưỡng 0.5→0.4→0.3→0.2 tìm điểm cân bằng khách chấp nhận được.
2. Train lại với `--loss ftl_focal --tv-beta 0.85` (mặc định đang 0.7) — Tversky β cao phạt False
   Negative nặng hơn False Positive.
3. `--best-metric recall_object` thay vì mặc định `f1_object` lúc train, để `best.pt` chọn theo Recall
   thay vì cân bằng P/R.
4. Soi 15 ảnh sót trong `loi_can_kiem_tra.txt` bằng mắt — lỗi tương phản thấp/quá nhỏ hay ngẫu nhiên,
   quyết định có cần thêm augmentation/data hay chỉ cần chỉnh ngưỡng+loss là đủ.

## 8. Vòng 2 (train 2026-07-23, đọc lại số liệu 2026-07-25) — 2 lần thử tăng Recall, CẢ HAI ĐỀU TỆ HƠN

Sau mục 6-7, default trong `train_kolektorsdd2.py` đã đổi (`--loss` mặc định thành `ftl_focal`,
`--tv-beta` mặc định thành `1.5` — cao hơn cả mức đề nghị 0.85 ở mục 6 đòn bẩy 2, bỏ qua nấc trung gian).
Train lại 2 lần (`results_260723_1`, `results_260723_2`), đánh giá lại bằng `test_kolektorsdd2.py` trên
đúng 1004 ảnh test. Số liệu tự tính lại từ `loi_can_kiem_tra.txt` (không có file nào lưu hyperparameter
đã dùng mỗi run — xem còn nợ mục 7):

| Run | Recall | Precision | F1 (tự tính) | sót (FN) | nhầm (FP) |
|---|---|---|---|---|---|
| `results_260721` (Vòng 1, mục 6) | 87.4% | 91.2% | 0.893 | 15/119 | 10 |
| `results_260723_1` | 77.3% | 92.9% | 0.844 | 27/119 | 7 |
| `results_260723_2` | 79.8% | 81.9% | 0.808 | 24/119 | 21 |

**Kết luận thật (không phải kỳ vọng):** cả 2 lần "thử đòn bẩy tăng Recall" đều làm Recall THẤP HƠN Vòng
1, không cao hơn. Trước khi thử đòn bẩy khác, phải trả lời được TẠI SAO đòn bẩy tưởng đúng hướng lại
phản tác dụng. 3 nghi vấn — nghi vấn 1 đã XÁC NHẬN ĐÚNG (2026-07-25), 2 và 3 còn mở:

1. **✅ XÁC NHẬN: `--conf` chưa từng chạm vào metric.** Đọc lại `test_kolektorsdd2.py`: `--conf` chỉ đổi
   `cls_mask` dùng để VẼ overlay trong `render()` (dòng ~119), còn `object_stats()` dùng để tính
   Recall/Precision luôn nhận `pred` từ `prob_full.argmax(0)` trong `predict_one()` (dòng ~110) — tức
   luôn là ngưỡng 0.5 cố định, bất kể `--conf` truyền vào bao nhiêu. Nghĩa là ở Vòng 2, đòn bẩy "hạ
   ngưỡng" (mục 6 đòn bẩy 1) CHƯA từng thật sự được đo — Recall/Precision đo được chỉ phản ánh 2 nghi vấn
   còn lại. **Còn nợ:** vá `object_stats()` nhận `pred` đã áp `--conf` (người học tự vá, 2026-07-25).
2. **Val set nhỏ (~233 ảnh, 10% của 2332) → `best.pt` chọn theo epoch may rủi.** Soi `train_log.txt`:
   `results_260723_2` epoch 1→2, `obj_recall` 0.80→0.90 nhưng `obj_precision` 0.48→0.07 CÙNG LÚC — dao
   động quá lớn giữa 2 epoch liền kề để tin là tín hiệu thật thay vì nhiễu thống kê (val set ít lỗi, vài
   con lỗi đổi tay đủ để precision nhảy hàng chục %). `best_metric[f1_object]` chọn epoch tốt nhất DỰA
   TRÊN nhiễu này — không chắc epoch được chọn là epoch tổng quát hoá tốt nhất thật.
3. **`tv-beta 1.5` có thể đã đi quá xa.** Mục 6 đòn bẩy 2 đề nghị thử 0.85 (so với 0.7 lúc đó) — nhưng
   default code đã nhảy thẳng lên 1.5, bỏ qua nấc thang trung gian, nên không tách được "beta cao có ăn
   không" khỏi "1.5 cụ thể có phải quá cao".

**Việc tiếp theo (theo thứ tự, đừng đổi 2 thứ cùng lúc):**
1. Vá `--conf` → `object_stats()` (nghi vấn 1, đang làm).
2. Thêm ghi log hyperparameter mỗi run (còn nợ mục 7).
3. Chạy lại ĐÚNG 1 biến mỗi lần: (a) `tv-beta 0.85` giữ nguyên loss `ftl_focal`, (b) quét `--conf`
   0.5→0.4→0.3→0.2 trên CÙNG 1 `best.pt` (không train lại) để tách ảnh hưởng threshold khỏi ảnh hưởng
   loss.
4. Cân nhắc tăng val set hoặc trung bình vài epoch cuối thay vì chọn 1 epoch "best" đơn lẻ, để giảm nhiễu
   khi chọn `best.pt`.

## 7. Còn nợ
- [ ] **(ưu tiên, chặn mọi so sánh công bằng)** Vá `test_kolektorsdd2.py` để `--conf` áp thẳng vào
      `object_stats()` (không chỉ `render()`) — xác nhận lại 2026-07-25 VẪN CHƯA sửa. Chi tiết: mục 8.
- [ ] Lưu hyperparameter (`--loss --tv-beta --tv-gamma --focal-gamma --best-metric`...) ra 1 file cạnh
      `model_cfg.yaml` mỗi lần train (VD `run_args.yaml`, ghi bằng `yaml.safe_dump(vars(args), ...)` ngay
      trong `train_kolektorsdd2.py`) — hiện `results_260723_1`/`_2` không có, phải suy ngược từ mặc định
      trong code, không tái lập được thí nghiệm đã chạy.
- [x] Thử lần lượt 3 đòn bẩy ở mục 6 — ĐÃ THỬ (Vòng 2, mục 8) nhưng KẾT QUẢ TỆ HƠN Vòng 1, chưa rõ tại
      sao hết (nghi vấn 1 đã xác nhận, 2-3 còn mở) — chưa tính là "xong", cần lặp lại sau khi vá 2 mục
      trên.
- [ ] Soi phân bố lớp train/val TRƯỚC khi đổ lỗi model thêm (kinh nghiệm Bài 9) — KolektorSDD2 mất cân
      bằng nặng, đúng bẫy đã gặp ở Bài 9: đa số ảnh không lỗi.
