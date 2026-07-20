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

## 6. Còn nợ
- [ ] Chạy thử thật, so log với kinh nghiệm Bài 9 (soi phân bố lớp train/val TRƯỚC khi đổ lỗi model —
      KolektorSDD2 mất cân bằng nặng, đúng bẫy đã gặp ở Bài 9: đa số ảnh không lỗi).
- [ ] So recall/F1 object-level đạt được với kỳ vọng thực tế — mục đích là thấy CÙNG 1 kỹ thuật áp dụng
      lên data khác thì số liệu khác nhau ra sao, không phải để đạt SOTA.
