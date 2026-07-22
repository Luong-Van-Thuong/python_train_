# tu_hoc_deep — sandbox tự học, tách khỏi code sản xuất

Thư mục này KHÔNG liên quan tới `src/` (code chạy dự án thật của công ty). Đây là chỗ tự luyện lại
toàn bộ quy trình (chuẩn bị data → train → đánh giá) trên dữ liệu công khai, để không phụ thuộc
`chia_data_unet.py`/`train_unet.py` có sẵn cầm tay — và để không lỡ tay sửa code sản xuất khi thử nghiệm.

Ghi chú học đi kèm: `../docs/HOC_kolektorsdd2_data_prep.md` (giải thích VÌ SAO từng bước làm vậy).
Trạng thái tổng của việc tự học: `../docs/TIEN_DO.md`.

## Chạy (2 bước)

```bash
cd tu_hoc_deep
python chia_data_kolektorsdd2.py          # KolektorSDD2 -> data_kolektorsdd2/ (images/masks/dataset.yaml)
python train_kolektorsdd2.py               # train UNet trên data vừa tạo
```

Mặc định đọc `C:/THUONG/Images/KolektorSDD2`. Đổi nguồn bằng `--src`, đổi kích thước resize bằng
`--size WxH` (phải chia hết 32).

`chia_data_kolektorsdd2.py` tự tách `train/` gốc thành train+val (mặc định 90/10, đổi bằng
`--val-ratio`, seed cố định `--seed` để tách lại y hệt). `test/` gốc giữ nguyên thành 1 split `test`
riêng — đây là bộ holdout thật, `train_kolektorsdd2.py` không đụng tới lúc train. Xem vì sao:
`../docs/HOC_generalization_overfitting.md` mục 4-5.

## Vì sao có `train_kolektorsdd2.py` riêng thay vì dùng `train_unet.py`

`train_unet.py` hard-code `weights_list` 5 phần tử cho đúng bài 4-lỗi của công ty — chạy thẳng với
KolektorSDD2 (2 lớp) sẽ vỡ `assert`. Thay vì sửa tạm rồi dễ quên đổi lại (rủi ro cho code sản xuất),
bản ở đây sinh `weights_list` tự động theo `num_classes`. Phần còn lại (kiến trúc, loss, metric,
resume) giống hệt `train_unet.py` — mọi thứ đã học ở `HOC_unet_loi_nho_6px.md` Bài 1-9 dùng lại nguyên vẹn.
