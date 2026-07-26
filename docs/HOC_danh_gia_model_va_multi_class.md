# 🎓 Ghi chú: Cách tự chấm "model tốt hơn chưa" + hướng tiếp theo (nhiều lớp lỗi nhỏ)

> Mở ra 2026-07-25 sau khi soi lại 3 lần train KolektorSDD2 (`results_260721`, `results_260723_1`,
> `results_260723_2`) và phát hiện 2 lần train sau — dù có ý định tăng Recall — lại cho kết quả TỆ HƠN
> lần đầu (xem `HOC_kolektorsdd2_data_prep.md` mục 8). File này tách phần **bài học chung** (áp dụng cho
> mọi bài toán inspection sau này, không riêng KolektorSDD2) ra khỏi phần "log số liệu" ở file kia.
> Trạng thái tổng: `TIEN_DO.md`.

---

## Phần A — Muốn biết "model A có tốt hơn model B không", cần đủ 4 điều kiện

Case study 2026-07-25 là ví dụ thật của việc thiếu 3/4 điều kiện dưới đây — không phải lỗi hiếm, gần
như ai tự học cũng vấp phải ít nhất 1 lần:

1. **Cùng 1 bộ test, chưa từng bị đụng tới lúc train/chọn best.pt/chỉnh threshold.**
   KolektorSDD2 đã làm đúng phần này (mục 4 `HOC_kolektorsdd2_data_prep.md` — sửa lỗi `test/` bị map
   thành `val`). Đây là điều kiện DỄ nhớ nhất vì đã học đau (`HOC_generalization_overfitting.md`).
2. **Cùng 1 định nghĩa metric, và định nghĩa đó phải THẬT SỰ được tính đúng như mình nghĩ.**
   Đây là chỗ vấp hôm nay: tưởng đang đo "Recall sau khi hạ ngưỡng `--conf`", nhưng `object_stats()`
   luôn tính trên `argmax` cố định — 2 lần train sau đo ra con số thấp hơn, nhưng con số đó KHÔNG phản
   ánh đòn bẩy định thử. Cách tự kiểm: mỗi khi đổi 1 cờ, tự hỏi "cờ này có thật sự chạy tới đoạn code
   tính ra con số mình đang nhìn không?" — lần theo biến từ chỗ in ra ngược lên, không tin vào tên cờ.
3. **Đủ mẫu để chênh lệch không phải nhiễu ngẫu nhiên.** 119 cục lỗi là con số NHỎ. Nếu coi mỗi lỗi là 1
   phép thử Bernoulli (bắt được / không), độ lệch chuẩn của Recall đo được ở n=119 xấp xỉ:
   `sqrt(p(1-p)/n)` — với p≈0.87 → `sqrt(0.87*0.13/119) ≈ 0.031` (~3.1 điểm %). Nghĩa là 2 lần đo Recall
   chênh nhau **dưới ~3 điểm %** (VD 87.4% vs 84%) CÓ THỂ chỉ là nhiễu lấy mẫu, chưa chắc model đổi thật.
   Vòng 2 hôm nay chênh 87.4% → 77-80%, tức ~7-10 điểm %, VƯỢT xa mức nhiễu này → tin được là tín hiệu
   thật (model thật sự tệ hơn), không phải may rủi — nhưng nếu lần sau chênh chỉ 2-3 điểm %, đừng vội kết
   luận "đòn bẩy X ăn", cần chạy lại thêm 1-2 seed để chắc.
4. **Biết chính xác đã đổi CÁI GÌ giữa 2 lần chạy (ghi log).** Không có bước này thì kể cả (1)(2)(3) đúng
   hết, vẫn không thể kết luận "đòn bẩy nào ăn" — chỉ biết "có gì đó đổi làm khác đi". Đây là gap cụ thể
   phát hiện hôm nay: `model_cfg.yaml` không lưu `--loss/--tv-beta/...`, phải suy ngược từ default trong
   code. **Sửa 1 lần, dùng mãi:** mỗi script train nên tự ghi `vars(args)` ra 1 file yaml/json cạnh
   checkpoint, TRƯỚC khi vòng epoch đầu tiên chạy (để cả run bị crash giữa chừng vẫn còn log).

**Tóm tắt 1 câu:** "model tốt hơn chưa" không phải câu hỏi có thể trả lời từ 1 con số — nó cần 1 quy
trình so sánh đủ 4 điều kiện trên; thiếu 1 điều kiện, con số vẫn ra nhưng không mang nghĩa đang tưởng.

---

## Phần B — So với paper gốc: khi nào so được, khi nào không

Đã học kỹ ở `HOC_paper_kolektorsdd2_mixed_supervision.md` mục 2 — nhắc lại phần cốt lõi vì hay bị quên:
paper đo **AP ở cấp ẢNH** (ảnh này có lỗi hay không — bài toán PASS/FAIL), còn UNet tự train ở đây đo
**Recall/Precision ở cấp CỤC LỖI** (pixel/blob nào là lỗi — bài toán định vị+đo). Hai câu hỏi khác nhau,
hai chỉ số khác nhau, **không có một con số chung để nói "ăn đứt paper" hay "thua paper"**.

Muốn so trực tiếp được, phải chọn 1 trong 2 hướng:
- Tự tính thêm AP cấp ảnh từ model UNet hiện có (coi "ảnh có ít nhất 1 pixel lỗi dự đoán" = ảnh NG) rồi so
  với số AP paper báo cáo — so được nhưng không công bằng 100% vì kiến trúc paper có thêm mạng quyết định
  chuyên biệt, không chỉ suy ra từ mask segmentation.
- Hoặc tái hiện đúng kiến trúc 2 mạng của paper (việc NẶNG, đã có kế hoạch riêng ở
  `HOC_paper_kolektorsdd2_mixed_supervision.md` mục 3, làm SAU khi bài toán Recall cấp cục lỗi ổn định).

Kết luận hiện tại: **câu hỏi đúng để tự hỏi không phải "so với paper", mà là "Recall/Precision cấp cục
lỗi này có đủ tốt cho yêu cầu thật (dung sai/tỉ lệ sót chấp nhận được) chưa"** — đây cũng chính là cách
sẽ phải làm việc với khách hàng AOI sau này (họ hỏi "% lỗi thật bị lọt là bao nhiêu", không hỏi "AP là
bao nhiêu").

---

## Phần C — Hướng tiếp theo: bài toán NHIỀU lớp lỗi nhỏ (mục tiêu bạn đã nêu)

KolektorSDD2 là bài toán **1 lớp lỗi + nền** — vẫn còn đơn giản hơn thứ sẽ gặp nhiều ở công ty (nhiều
dạng lỗi nhỏ khác nhau trên cùng 1 sản phẩm, VD trầy/bavia/thủng/đốm — mỗi loại 1 class). Bài 1-9
(`HOC_unet_loi_nho_6px.md`) đã cho nền tảng đúng hướng (kiến trúc, loss, class imbalance CƠ BẢN), nhưng
có vài thứ CHƯA từng phải đối mặt vì bài toán đến giờ luôn là nhị phân (lỗi/nền):

### 1. Nhầm GIỮA các lớp lỗi — thất bại mới, nhị phân không có
Với 1 lớp lỗi, chỉ có 3 khả năng: bắt đúng / bỏ sót / báo nhầm ở nền. Với N lớp, thêm khả năng thứ 4:
**đúng vị trí, SAI loại lỗi** (model thấy có lỗi, khoanh đúng chỗ, nhưng gọi tên sai — VD lẫn "trầy" với
"bavia"). Object-level Recall/Precision hiện tại (đếm theo từng lớp riêng, xem `object_stats()` — vòng
`for c in range(1, num_classes)`) đã VÔ TÌNH đúng hướng để mở rộng (code đã per-class sẵn), nhưng cần
thêm 1 bảng **ma trận nhầm lẫn giữa các lớp lỗi** (không chỉ lỗi-vs-nền) để thấy cặp lớp nào hay bị lẫn.

### 2. Mất cân bằng CHỒNG mất cân bằng
Bài 9 đã học: 1 lớp có 0 tile train → model không học nổi lớp đó (không phải model dốt, data thủng).
Với N lớp, xác suất có ít nhất 1 lớp hiếm/thủng data tăng lên nhiều — **luôn đếm phân bố lớp train/val
TRƯỚC** (thói quen đã có từ Bài 9) là bắt buộc hơn nữa, không phải tùy chọn.

### 3. Có thể cần threshold/trọng số RIÊNG cho từng lớp
Lớp lỗi nguy hiểm (ảnh hưởng an toàn/chức năng) cần Recall cao (chấp nhận báo nhầm nhiều hơn); lớp lỗi
thẩm mỹ nhẹ có thể chấp nhận Precision cao hơn (đỡ báo nhầm làm phiền dây chuyền). 1 ngưỡng `--conf`
chung cho mọi lớp (như `test_kolektorsdd2.py` hiện tại) sẽ không tối ưu khi có nhiều lớp khác mức độ
quan trọng — cần quét threshold **theo từng lớp** (PR curve riêng mỗi class), không phải 1 đường chung.

### 4. Kỷ luật ghi log thí nghiệm — học được đau hôm nay, PHẢI mang sang
Với 1 lớp, số tổ hợp cần thử (loss/threshold/best-metric) đã đủ dễ nhầm lẫn thứ gì gây ra kết quả gì
(chính là bài học Phần A #4). Với N lớp, số tổ hợp nhân lên (mỗi lớp có thể cần threshold/weight riêng)
— nếu không ghi log hyperparameter mỗi run NGAY TỪ ĐẦU, việc so sánh sẽ rối nhanh hơn nhiều so với hôm
nay. Làm việc này TRƯỚC khi mở rộng sang nhiều lớp, không phải sau.

### 5. Lớp quyết định cấp ảnh (nếu sau này cần PASS/FAIL 1 câu trả lời)
Nếu sản phẩm cuối cần 1 quyết định duy nhất "ảnh/sản phẩm này PASS hay FAIL" (gộp từ N lớp lỗi), đây
chính là vai trò "mạng quyết định" trong paper KolektorSDD2 gốc (Phần B) — không cần tái hiện nguyên
kiến trúc, nhưng nguyên lý (gộp nhiều tín hiệu cấp lớp thành 1 quyết định cấp ảnh) sẽ cần dùng tới.

### Đề xuất thứ tự học (không cần làm hết 1 lúc)
1. Vá xong 2 việc còn nợ trên KolektorSDD2 (threshold áp đúng vào metric + ghi log hyperparameter) —
   đang làm, xem `HOC_kolektorsdd2_data_prep.md` mục 7-8. Đây là 2 kỹ năng nền cho MỌI bài multi-class
   sau này, rẻ nhất để học ngay trên bài toán 1 lớp trước khi bài toán phức tạp hơn.
2. Thử nghiệm multi-class NGAY TRÊN data công ty hiện có (4 lớp: `ban, burrs, defo, phong` từ Bài 9) —
   đã có sẵn, không cần benchmark mới: viết 1 bảng "ma trận nhầm lẫn giữa các lớp lỗi" (mục C.1) từ
   `train_unet.py`/`predict_unet.py` hiện có, xem có cặp lớp nào hay bị lẫn không.
3. Học PR curve theo từng lớp (mục C.3) — vá `test_kolektorsdd2.py`/`predict_unet.py` để quét nhiều
   ngưỡng cùng lúc, vẽ 1 đường Recall-Precision mỗi lớp thay vì 1 điểm duy nhất.
4. (Sau, khi có nhu cầu PASS/FAIL thật) Đọc lại paper KolektorSDD2 với góc nhìn "mạng quyết định gộp
   nhiều tín hiệu" — lúc đó đọc sẽ CÓ Ý NGHĨA hơn vì đã có bài toán multi-class thật để đối chiếu, thay
   vì đọc trừu tượng.

---

## Checklist rút gọn (dán vào đầu buổi học sau khi quay lại file này)
- [ ] Vá `--conf` áp vào `object_stats()` trong `test_kolektorsdd2.py` (đang làm, người học tự vá).
- [ ] Thêm ghi log hyperparameter mỗi lần train (`run_args.yaml` cạnh `model_cfg.yaml`).
- [ ] Chạy lại Vòng 3 KolektorSDD2 đúng 1 biến/lần, có log đầy đủ.
- [ ] Thử vẽ ma trận nhầm lẫn GIỮA các lớp lỗi trên data công ty 4 lớp (Bài 9) — bước đầu tiên vào
      multi-class, dùng data đã có sẵn, không cần chờ benchmark mới.
