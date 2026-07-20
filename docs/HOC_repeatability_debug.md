# 🎓 Ghi chú: Vấn đề THẬT ở công ty — mất tính lặp lại (repeatability) khi dung sai ±0.03mm

> Buổi học cùng Claude (2026-07-19). Đây KHÔNG phải bài tự học benchmark công khai — đây là bài toán
> sản xuất thật, vendor Samsung, dung sai đo/inspection ±0.03mm. Trạng thái tổng: `TIEN_DO.md`.
> Phương pháp làm việc ở file này: **Socratic** — Claude đặt câu hỏi, mình tự trả lời bằng quan sát thật
> từ pipeline của mình trước, KHÔNG xem gợi ý/đáp án trước. Mục đích không phải để nhớ kiến thức, mà để
> tự lộ ra chỗ mình *tưởng* hiểu nhưng chưa hiểu thật, và luyện kỹ năng chẻ vấn đề ra từng lớp nguyên nhân.

---

## 1. Bối cảnh bài toán

Công ty là vendor của Samsung. Yêu cầu: đo đạc (measurement) và inspection (phát hiện lỗi) với dung sai
**±0.03mm**. Đang tập trung nhánh Deep Learning để:
- Nhận diện lỗi nhỏ **trước khi nó ăn vào nền** (liên quan Bài 3-4 `HOC_unet_loi_nho_6px.md`: class
  imbalance + downsample làm lỗi nhỏ "bốc hơi"/lem biên).
- Có **tính lặp lại (repeatability)** ổn định — đây là phần CHƯA đạt.

## 2. Hiện tượng quan sát (triệu chứng thật, đã thấy)

- Cùng **1 tấm ảnh**, chạy lại nhiều lần → có lúc báo **NG**, có lúc báo **OK**.
- Bài toán đo: cùng 1 ảnh, chạy lại → có lúc ra **0.05mm**, có lúc ra **0.08mm**. Trong khi dung sai chấp
  nhận được là **0.03mm** → sai lệch giữa các lần chạy còn LỚN HƠN cả dung sai → hệ thống chưa dùng được
  ở mức chính xác này, bất kể model "đúng trung bình" hay không.

## 3. Round 1 — câu hỏi chẩn đoán (ĐANG TRẢ LỜI, chưa có đáp án)

Mục tiêu: khoanh vùng nguyên nhân nằm ở tầng nào — chế độ model, GPU/thuật toán không xác định
(non-determinism), ngưỡng quyết định nhạy biên, hay tầng đo (measurement) kế thừa bất ổn từ tầng DL.

1. "Chạy nhiều lần cùng 1 ảnh" — chạy lại toàn bộ script (load lại model + ảnh từ đĩa mỗi lần), hay giữ
   nguyên process và gọi `model(img)` nhiều lần trên cùng 1 tensor có sẵn?
2. Trước khi inference đã gọi `model.eval()` chưa? Nếu quên, BatchNorm ở `model.train()` với batch nhỏ/1
   ảnh sẽ hành xử khác `eval()` thế nào — đủ để đổi kết quả giữa các lần chạy không?
3. Đã set `torch.backends.cudnn.deterministic` / `torch.use_deterministic_algorithms` chưa? Vì sao cuDNN
   mặc định không xác định (tự chọn thuật toán nhanh nhất theo phần cứng lúc đó) lại có thể gây sai khác
   kết quả giữa các lần chạy?
4. Bước ra quyết định NG/OK và đo mm lấy từ mask predict bằng cách nào (argmax, hay threshold theo 1
   ngưỡng xác suất cụ thể)? Nếu tại biên lỗi xác suất dao động quanh 0.49–0.51, chuyện gì xảy ra với
   quyết định nhị phân? Đây có phải CÙNG 1 nguyên nhân gốc với hiện tượng "ăn vào nền" (Bài 4), hay là 2
   vấn đề tách biệt?
5. Phép đo mm bằng OpenCV lấy input là mask nhị phân xuất từ model, hay đo trực tiếp trên ảnh gốc độc lập
   với DL? Nếu input là mask chưa ổn định (theo câu 2-4), phép đo phía sau có thể ổn định hơn cái nó nhận
   đầu vào không?

> Sau khi trả lời 5 câu trên bằng quan sát thật (không đoán lý thuyết) → bước tiếp theo là thiết kế
> **thí nghiệm cụ thể để tự chứng minh bằng số** (VD: cố định seed + eval() + deterministic, chạy lại,
> xem có bit-exact không), không nghe Claude nói đúng/sai.

## 4. Round 2 — tạm hoãn: cơ chế tối ưu trong `tu_hoc_deep/train_kolektorsdd2.py`

4 câu hỏi về phần "cách tối ưu mô hình" mà `HOC_unet_loi_nho_6px.md` Bài 1-9 CHƯA dạy riêng (kiến trúc/
loss/metric thì đã có bài rồi):

1. **Optimizer `AdamW`** (dòng 254): `weight_decay=1e-4` khác gì cộng L2 penalty thẳng vào loss? Vì sao
   AdamW tách decoupled weight decay ra riêng so với Adam+L2 gốc?
2. **LR schedule `CosineAnnealingLR`** (dòng 255, dùng ở dòng 300): với `lr=1e-3`, `T_max=60`, LR ở epoch
   1/30/60 là bao nhiêu (tự tính bằng công thức, không tra)? Vì sao giảm LR dần giúp hội tụ ổn hơn LR
   cố định?
3. **Mixed precision `autocast` + `GradScaler`** (dòng 292-297): rủi ro gradient underflow ở fp16 là gì?
   `GradScaler` giải quyết bằng cách nào (scale loss trước backward, unscale trước step)?
4. **Checkpoint/resume** (dòng 268-282, 328-337): vì sao lưu cả `optimizer_state_dict` +
   `scheduler_state_dict`, không chỉ `model_state_dict`? Thử: xoá load `optimizer_state_dict` khi resume,
   train tiếp — loss có nhảy giật không so với resume đầy đủ?

## 5. Gap thật đã tự nhận ra (2026-07-19) — meta, không phải kỹ thuật

Không phải thiếu kiến thức lẻ — là **chưa chắc mình có biết chẻ vấn đề ra đúng lớp nguyên nhân không**,
và **chưa biết cách tự kiểm tra xem mình hiểu thật hay chỉ tưởng hiểu**. Cách xử lý: giữ nguyên phương
pháp Socratic ở file này (và các buổi sau) — trả lời câu hỏi bằng quan sát/thí nghiệm thật trước, không
đọc đáp án trước khi tự thử.
