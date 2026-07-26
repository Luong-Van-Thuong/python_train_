# 🎓 Ghi chú: Tái hiện paper gốc của KolektorSDD2 (mixed supervision)

> Mục tiêu HỌC NÂNG CAO — khác hẳn `HOC_kolektorsdd2_data_prep.md` (chỗ đó chỉ train 1 UNet đơn giản
> trên mask có sẵn). File này là bài tập ĐỌC PAPER CHỦ ĐỘNG rồi code lại, để trả lời câu hỏi gốc của bạn:
> "đọc paper có giúp gì không?" — câu trả lời chỉ có được khi tự tái hiện ra số, không phải đọc suông.
> Trạng thái tổng: `TIEN_DO.md`. ⚠️ Đây là việc NẶNG, làm sau khi `tu_hoc_deep/train_kolektorsdd2.py`
> đã chạy được (mục tiêu thấp hơn, làm trước để có baseline so sánh).

---

## 0. Paper là gì, tìm ở đâu

Bộ KolektorSDD2 được công bố cùng paper:

> **"Mixed supervision for surface-defect detection: from weak to fully supervised learning"**
> Jakob Božič, Domen Tabernik, Danijel Skočaj — *Computers in Industry*, 2021.
> (nhóm ViCoS, University of Ljubljana — cùng nhóm làm ra KolektorSDD gốc 2019).

Tác giả có public code kèm paper — tìm trên GitHub với tên tổ chức **vicoslab**, repo tên có chữ
**mixed-segdec-net**. Tự tìm bản PDF (Google Scholar / arXiv / trang nhà xuất bản) — tôi không tự tạo
link, tự bạn tra để chắc đúng bản mới nhất và không dính link rác.

**Vì sao đáng tái hiện:** đây CHÍNH LÀ paper tạo ra bộ dữ liệu bạn đang dùng, và giải thích thẳng ý nghĩa
mấy file `split_weakly_*.pyb` (0/16/53/126/246) — nó là 1 thí nghiệm ablation NẰM TRONG paper này.

---

## 1. Trước khi code — ĐỌC và trả lời được các câu hỏi sau (tự chấm điểm hiểu)

Đọc xong phải tự trả lời được (không nhìn đáp án gợi ý bên dưới trước):

1. Paper dùng **mấy mạng** (network), mỗi mạng nhận vào gì, ra gì?
2. "Mixed supervision" nghĩa là gì — trong tập train, có ảnh nào KHÔNG có mask pixel-level mà chỉ có
   nhãn ảnh (có lỗi / không lỗi) không? Vì sao làm vậy vẫn train được?
3. 2 mạng train CÙNG LÚC (end-to-end) hay TÁCH GIAI ĐOẠN (train mạng 1 xong mới train mạng 2)?
4. Paper đo hiệu năng bằng chỉ số gì ở cấp độ ẢNH (image-level) — có phải AP (Average Precision) không?
   Khác gì với Recall/F1 object-level (cấp độ CỤC LỖI) mà bạn đã học ở `HOC_unet_loi_nho_6px.md` Bài 6?
5. Thí nghiệm dùng `split_weakly_N.pyb` (N=0,16,53,126,246) để trả lời câu hỏi gì? Kết luận chính của
   thí nghiệm đó là gì (thêm bao nhiêu ảnh weak-label thì lợi nhiều nhất, lợi ích có bão hoà không)?

> Gợi ý về kiến trúc (theo trí nhớ của tôi, CẦN ĐỐI CHIẾU LẠI VỚI PAPER THẬT vì tôi không chắc 100% số
> liệu/tham số chính xác — chỉ chắc về Ý TƯỞNG tổng thể):
> - **Mạng 1 — Segmentation network:** mạng tích chập (không phải UNet có skip như bạn học — kiến trúc
>   riêng của nhóm này, dạng chuỗi conv+pool) → ra 1 bản đồ xác suất lỗi (giống mask bạn đã quen).
> - **Mạng 2 — Decision/Classification network:** LẤY feature từ mạng 1 (cả feature map giữa chừng lẫn
>   bản đồ lỗi ở trên) → global max-pool + global avg-pool → ghép lại → ra **1 con số duy nhất**: ảnh này
>   có lỗi hay không (khác Bài 1: đây là quay lại mức Classification, dùng đè lên kết quả Segmentation).
> - Nhờ có mạng 2, ảnh nào chỉ có nhãn "có/không lỗi" (không có mask) vẫn train được (loss của mạng 2 là
>   nhãn ảnh, không cần mask) → đây chính là cơ chế "mixed supervision".
> Tất cả chi tiết số (learning rate, số epoch, trọng số loss...) → PHẢI đọc bản paper thật, tôi không
> bịa số.

---

## 2. So sánh với những gì bạn đã làm (`tu_hoc_deep/train_kolektorsdd2.py`)

| | Bạn đã làm (UNet đơn giản) | Paper gốc |
|---|---|---|
| Số mạng | 1 (UNet, segmentation only) | 2 (segmentation + decision) |
| Nhãn cần | mask pixel-level cho MỌI ảnh train | mask cho 1 phần, ảnh còn lại chỉ cần nhãn có/không lỗi |
| Đích ra | mask (pixel nào là lỗi) | mask + 1 quyết định ảnh có lỗi hay không |
| Chỉ số chấm | Recall/F1 **object-level** (Bài 6) | AP (Average Precision) **image-level** |
| Kiến trúc encoder | resnet34 (Unet chuẩn, có sẵn trong `segmentation_models_pytorch`) | mạng tự thiết kế riêng của paper |

**Vì sao khác:** bài toán của bạn ở Bài 1-9 quan tâm "**pixel/cục lỗi nào**" (để đo mm, quyết định
đạt/không đạt theo kích thước) → object-level Recall/F1 đúng. Paper KolektorSDD2 quan tâm "**ảnh này có
lỗi hay không**" (bài toán phân loại sản phẩm PASS/FAIL ở dây chuyền) → AP image-level đúng hơn. Cùng 1
bộ dữ liệu, 2 câu hỏi khác nhau ra 2 kiến trúc + 2 chỉ số khác nhau — **đây chính là bài học lớn nhất**
của việc tái hiện paper này: không có "kiến trúc đúng tuyệt đối", chỉ có kiến trúc đúng CHO CÂU HỎI đang hỏi.

---

## 3. Kế hoạch tái hiện (làm dần, không cần 1 lần)

- [ ] **Bước 1 (bắt buộc trước):** đọc paper, tự trả lời 5 câu hỏi mục 1. Ghi câu trả lời của bạn vào
      cuối file này (phần mục 5), TỰ VIẾT bằng lời mình — đừng copy nguyên văn paper.
- [ ] **Bước 2:** đọc code gốc trên GitHub (repo `mixed-segdec-net...` của vicoslab) — đối chiếu kiến
      trúc mạng 1/mạng 2 thật với phần "Gợi ý" ở mục 1, sửa lại chỗ tôi đoán sai.
- [ ] **Bước 3 (stretch, nặng nhất):** tự code lại kiến trúc 2 mạng bằng PyTorch thuần (không dùng
      `segmentation_models_pytorch` vì đó là UNet chuẩn, không phải kiến trúc riêng của paper) — đặt ở
      `tu_hoc_deep/paper_repro/` (thư mục MỚI, tách khỏi `train_kolektorsdd2.py` vì đây là kiến trúc
      khác hẳn, không phải bản vá của UNet).
- [ ] **Bước 4:** dùng `split_weakly_*.pyb` (đọc bằng `pickle.load`) để tái hiện đúng thí nghiệm ablation
      của paper — train nhiều lần với số ảnh weak-label khác nhau, vẽ biểu đồ AP theo N giống paper.
- [ ] **Bước 5:** so AP đo được của mình với con số paper báo cáo — lệch nhiều/ít, đoán vì sao (paper
      thường train lâu hơn, tune kỹ hơn — không kỳ vọng khớp 100%, mục tiêu là HIỂU chứ không phải SOTA).

## 4. Câu trả lời của bạn (điền sau khi đọc — Bước 1)

*(để trống, điền tay sau khi đọc paper thật — đây là phần TỰ HỌC, không phải tôi điền hộ)*

1.
2.
3.
4.
5.
