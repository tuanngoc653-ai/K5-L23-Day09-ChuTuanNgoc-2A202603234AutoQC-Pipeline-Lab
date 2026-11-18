# Báo cáo Day 09 — Chu Tuan Ngoc, 2A202603234

Cấu hình: `phash_dist = 12`, `second_opinion.conf = 0.6` (các mục khác giữ nguyên). Số liệu lấy từ `submission/reports/` (`08_summary.json`, `config_sha1 = 58d1755c…`).

## 1. Kết luận trước
**HOLD** (`python -m pipeline.run` → mã thoát **10**).
- Vào **224** ảnh → bước 1 loại **22** → **202** ảnh qua QC → xuất **202** ảnh ở `submission/export/clean_v1/` dưới dạng **bản nháp** (`manifest.json` ghi `decision = Hold`, chưa được phát hành).
- **0 lỗi bắt buộc** còn lại sau sửa tự động và người duyệt, nhưng còn **38/202 ảnh (18,8 %)** có cảnh báo mà chưa ai duyệt, vượt ngưỡng `warn_ratio_hold = 10 %`.
- Lý do chính: hàng đợi người duyệt có **125 ảnh**, ngân sách chỉ **80**. Cảnh báo còn lại chủ yếu là box lỏng (20) và box có thể thừa (21), không phải lỗi cú pháp.

## 2. Lỗi và phân việc
Số liệu từ `01_intake` và `02_error_list_v1` (trước khi sửa). Mẫu số: bước 1 tính trên 224 ảnh, QC tính trên 202 ảnh.

| Mã lỗi | Số dòng / ảnh | Ai sửa | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT | 8 ảnh / 224 | máy (loại ở bước 1) | md5 trùng từng byte, không có rủi ro loại nhầm |
| DUP_IMAGE_NEAR | 4 ảnh / 224 | máy (loại ở bước 1) | phash ≤ 12; đã mở xem cả 4 cặp, đều là trùng thật (§3) |
| LEAK_TEST | 5 ảnh / 224 | máy (loại ở bước 1) | trùng ảnh trong test; giữ lại sẽ làm số đo trên test bị thổi phồng |
| CORRUPT_IMAGE · BAD_SIZE | 2 · 3 ảnh / 224 | máy (loại ở bước 1) | ảnh không mở được, hoặc cạnh ngắn < 250 px |
| BAD_CLASS_ID | 6 dòng / 6 ảnh | máy: `drop_line` | không biết lớp đúng nên xoá dòng; ảnh vẫn vào hàng đợi (FIXED_DROP_LINE) |
| BAD_FORMAT | 3 dòng / 3 ảnh | máy: `drop_line` | dòng không đọc được |
| BOX_ZERO | 4 dòng / 4 ảnh | máy: `drop_line` | box không có diện tích |
| DUP_BOX | 7 dòng / 7 ảnh | máy: `drop_line` | cùng lớp, IoU > 0,9 với một dòng trước đó |
| BOX_OUT_OF_IMAGE | 8 dòng / 8 ảnh | máy: `clip_box` | cắt box về [0,1], giữ lớp |
| BAD_CLASS_NAME | 2 lỗi / data.yaml | máy: `rename_classes` | ghi lại `names` theo `cfg.classes` |
| MISSING_LABEL_FILE | 6 ảnh / 202 | **người** (`needs_label`) | máy không bịa nhãn và không tạo file rỗng (file rỗng nghĩa là "ảnh nền") |
| POSSIBLE_MISSING | 53 / 52 ảnh | **người** | model lớn thấy vật mà nhãn không có |
| POSSIBLE_WRONG_CLASS | 18 / 18 ảnh | **người** | cùng vật (IoU ≥ 0,5) nhưng khác lớp |
| POSSIBLE_LOOSE_BOX | 28 / 24 ảnh | **người** | cùng lớp, IoU < 0,5 |
| POSSIBLE_FALSE_POS | 29 / 27 ảnh | **người** | box nhãn không khớp dự đoán nào |
| EMPTY_LABEL · BOX_TINY · BOX_HUGE · CLASS_RARE | 12 · 6 · 3 · 4 | **người** | thuộc về ý nghĩa: ảnh nền thật hay thiếu nhãn, vật nhỏ có nên gán không… |

Máy ghi **35 thay đổi** vào `03_fixes.csv` (20 `drop_line`, 8 `clip_box`, 6 `needs_label`, 1 `rename_classes`). Sau sửa, `04_error_list_v2` chỉ còn **6 lỗi bắt buộc** (đúng 6 ảnh MISSING_LABEL_FILE). Có một tác dụng phụ: EMPTY_LABEL tăng từ 12 lên 15, vì 3 file chỉ có một dòng hỏng và sau khi xoá dòng đó thì file thành rỗng. Những ảnh này đã vào hàng đợi.

**Hàng đợi** (`05_review_queue.csv`, 125 ảnh): độ ưu tiên = tổng trọng số `review.weights` × (1 + conf). Vì vậy 6 ảnh NEEDS_LABEL (trọng số 10) đứng đầu, cùng img_0020 (11,5 điểm: một dòng bị xoá, một MISSING và một WRONG_CLASS truck/bus). Tiếp theo là các ảnh có WRONG_CLASS (3) hoặc MISSING (2) với conf cao. Ảnh chỉ có BOX_TINY (0,3) xếp cuối. Ảnh thứ 80, nằm ở mép ngân sách, là img_0012 (3,3 điểm, MISSING). Trong 80 ảnh đã duyệt (`06_review_log.csv`) có: **6/6** ảnh chưa có nhãn, **18/18** ảnh WRONG_CLASS, **50/52** ảnh MISSING, nhưng chỉ **8/24** ảnh LOOSE_BOX và **12/23** ảnh FALSE_POS. Lỗi ngữ nghĩa nặng đã được người xem gần hết; phần còn lại phần lớn là box lỏng hoặc có thể thừa, có trọng số thấp.

## 3. Calibration (2 ngưỡng)
Đo trên `data/calib` (39 ảnh, đáp án ở `MANIFEST.csv`); bảng sweep đầy đủ ở `submission/calibration/calibration_report.md`.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | NEAR: 1,00 / 0,50 → **1,00 / 0,67**; EXACT giữ 1,00 / 1,00 | Đây là lỗi máy **tự loại ảnh**, nên precision phải đứng trước. Trên calib, precision bằng 1,00 cho tới 20, nên calib không phân biệt được các giá trị. Kiểm trên raw: từ 10 đến 14 cho kết quả giống hệt nhau (4 NEAR + 5 LEAK, đã mở xem, đều trùng thật). Chọn 12 ở giữa vùng phẳng này để có biên an toàn về cả hai phía. |
| `second_opinion.conf` | 0,25 | **0,6** | MISSING 0,50 / 1,00 → **0,57 / 1,00**; WRONG_CLASS 0,56 / 0,83 → **0,62 / 0,83**; LOOSE 0,29 / 0,40 → 0,20 / 0,20; FALSE_POS 0,31 / 1,00 (không đổi, vì dùng `fp_conf`) | Đây là cảnh báo cho người, nên một FN (thiếu box lọt vào dataset) đắt hơn một FP (người mất vài giây xem). 0,6 là giá trị **cao nhất mà recall MISSING vẫn bằng 1,00**; precision MISSING và WRONG_CLASS đều tăng, hàng đợi trên raw giảm từ 143 xuống 125 ảnh. Lên 0,7 thì MISSING tụt còn 7/8, lên 0,8 còn 6/8. Cái giá phải chịu là LOOSE_BOX giảm (1/5), xem §4. |

**Một chỗ calibration có thể đánh lừa bạn:**
1. **`phash_dist = 16` đẹp nhất trên calib nhưng hỏng trên raw.** Trên calib, 16 cho NEAR 1,00 / 0,83 (F1 0,91, cao nhất, không có FP). Chạy cùng giá trị đó trên raw thì LEAK_TEST tăng từ 5 lên **11** và NEAR từ 4 lên **7**. Mình mở 9 cặp mới (`submission/evidence/phash_pairs.jpg`): **8 cặp là hai ảnh khác hẳn nhau** (sân tennis với người trượt tuyết, bàn làm việc với sân bóng chày, lướt sóng với bánh kem…), chỉ 1 cặp là trùng thật (img_0084/img_0074). Nếu chọn 16, pipeline sẽ loại nhầm 8 ảnh tốt, trong đó 6 ảnh bị gán là "rò rỉ test". Ở 20 còn tệ hơn: 105 LEAK giả. Calib chỉ có 39 ảnh và **không có ca LEAK_TEST** nào, nên không thể hiện được mức gộp nhầm này. Ngược lại, giá trị khởi đầu 2 thì bỏ sót trên raw một ca rò rỉ thật (img_0221 với test_009, phash 6) và hai cặp trùng thật (khoảng cách 6 và 10).
2. **`conf = 0.8` cho kết quả Release.** Mình chạy cả pipeline với từng giá trị conf. Từ 0,25 đến 0,7 đều ra Hold (28,7 % → 14,4 %); riêng 0,8 cho **Release** (7,9 %). Nhưng ở 0,8, recall MISSING trên calib chỉ còn 0,75, và trên raw số cờ MISSING giảm từ 53 xuống 32. Hàng đợi vừa ngân sách chỉ vì checker được làm cho "mù" bớt, tức là một Release giả, nên mình không chọn.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn đúng.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt):
`kết luận: REWORK — còn 6 lỗi bắt buộc (MISSING_LABEL_FILE) -> trả về sửa`, mã thoát **20**, log ghi `KHÔNG xuất: còn lỗi bắt buộc (Rework)`, và thư mục `work/exp_nb/` **không được tạo**. Sáu ảnh img_0031, img_0053, img_0126, img_0135, img_0170, img_0219 vẫn chưa có nhãn, và autofix không tạo file rỗng để "lách" cổng. Unit test `test_error_cannot_be_hidden_by_reviewed` cũng cho thấy lỗi bắt buộc không bị bỏ qua kể cả khi ảnh đã được duyệt.

**(b) Điểm mù (có bằng chứng):**
1. **Vật nhỏ và dày đặc.** Ở img_0160 (bãi xe tải, `submission/evidence/queue_top_viz.jpg`) có **hàng chục** xe tải không có nhãn, nhưng pipeline chỉ cờ **1** POSSIBLE_MISSING. Xe ở xa nhỏ hơn `min_area_px = 600` hoặc có conf < 0,6 nên bị bỏ qua. Ảnh này lên được đầu hàng đợi là nhờ có WRONG_CLASS, chứ không phải vì checker thấy đúng mức độ thiếu nhãn.
2. **Model lớn cũng nhầm lớp.** Trên calib, lỗi chèn truck→bus ở calib_009 không bị bắt: yolo26l nhìn vật đó và nói "car" (conf 0,72, IoU 0,96), trùng lớp với một nhãn car chồng lên nên không cờ. Nhầm lẫn giữa car, truck và bus chính là điểm yếu của checker. Cũng vì vậy mà trên raw, nhiều cờ WRONG_CLASS lại là truck/bus (img_0020, img_0160).
3. **Box lỏng nặng bị chẩn đoán sai loại.** Recall LOOSE_BOX trên calib chỉ là 1/5. Khi box bị phóng to khoảng 1,5 lần, IoU với dự đoán rơi xuống dưới `iou_missing = 0,3`, và lỗi hiện ra thành cặp MISSING + FALSE_POS (calib_002) hoặc chỉ FALSE_POS (calib_006, nơi model không thấy vật). Ảnh vẫn vào hàng đợi nhưng người duyệt nhận sai gợi ý.
4. **Ảnh trùng bị biến đổi mạnh.** calib_037 và calib_038 (mức biến đổi 3–4) không bị bắt ở mọi `phash_dist` ≤ 14. Trên raw, img_0084/img_0074 là trùng thật nhưng có khoảng cách 16, nên lọt qua với ngưỡng 12.

Lỗi mình tự nhìn thấy khi mở ảnh: img_0070 (một người bị gán `car`), img_0020 (xe buýt bị gán `truck`), img_0127 (một box person lỏng bao cả con bò, nhiều trẻ em không có box). Pipeline có cờ cả ba ảnh, nhưng với img_0127 thì chỉ cờ 1 MISSING trong khi thực tế thiếu nhiều box hơn.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Cấu hình:** `scripts/train_compare.py --variants raw clean --seeds 0 1 --epochs 20`: yolo26n.pt, imgsz 640, batch 16, cùng val (`gold_val`, 40 ảnh) và cùng test (100 ảnh nhãn người). `clean` là bản **nháp** `clean_v1` (pipeline kết luận Hold), `manifest_sha1 = 0d0bce97…`.
Về nơi chạy: Colab chỉ cấp CPU (`nvidia-smi: command not found`, ước tính khoảng 4 giờ), nên mình chuyển sang GPU trên máy (RTX 3050 Laptop 4 GB, ultralytics **8.4.215**, notebook ghim 8.4.214). Máy chỉ còn khoảng 2,7 GB RAM trống nên DataLoader worker bị chết. Mình chạy `train_compare.py` **nguyên bản** qua một wrapper ngoài repo (`work/train_w0.py`) chỉ ép `workers=0` cho `train`/`val`; mọi siêu tham số, seed và tập dữ liệu không đổi, và áp dụng giống nhau cho cả hai biến thể.

| biến thể | #ảnh train | số seed | mAP50 (TB ± độ lệch) | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.703 ± 0.009 | 0.497 ± 0.017 | 0.775 ± 0.082 | 0.608 ± 0.044 |
| clean | 202 | 2 | 0.704 ± 0.008 | 0.487 ± 0.010 | 0.824 ± 0.055 | 0.594 ± 0.035 |

(`submission/results/comparison.md`, sinh bởi `summarize_results.py`.) Theo từng seed, mAP50 của raw là 0,709 / 0,696 và của clean là 0,699 / 0,710.

**So với nhiễu:** chênh mAP50 (clean − raw) = **+0,002**, trong khi độ lệch giữa 2 seed của *mỗi* biến thể là khoảng 0,008–0,009, và riêng seed 0 thì raw còn cao hơn clean. mAP50-95 thì clean thấp hơn 0,010, cũng nằm trong ±0,017. **Kết luận trung thực: chưa thể nói clean tốt hơn hay kém hơn raw.** Chênh lệch nằm trong nhiễu, với 2 seed và test chỉ 100 ảnh. Thay đổi có hướng rõ hơn là precision tăng (0,775 → 0,824) đổi lại recall giảm nhẹ (0,608 → 0,594), nhưng độ lệch giữa các seed của precision (±0,06–0,08) vẫn lớn hơn mức chênh.
Theo lớp (AP50, trung bình 2 seed): bus 0,642 → **0,672** (+0,03), truck 0,593 → 0,596, car 0,745 → 0,728 (−0,016), person 0,831 → 0,821 (−0,011). Bus là lớp duy nhất tăng đáng kể, khớp với việc 18/18 ảnh WRONG_CLASS (chủ yếu nhầm truck/bus) đã được người duyệt sửa; nhưng chỉ có 2 seed nên chưa chắc chắn.

**Giả thuyết vì sao không thấy khác biệt, và cách kiểm:**
1. **Ultralytics đã tự "làm sạch" một phần raw:** khi train raw, nó tự bỏ 11 ảnh hỏng (class 93/7/4, nhãn âm, box tràn, ảnh cắt cụt), nên raw thực tế chỉ train trên **213** ảnh. Lỗi cú pháp, vốn là phần pipeline sửa chắc chắn nhất, gần như không ảnh hưởng tới raw. *Kiểm:* đếm dòng `ignoring corrupt` trong log train.
2. **Raw có 5 ảnh rò rỉ test (LEAK_TEST) và 12 ảnh trùng**, nên điểm test của raw có thể bị thổi phồng, che bớt lợi ích của clean. *Kiểm:* đo lại mAP trên test sau khi bỏ 5 ảnh test bị rò (test_009/033/039/044/085), hoặc so AP trên riêng 5 ảnh đó giữa hai model.
3. **Clean ít dữ liệu hơn (202 so với 224) và vẫn còn nhãn máy chưa duyệt** (38 ảnh còn cảnh báo, 20 LOOSE_BOX, 21 FALSE_POS; img_0160 còn hàng chục xe tải không có box). Bản "clean" chưa phải chặn trên. *Kiểm:* chạy thêm biến thể `oracle` (nhãn người) để xem khoảng trống còn lại; tăng lên 3–5 seed và tăng epoch.

**Việc tiếp theo:** thêm seed (≥ 3) và biến thể `oracle`; đo lại sau khi duyệt hết hàng đợi (Release); báo cáo mAP trên test đã loại 5 ảnh rò rỉ.

## 6. Quyết định và ghi chú AI
**(a) Quyết định: Hold.** Không còn lỗi bắt buộc nào, nhưng 38/202 ảnh (18,8 %) còn cảnh báo chưa được duyệt. Bản `clean_v1` chỉ là bản nháp để thử nghiệm, chưa phát hành. Để thành **Release**:
- người duyệt thêm khoảng **18 ảnh** nữa theo đúng thứ tự `05_review_queue.csv` (tổng khoảng 98 ảnh) để số ảnh cảnh báo chưa duyệt còn ≤ 20/202 (< 10 %); hoặc
- chạy thêm một vòng duyệt tập trung vào 20 ảnh LOOSE_BOX và 21 ảnh FALSE_POS còn lại.

Kết quả train (§5) cũng không cho lý do để phát hành sớm: clean và raw ngang nhau trong nhiễu. Không được đạt Release bằng cách nâng `conf` lên 0,8 (§3). Nếu sau vòng duyệt mà người xác nhận thêm ảnh thiếu nhãn, ảnh đó phải được gán xong trước khi phát hành, nếu không gate sẽ trả Rework.

**(b) AI:** có dùng (Claude Code) để viết 4 module.
- *Prompt chính:* "đọc GUIDE.md, docstring của từng stub và test; viết `label_format`, `gate`, `autofix`, `second_opinion` đúng đầu vào/ra, ngưỡng lấy từ `cfg`, không đọc `data/test`/`MANIFEST`, không sửa `data/raw`, chạy pytest."
- *Cách kiểm:* chạy `pytest` cho từng module rồi cả bộ (**79 passed**). Tự thử thêm 3 ca: một dòng vừa tràn ảnh vừa lặp box (chỉ ghi 1 `drop_line`), file có dòng trống ở giữa (số dòng gốc giữ nguyên, chạy lần 2 không đổi gì), ảnh có 5 dự đoán (ra đúng MISSING, WRONG_CLASS, LOOSE và bỏ qua dự đoán có conf thấp). Sau đó đối chiếu với calib và mở ảnh bằng `viz_labels.py`.
- *Một lần AI sai:* lần đầu ghi `autofix.py`, chuỗi `"\n"` bị biến thành ký tự xuống dòng thật bên trong literal, khiến file lỗi cú pháp; pytest báo lỗi ngay ở bước collection nên phát hiện và sửa được. Ngoài ra, nếu chỉ nhìn kết quả gate, AI có thể chọn `conf = 0.8` để ra Release; việc đối chiếu sweep cho thấy đó là Release giả (§3).
