# AutoQC web: kiểm nhãn cho dataset YOLO bất kỳ

Công cụ chạy **trên máy bạn**: chọn một dataset YOLO, máy tự kiểm và tự sửa lỗi quy tắc, (tuỳ chọn) nhờ một model "nhìn lại" để tìm chỗ nghi gán sai, rồi cho người duyệt trên ảnh và xuất dataset sạch.

Công cụ dùng lại các check trong `pipeline/` của bài lab với một cấu hình sinh riêng cho từng dataset. Đây không phải phần chấm điểm; mọi thứ nằm trong `work/web/`, và **thư mục dataset gốc không bao giờ bị sửa**.

```bash
pip install -r webapp/requirements.txt
python webapp/server.py        # mở http://127.0.0.1:8009
```

## Dataset được hỗ trợ
- `images/` + `labels/` (YOLO: `id cx cy w h`, toạ độ 0–1), **hoặc** `images/<phần>/` + `labels/<phần>/` (ví dụ `train`, `val`): chọn phần cần kiểm trên web.
- `data.yaml` có `names` (list hoặc dict). Thiếu thì nhập tên lớp trên web. **Thứ tự lớp = id**, sửa được.
- Ảnh `.jpg .jpeg .png` (không phân biệt hoa thường).

## Ý kiến thứ hai (model), tuỳ chọn
| Lựa chọn | Khi nào dùng |
|---|---|
| **Tắt** | Chỉ kiểm quy tắc (định dạng, mã lớp, box tràn/rỗng/lặp, ảnh hỏng/trùng/lộ test). Nhanh, không cần model. |
| **Model YOLO** | Model của bạn (train cho đúng các lớp này), hoặc model COCO (`yolo26l.pt`…) nếu lớp của bạn có trong COCO. Lớp được ghép **theo tên** (có vài tên đồng nghĩa: motorbike ↔ motorcycle…). Bấm **Kiểm tra model** để xem lớp nào khớp. |
| **Dự đoán có sẵn** | Chỉ với dữ liệu bài lab (`data/raw`). |

Model chạy bằng một Python **có cài `ultralytics`** (có thể khác Python chạy server). Điền đường dẫn trên web, hoặc đặt sẵn biến môi trường:
```bash
set AUTOQC_PREDICT_PYTHON=C:\path\to\venv\Scripts\python.exe
```
Dự đoán được lưu riêng theo từng dataset và model (`work/web/preds.json`), và dùng lại nếu chạy lại cùng dataset và model.

> ⚠ Lớp **không có** trong model thì mọi box của lớp đó sẽ bị cờ "box thừa". Khi đó hãy dùng model của bạn, hoặc tắt.

## Quy tắc dự án (mục nâng cao)
Ngân sách duyệt (mặc định 80 ảnh), ngưỡng Tạm giữ (10%), cạnh ảnh tối thiểu (250 px), thư mục ảnh tập test để phát hiện ảnh lộ, độ nhạy phát hiện ảnh trùng (`phash_dist` 12), độ tin tối thiểu của model (`conf` 0.6). Các giá trị mặc định là của bài lab: **hãy calibrate lại cho dataset của bạn** (`python -m pipeline.calibrate`, xem `docs/CALIBRATION.md`).

## 3 bước
1. **Kiểm tra**: đọc thư mục → chọn model → *Bắt đầu kiểm tra* (chạy nền, có tiến độ).
2. **Duyệt ảnh**:
   - danh sách xếp ảnh nghi sai nặng nhất lên đầu;
   - mỗi ảnh có lời giải thích và **nút sửa nhanh** (Thêm box theo gợi ý, Đổi lớp, Dùng box của model, Xoá box);
   - nhãn đúng thì **✓ Nhãn đúng**, sửa xong thì **💾 Lưu** (Enter);
   - xem ảnh: lăn chuột để zoom, kéo để di chuyển; **J/K** để sang ảnh sau/trước.
3. **Kết quả và xuất**:
   - **ĐẠT** / **TẠM GIỮ** / **CHƯA ĐẠT**, kèm việc cần làm tiếp;
   - **⬇ Tải dataset (.zip)**: `images/`, `labels/`, `data.yaml`, `manifest.json`.

Trên web, ảnh được **xác nhận "nhãn đúng"** cũng tính là đã duyệt (khác `pipeline.run --resume-after-review`, vốn chỉ tính ảnh bị sửa nhãn).

## Giới hạn hiện tại
- Một dataset tại một thời điểm (kiểm dataset mới sẽ thay kết quả cũ trong `work/web/`).
- Chỉ bounding box (không segment/pose). Dataset rất lớn (hàng chục nghìn ảnh) sẽ chậm, vì các check của lab duyệt ảnh tuần tự.
- Model "ý kiến thứ hai" có điểm mù riêng: vật nhỏ và dày đặc, các lớp dễ nhầm với nhau. Model nhỏ (yolo26n) bỏ sót nhiều hơn, nên sẽ cờ "box thừa" nhiều hơn.
