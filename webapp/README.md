# AutoQC web: kiểm nhãn ảnh theo 3 bước

Web chạy **trên máy bạn**, dùng lại các bước của `pipeline/`. Đây không phải phần chấm điểm của lab; mọi thứ web sinh ra nằm trong `work/web/`, không đụng tới `submission/`, `configs/` hay `data/`.

```bash
pip install -r webapp/requirements.txt
python webapp/server.py        # mở http://127.0.0.1:8009
```

## Bước 1 · Kiểm tra
Nhập thư mục lô ảnh (mặc định `data/raw`; lô khác cần có `images/`, `labels/`, `data.yaml`) rồi bấm **Bắt đầu kiểm tra**. Máy sẽ loại ảnh hỏng, trùng hoặc lộ tập test, tự sửa lỗi định dạng nhãn, rồi lập danh sách các ảnh nó nghi gán sai.
*Cài đặt nâng cao*: 2 ngưỡng calibration, và **Chạy demo** (người duyệt mô phỏng của lab, ra kết quả ngay).

## Bước 2 · Duyệt ảnh
Đi lần lượt từ trên xuống danh sách (ảnh nghi sai nặng nhất đứng đầu; ngân sách 80 ảnh). Với mỗi ảnh:
- khung bên phải ghi **máy nghi gì** bằng lời thường, có nút **🔎 Xem chỗ này** và **nút sửa nhanh**: *Thêm box theo gợi ý*, *Đổi thành bus*, *Dùng box của model*, *Xoá box này*, *Lấy gợi ý model làm nháp*;
- nhãn đúng thì bấm **✓ Nhãn đúng → ảnh tiếp** (Enter);
- sửa xong thì bấm **💾 Lưu & sang ảnh tiếp** (Enter);
- sửa tay trên ảnh: bấm box để chọn, kéo để di chuyển, kéo góc để đổi cỡ, **✏️ Vẽ box** (phím D), **Del** để xoá;
- xem ảnh: lăn chuột để zoom, kéo để di chuyển, `0` để về vừa khung, **J/K** để sang ảnh sau/trước.

Ảnh **chưa có nhãn** (chữ đỏ) bắt buộc phải gán, hoặc bấm *Ảnh không có vật nào*.

## Bước 3 · Kết quả & xuất
Bấm **Xong, xem kết quả**. Máy kiểm lại lần cuối rồi kết luận:
- **ĐẠT**: không còn lỗi, và dưới 10% ảnh còn nghi ngờ chưa xem;
- **TẠM GIỮ**: không còn lỗi, nhưng từ 10% ảnh trở lên còn nghi ngờ chưa xem (trang ghi rõ cần duyệt thêm bao nhiêu ảnh);
- **CHƯA ĐẠT**: còn lỗi bắt buộc, ví dụ ảnh chưa có nhãn (bấm vào tên ảnh để đi sửa).

Khi ĐẠT hoặc TẠM GIỮ: bấm **⬇ Tải dataset (.zip)** để lấy `images/`, `labels/` (YOLO), `data.yaml` và `manifest.json`.

> Khác với `python -m pipeline.run --resume-after-review`: trên web, ảnh được **xác nhận "nhãn đúng"** cũng tính là đã duyệt (người duyệt thật có xem ảnh), chứ không chỉ ảnh bị sửa nhãn. Ngân sách vẫn là 80 ảnh.
