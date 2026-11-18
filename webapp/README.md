# AutoQC web: dùng pipeline kiểm nhãn qua trình duyệt

Web chạy **cục bộ**, gọi đúng `python -m pipeline.run`. Đây không phải phần chấm điểm, và web không ghi gì vào `submission/`, `configs/` hay `data/`.

```bash
pip install -r webapp/requirements.txt
python webapp/server.py        # mở http://127.0.0.1:8009
```

## Làm gì trên web
1. **Chỉnh 2 ngưỡng** (`phash_dist`, `conf`) và thư mục lô (mặc định `data/raw`; lô khác cần có `images/`, `labels/`, `data.yaml`).
2. **▶ Chạy tự động**: người duyệt được mô phỏng (80 ảnh), ra kết luận Release / Hold / Rework ngay.
3. **⏸ Chạy & dừng ở hàng đợi**: pipeline dừng (mã 30) để **bạn** duyệt. Ở tab *Duyệt ảnh*:
   - danh sách xếp theo `05_review_queue.csv`;
   - ảnh hiện nhãn (nét liền), dự đoán yolo26l (nét đứt), nhãn gốc (chấm xám), ô "thiếu?" màu đỏ (POSSIBLE_MISSING), dòng bị cờ có viền đỏ;
   - sửa ngay trên ảnh: bấm chọn box rồi đổi lớp, kéo để di chuyển, kéo góc để đổi cỡ, **Del** để xoá, **＋ Vẽ box** để thêm, **Ctrl+S** để lưu; **J/K** chuyển ảnh.
4. **⏵ Chạy tiếp sau khi duyệt**: ảnh có nhãn đổi nội dung được tính là đã duyệt (tối đa 80), rồi QC vòng 3 và cửa chặn.

Tab *Tổng quan* hiện kết luận, tỉ lệ ảnh còn cảnh báo so với ngưỡng Hold, và bảng số lỗi theo mã qua 3 vòng (trước sửa, sau máy sửa, cuối). Bấm một mã để lọc ảnh có mã đó.

Kết quả nằm trong `work/web/` (`reports/`, `run/ds/`, `export/`).
