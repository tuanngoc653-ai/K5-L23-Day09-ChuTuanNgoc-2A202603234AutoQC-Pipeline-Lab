"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def _overflow(cx, cy, w, h):
    """Mức tràn lớn nhất của 4 mép box ra ngoài [0,1] (0 nếu nằm trọn trong ảnh)."""
    return max(0.0, -(cx - w / 2), (cx + w / 2) - 1.0, -(cy - h / 2), (cy + h / 2) - 1.0)


def run(ds, cfg):
    p = cfg["label_format"]
    tol, min_px, max_frac, dup_iou = p["overflow_tol"], p["min_box_px"], p["max_box_frac"], p["dup_box_iou"]
    n_cls = len(ds.classes)
    issues = []

    def add(sev, code, stem, msg, line, extra=None):
        issues.append(Issue("label_format", sev, code, stem, msg, line, extra or {}))

    for lp in ds.label_files():
        stem = lp.stem
        size = ds.image_size(stem)                                    # None (nhãn mồ côi / ảnh hỏng): bỏ qua BOX_TINY
        prev = []                                                     # (số dòng, box) các dòng hợp lệ đã thấy
        for ln, text in ds.read_label_lines(stem) or []:              # số dòng GỐC, đã bỏ dòng trống
            try:
                c, cx, cy, w, h = parse_yolo_line(text)
            except ValueError as e:
                add(ERROR, "BAD_FORMAT", stem, f"dòng không đúng 5 cột số: {e}", ln)
                continue
            if not 0 <= c < n_cls:
                add(ERROR, "BAD_CLASS_ID", stem, f"id lớp {c} ngoài 0..{n_cls - 1}", ln, {"class_id": c})
                continue
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", stem, f"box rỗng: w={w}, h={h}", ln)
                continue
            box = (cx, cy, w, h)
            over = _overflow(*box)
            if over > tol:
                add(ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn ảnh {over:.4f} > {tol}", ln, {"overflow": round(over, 6)})
            dup = next((pl for pl, (pc, pb) in prev if pc == c and iou_xywh(pb, box) > dup_iou), None)
            if dup is not None:
                add(ERROR, "DUP_BOX", stem, f"box lặp với dòng {dup} (cùng lớp {c})", ln, {"of_line": dup})
            if size:
                short = min(w * size[0], h * size[1])
                if short < min_px:
                    add(WARN, "BOX_TINY", stem, f"cạnh ngắn {short:.1f}px < {min_px}px", ln, {"short_px": round(short, 1)})
            if w * h > max_frac:
                add(WARN, "BOX_HUGE", stem, f"box chiếm {w * h:.2f} ảnh > {max_frac}", ln, {"frac": round(w * h, 4)})
            prev.append((ln, (c, box)))
    return issues
