"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def _label_boxes(ds, stem):
    """[(số dòng gốc, id lớp, (cx, cy, w, h))] của các dòng nhãn hợp lệ; dòng hỏng / id lạ bị bỏ qua (check khác báo)."""
    out = []
    for ln, text in ds.read_label_lines(stem) or []:
        try:
            c, cx, cy, w, h = parse_yolo_line(text)
        except ValueError:
            continue
        if 0 <= c < len(ds.classes):
            out.append((ln, c, (cx, cy, w, h)))
    return out


def run(ds, cfg, preds=None):
    p = cfg["second_opinion"]
    conf_min, min_area, fp_conf = p["conf"], p["min_area_px"], p["fp_conf"]
    iou_missing, iou_match, iou_loose = p["iou_missing"], p["iou_match"], p["iou_loose"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    names = ds.classes
    issues = []

    def add(code, stem, msg, line=None, **extra):
        issues.append(Issue("second_opinion", WARN, code, stem, msg, line, extra))

    for img in ds.image_files():
        stem = img.stem
        if stem not in preds or ds.read_label_lines(stem) is None:
            continue
        size = ds.image_size(stem)
        if size is None:                                   # ảnh hỏng: check images báo
            continue
        W, H = size
        labels = _label_boxes(ds, stem)
        rows = []
        for r in preds[stem]:
            try:
                rc, rconf, *rbox = r
                rows.append((int(rc), float(rconf), tuple(float(v) for v in rbox[:4])))
            except (TypeError, ValueError):
                continue

        # --- mỗi dự đoán tin cậy, đủ lớn: tối đa MỘT issue
        for rc, rconf, rbox in rows:
            if rconf < conf_min or rbox[2] * W * rbox[3] * H < min_area:
                continue
            cands = [(iou_xywh(rbox, lb), lc) for _, lc, lb in labels]
            cands = [(iou, lc) for iou, lc in cands if iou >= iou_missing]
            cname = names[rc] if 0 <= rc < len(names) else rc
            if not cands:
                add("POSSIBLE_MISSING", stem, f"model thấy {cname} (conf {rconf:.2f}) mà nhãn không có box",
                    conf=round(rconf, 4), cls=rc, box=[round(v, 4) for v in rbox])
                continue
            same = [iou for iou, lc in cands if lc == rc]
            if same:
                best = max(same)
                if best < iou_loose:
                    add("POSSIBLE_LOOSE_BOX", stem, f"box {cname} lỏng/lệch: IoU {best:.2f} < {iou_loose}",
                        conf=round(rconf, 4), iou=round(best, 4), cls=rc)
                continue                                   # nhãn đúng lớp đã phủ vật này
            best_iou, best_cls = max(cands)
            if best_iou >= iou_match:
                other = names[best_cls]
                add("POSSIBLE_WRONG_CLASS", stem, f"nhãn {other} nhưng model nói {cname} (conf {rconf:.2f}, IoU {best_iou:.2f})",
                    conf=round(rconf, 4), iou=round(best_iou, 4), cls=rc, label_cls=best_cls)
            # IoU khác lớp trong [iou_missing, iou_match): mơ hồ -> không cờ

        # --- mỗi box nhãn: không khớp dự đoán nào (conf >= fp_conf, không lọc diện tích)
        fp_boxes = [rbox for _, rconf, rbox in rows if rconf >= fp_conf]
        for ln, lc, lb in labels:
            best = max((iou_xywh(lb, rb) for rb in fp_boxes), default=0.0)
            if best < iou_missing:
                add("POSSIBLE_FALSE_POS", stem, f"box {names[lc]} không khớp vật nào model thấy (IoU tốt nhất {best:.2f})",
                    ln, iou=round(best, 4), cls=lc)
    return issues
