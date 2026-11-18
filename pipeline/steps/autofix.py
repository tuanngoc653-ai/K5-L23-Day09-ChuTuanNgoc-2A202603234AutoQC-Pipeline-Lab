"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}          # + mọi mã bắt đầu bằng DUP_IMAGE_


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _clip(text):
    """Cắt box về [0,1]. Trả về dòng mới, hoặc None nếu sau khi cắt box không còn diện tích (hoặc dòng không đọc được)."""
    try:
        c, cx, cy, w, h = parse_yolo_line(text)
    except ValueError:
        return None
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    nw, nh = round(x2 - x1, 6), round(y2 - y1, 6)
    if nw <= 0 or nh <= 0:
        return None
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {nw:.6f} {nh:.6f}"


def _remove_image(ds, stem):
    """Xoá ảnh (mọi đuôi) + nhãn của `stem` khỏi ds. Trả về tên file ảnh đã xoá, None nếu đã không còn."""
    imgs = [p for p in ds.image_files() if p.stem == stem]
    lp = ds.label_path(stem)
    if not imgs and not lp.exists():
        return None
    for p in imgs:
        p.unlink()
    lp.unlink(missing_ok=True)
    ds._size_cache.pop(stem, None)
    return ", ".join(p.name for p in imgs) or lp.name


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]          # cảnh báo là việc của người: không động tới
    fixes = []

    # 1) xoá ảnh (trùng, rò rỉ test, hỏng, sai kích thước) — làm trước để bỏ qua issue dòng của ảnh đã xoá
    removed = set()
    for i in errors:
        if _is_remove(i.code) and i.image not in removed:
            removed.add(i.image)
            gone = _remove_image(ds, i.image)
            if gone:
                fixes.append(dict(image=i.image, line="", action="remove_image", before=gone, after=f"(đã xoá: {i.code})"))

    # 2) data.yaml sai tên lớp: ghi lại names đúng cfg["classes"] (một lần)
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        d = {}
        if ds.yaml_path.exists():
            d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        before = d.get("names")
        want = {k: v for k, v in enumerate(cfg["classes"])}
        if before != want:
            d["names"] = want
            if "nc" in d:
                d["nc"] = len(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append(dict(image="data.yaml", line="", action="rename_classes", before=str(before), after=str(cfg["classes"])))

    # 3) ảnh chưa có nhãn: máy không bịa nhãn, chuyển người gán (không tạo file rỗng)
    for i in errors:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed and not ds.label_path(i.image).exists():
            fixes.append(dict(image=i.image, line="", action="needs_label", before="(không có file)", after="chuyển người gán"))

    # 4) lỗi theo dòng: gom theo file, số dòng là số dòng GỐC, ghi lại file MỘT lần
    drop, clip = defaultdict(set), defaultdict(set)
    for i in errors:
        if i.line is None or i.image in removed:
            continue
        if i.code in DROP_CODES:
            drop[i.image].add(int(i.line))
        elif i.code == "BOX_OUT_OF_IMAGE":
            clip[i.image].add(int(i.line))
    for stem in sorted(set(drop) | set(clip)):
        lp = ds.label_path(stem)
        if not lp.exists():
            continue
        raw = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        out, changed = [], False
        for n, text in enumerate(raw, 1):                          # dòng trống vẫn được đếm như số dòng gốc
            if n in drop[stem]:                                    # xoá thắng cắt: chỉ ghi 1 fix drop_line
                fixes.append(dict(image=stem, line=n, action="drop_line", before=text, after=""))
                changed = True
            elif n in clip[stem]:
                new = _clip(text)
                if new is None:                                    # box nằm hẳn ngoài ảnh: xoá thay vì ghi box rỗng
                    fixes.append(dict(image=stem, line=n, action="drop_line", before=text, after="(box ngoài ảnh)"))
                else:
                    fixes.append(dict(image=stem, line=n, action="clip_box", before=text, after=new))
                    out.append(new)
                changed = True
            else:
                out.append(text)
        if changed:
            with open(lp, "w", encoding="utf-8", newline="\n") as f:
                f.write("".join(l + "\n" for l in out))
    return fixes
