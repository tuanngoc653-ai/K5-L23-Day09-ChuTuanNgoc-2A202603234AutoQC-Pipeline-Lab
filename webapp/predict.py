"""Chạy model YOLO lên mọi ảnh của một dataset, ghi dự đoán theo LỚP CỦA DATASET (dùng cho "ý kiến thứ hai").

Chạy bằng một Python CÓ ultralytics (có thể khác Python chạy server):
    python webapp/predict.py --weights yolo26l.pt --images <thư mục ảnh> --classes car,person --out preds.json
                             [--conf 0.1] [--imgsz 640] [--device auto] [--progress progress.json] [--inspect]

- Lớp của model được ghép với lớp của dataset THEO TÊN (không phân biệt hoa thường, có vài tên đồng nghĩa).
  Lớp của model không ghép được thì bỏ qua.
- Ghi mọi ảnh vào `out`, kể cả ảnh không có dự đoán (danh sách rỗng): {tên_ảnh_không_đuôi: [[id_lớp, conf, cx, cy, w, h], ...]}.
- `--inspect`: chỉ in JSON {names, mapping, unmatched} rồi thoát (để giao diện báo trước lớp nào khớp).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SYNONYMS = {
    "person": {"people", "pedestrian", "human"},
    "motorcycle": {"motorbike", "moto", "xe máy"},
    "bicycle": {"bike"},
    "airplane": {"aeroplane", "plane"},
    "tv": {"tvmonitor", "monitor"},
    "couch": {"sofa"},
    "dining table": {"diningtable", "table"},
    "potted plant": {"pottedplant"},
    "cell phone": {"phone", "mobile phone"},
}


def norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def build_mapping(model_names: dict, classes: list[str]) -> dict[int, int]:
    """id lớp model -> id lớp dataset (khớp theo tên hoặc tên đồng nghĩa)."""
    want = {norm(c): i for i, c in enumerate(classes)}
    out = {}
    for mid, mname in model_names.items():
        n = norm(mname)
        cands = {n} | SYNONYMS.get(n, set())
        for k, syns in SYNONYMS.items():
            if n in syns:
                cands |= {k} | syns
        hit = next((want[c] for c in cands if c in want), None)
        if hit is not None:
            out[int(mid)] = hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images")
    ap.add_argument("--classes", required=True, help="tên lớp của dataset, cách nhau bằng dấu phẩy, đúng thứ tự id")
    ap.add_argument("--out")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--progress")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    classes = [c.strip() for c in a.classes.split(",")]
    model = YOLO(a.weights)
    names = {int(k): v for k, v in model.names.items()}
    mapping = build_mapping(names, classes)
    matched = set(mapping.values())
    info = dict(names=names, mapping={str(k): v for k, v in mapping.items()},
                matched=[classes[i] for i in sorted(matched)], unmatched=[c for i, c in enumerate(classes) if i not in matched])
    if a.inspect:
        print(json.dumps(info, ensure_ascii=False))
        return
    if not mapping:
        sys.exit("Model không có lớp nào trùng tên với lớp của dataset.")

    device = None
    if a.device != "auto":
        device = a.device
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    out: dict = {p.stem: [] for p in imgs}
    t0 = time.time()
    B = 16
    for i in range(0, len(imgs), B):
        batch = imgs[i:i + B]
        try:
            results = model.predict([str(p) for p in batch], conf=a.conf, imgsz=a.imgsz, classes=sorted(mapping), device=device, verbose=False)
        except Exception:                                   # một ảnh hỏng làm hỏng cả lô: chạy lại từng ảnh, bỏ qua ảnh lỗi
            results = []
            for p in batch:
                try:
                    results += model.predict(str(p), conf=a.conf, imgsz=a.imgsz, classes=sorted(mapping), device=device, verbose=False)
                except Exception:
                    pass
        for r in results:
            rows = [[mapping[int(c)], round(float(s), 4)] + [round(float(v), 6) for v in xywhn]
                    for c, s, xywhn in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()) if int(c) in mapping]
            out[Path(r.path).stem] = rows
        if a.progress:
            Path(a.progress).write_text(json.dumps(dict(done=min(i + B, len(imgs)), total=len(imgs), seconds=round(time.time() - t0, 1))), encoding="utf-8")
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(dict(images=len(imgs), seconds=round(time.time() - t0, 1), **info), ensure_ascii=False))


if __name__ == "__main__":
    main()
