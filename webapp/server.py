"""Web tương tác cho pipeline kiểm nhãn (chạy CỤC BỘ, không phải phần chấm điểm).

    pip install flask
    python webapp/server.py            # mở http://127.0.0.1:8009

Mọi thứ web sinh ra nằm trong work/web/ (không đụng submission/, configs/, data/):
  work/web/pipeline.yaml   cấu hình = configs/pipeline.yaml + 2 ngưỡng bạn chỉnh trên web
  work/web/run/            thư mục làm việc của pipeline (ds/ là bản sao đang sửa)
  work/web/reports/        báo cáo 01..08
  work/web/export/         dataset xuất (khi không Rework)
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import IMG_EXTS, load_config, parse_yolo_line  # noqa: E402
from pipeline.model_cache import load_cache  # noqa: E402
from pipeline.run import nsha  # noqa: E402

WEB = ROOT / "work" / "web"
WORK, REP, EXP, CFG, LAST = WEB / "run", WEB / "reports", WEB / "export", WEB / "pipeline.yaml", WEB / "last_run.json"
DS = WORK / "ds"

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
_preds_cache: dict = {}


# ---------- tiện ích
def cfg_now() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else load_config()


def last_run() -> dict:
    return json.loads(LAST.read_text(encoding="utf-8")) if LAST.exists() else {}


def read_csv(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def latest_issue_file() -> Path:
    """Danh sách lỗi mới nhất: v3 nếu đã chạy xong, nếu không thì v2 (đang chờ người duyệt)."""
    return REP / "07_error_list_v3.csv" if (REP / "07_error_list_v3.csv").exists() else REP / "04_error_list_v2.csv"


def preds() -> dict:
    path = cfg_now()["models"]["checker"]["cache"]
    if path not in _preds_cache:
        _preds_cache[path] = load_cache(path)
    return _preds_cache[path]


def find_image(folder: Path, stem: str) -> Path | None:
    d = folder / "images"
    if not d.exists():
        return None
    return next((p for p in d.iterdir() if p.stem == stem and p.suffix.lower() in IMG_EXTS), None)


def raw_dir() -> Path:
    r = last_run().get("raw", "data/raw")
    return Path(r) if Path(r).is_absolute() else ROOT / r


def label_lines(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    out = []
    for n, text in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not text.strip():
            continue
        try:
            c, cx, cy, w, h = parse_yolo_line(text)
            out.append(dict(line=n, text=text, cls=c, box=[cx, cy, w, h]))
        except ValueError as e:
            out.append(dict(line=n, text=text, error=str(e)))
    return out


def edited_stems() -> list[str]:
    """Ảnh có nhãn đổi nội dung so với lúc dừng chờ duyệt (đúng cách pipeline tính 'đã duyệt')."""
    snap = WORK / "snapshot.json"
    if not snap.exists() or not (DS / "images").exists():
        return []
    before = json.loads(snap.read_text(encoding="utf-8"))
    stems = [p.stem for p in (DS / "images").iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(s for s in stems if nsha(DS / "labels" / f"{s}.txt") != before.get(s))


def agg(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        a = out.setdefault(r["code"], {"severity": r["severity"], "n": 0, "images": set()})
        a["n"] += 1
        a["images"].add(r["image"])
    return {c: {"severity": v["severity"], "n": v["n"], "images": len(v["images"])} for c, v in out.items()}


# ---------- trang
@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/state")
def state():
    cfg = cfg_now()
    lr = last_run()
    stages = {}
    for key, f in [("intake", "01_intake.csv"), ("v1", "02_error_list_v1.csv"), ("v2", "04_error_list_v2.csv"), ("v3", "07_error_list_v3.csv")]:
        stages[key] = agg(read_csv(REP / f)) if (REP / f).exists() else None
    summary = json.loads((REP / "08_summary.json").read_text(encoding="utf-8")) if (REP / "08_summary.json").exists() else None
    fixes = read_csv(REP / "03_fixes.csv")
    fix_count: dict = {}
    for f in fixes:
        fix_count[f["action"]] = fix_count.get(f["action"], 0) + 1
    n_scanned = None
    rd = raw_dir() / "images"
    if rd.exists():
        n_scanned = sum(p.suffix.lower() in IMG_EXTS for p in rd.iterdir())
    n_ds = sum(p.suffix.lower() in IMG_EXTS for p in (DS / "images").iterdir()) if (DS / "images").exists() else None
    paused = lr.get("exit") == 30
    return jsonify(
        has_run=bool(lr), last=lr, paused=paused, summary=summary, stages=stages, fixes=fix_count,
        queue_len=len(read_csv(REP / "05_review_queue.csv")), reviewed_sim=len(read_csv(REP / "06_review_log.csv")),
        edited=edited_stems() if paused else [], n_scanned=n_scanned, n_ds=n_ds,
        thresholds=dict(phash_dist=cfg["duplicates"]["phash_dist"], conf=cfg["second_opinion"]["conf"]),
        budget=cfg["review"]["budget_images"], hold_ratio=cfg["gate"]["warn_ratio_hold"], classes=cfg["classes"],
    )


@app.post("/api/run")
def run_pipeline():
    body = request.get_json(force=True) or {}
    mode = body.get("mode", "full")                      # full | no_review | resume
    if mode not in ("full", "no_review", "resume"):
        abort(400, "mode sai")
    lr = last_run()
    raw = body.get("raw") or lr.get("raw") or "data/raw"
    if mode == "resume":
        if lr.get("exit") != 30:
            abort(400, "chưa có lần chạy nào đang chờ người duyệt")
        raw = lr.get("raw", raw)
    else:
        rp = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if not (rp / "images").is_dir():
            abort(400, f"không thấy thư mục {rp / 'images'}")
        cfg = load_config()
        cfg["duplicates"]["phash_dist"] = int(body.get("phash_dist", cfg["duplicates"]["phash_dist"]))
        cfg["second_opinion"]["conf"] = float(body.get("conf", cfg["second_opinion"]["conf"]))
        WEB.mkdir(parents=True, exist_ok=True)
        CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args = [sys.executable, "-m", "pipeline.run", "--raw", str(raw), "--work", str(WORK), "--reports", str(REP),
            "--export-dir", str(EXP), "--config", str(CFG)]
    if mode == "no_review":
        args.append("--no-review")
    elif mode == "resume":
        args.append("--resume-after-review")
    elif body.get("budget") not in (None, ""):
        args += ["--budget", str(int(body["budget"]))]
    t0 = time.time()
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONUTF8": "1"}, timeout=1800)
    rec = dict(mode=mode, raw=str(raw), exit=p.returncode, seconds=round(time.time() - t0, 1),
               log=(p.stdout + ("\n" + p.stderr if p.stderr.strip() else ""))[-20000:], at=time.strftime("%H:%M:%S"))
    LAST.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return jsonify(rec)


@app.get("/api/images")
def images():
    queue = read_csv(REP / "05_review_queue.csv")
    rank = {r["image"]: (i + 1, float(r["priority"]), r["codes"]) for i, r in enumerate(queue)}
    issues: dict = {}
    for r in read_csv(latest_issue_file()):
        issues.setdefault(r["image"], set()).add(r["code"])
    reviewed = {r["image"] for r in read_csv(REP / "06_review_log.csv")}
    edited = set(edited_stems()) if last_run().get("exit") == 30 else set()
    rows = []
    if (DS / "images").exists():
        for p in sorted((DS / "images").iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            s = p.stem
            rk = rank.get(s)
            rows.append(dict(image=s, rank=rk[0] if rk else None, priority=rk[1] if rk else 0, queue_codes=rk[2] if rk else "",
                             codes=sorted(issues.get(s, [])), reviewed=s in reviewed, edited=s in edited,
                             has_label=(DS / "labels" / f"{s}.txt").exists()))
    removed = [dict(image=r["image"], code=r["code"], message=r["message"]) for r in read_csv(REP / "01_intake.csv")]
    return jsonify(images=rows, removed=removed)


@app.get("/api/image/<stem>")
def image_info(stem):
    in_ds = find_image(DS, stem) is not None
    src = find_image(DS, stem) or find_image(raw_dir(), stem)
    if src is None:
        abort(404)
    try:
        with Image.open(src) as im:
            size = im.size
    except Exception:
        size = None
    def rows_for(f):
        return [dict(code=r["code"], severity=r["severity"], line=int(r["line"]) if r["line"] else None,
                     message=r["message"], extra=json.loads(r["extra"]) if r["extra"] else {})
                for r in read_csv(REP / f) if r["image"] == stem]
    latest = latest_issue_file().name
    q = next(((i + 1, r) for i, r in enumerate(read_csv(REP / "05_review_queue.csv")) if r["image"] == stem), None)
    return jsonify(
        image=stem, in_ds=in_ds, size=size,
        labels=label_lines(DS / "labels" / f"{stem}.txt") if in_ds else None,
        raw_labels=label_lines(raw_dir() / "labels" / f"{stem}.txt"),
        preds=[dict(cls=int(p[0]), conf=float(p[1]), box=[float(v) for v in p[2:6]]) for p in preds().get(stem, [])],
        issues_latest=rows_for(latest), latest_name=latest, issues_v1=rows_for("02_error_list_v1.csv"),
        intake=rows_for("01_intake.csv"),
        fixes=[f for f in read_csv(REP / "03_fixes.csv") if f["image"] == stem],
        queue=dict(rank=q[0], priority=q[1]["priority"], codes=q[1]["codes"]) if q else None,
        reviewed_sim=any(r["image"] == stem for r in read_csv(REP / "06_review_log.csv")),
        paused=last_run().get("exit") == 30,
    )


@app.get("/img/<stem>")
def image_file(stem):
    src = find_image(DS, stem) or find_image(raw_dir(), stem)
    if src is None:
        abort(404)
    return send_file(src)


@app.post("/api/labels/<stem>")
def save_labels(stem):
    if last_run().get("exit") != 30:
        abort(400, "chỉ sửa nhãn khi pipeline đang TẠM DỪNG chờ người duyệt (chạy 'Dừng ở hàng đợi' trước)")
    if find_image(DS, stem) is None:
        abort(404)
    n_cls = len(cfg_now()["classes"])
    lines = []
    for b in (request.get_json(force=True) or {}).get("boxes", []):
        c, cx, cy, w, h = int(b["cls"]), *[float(v) for v in b["box"]]
        if not 0 <= c < n_cls or w <= 0 or h <= 0:
            abort(400, f"box không hợp lệ: {b}")
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (DS / "labels").mkdir(parents=True, exist_ok=True)
    (DS / "labels" / f"{stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    ed = edited_stems()
    return jsonify(ok=True, edited=len(ed), budget=cfg_now()["review"]["budget_images"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8009))
    print(f"Mở http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
