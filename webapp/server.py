"""Web tương tác cho pipeline kiểm nhãn (chạy CỤC BỘ, không phải phần chấm điểm).

    pip install -r webapp/requirements.txt
    python webapp/server.py            # mở http://127.0.0.1:8009

Luồng 3 bước:
  1. Kiểm tra   : chạy pipeline tới hàng đợi rồi dừng (= python -m pipeline.run --no-review)
  2. Duyệt ảnh  : người xem từng ảnh trong hàng đợi; sửa nhãn hoặc xác nhận "nhãn đúng"
  3. Kết quả    : QC vòng cuối + cửa chặn + xuất (giống phần sau người duyệt của pipeline.run),
                  với "đã duyệt" = ảnh đã sửa  ∪  ảnh được xác nhận đúng (ngân sách vẫn 80 ảnh)
Chế độ demo (= python -m pipeline.run) dùng người duyệt mô phỏng của lab.

Mọi thứ web sinh ra nằm trong work/web/ (không đụng submission/, configs/, data/).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.checks.stats import class_counts  # noqa: E402
from pipeline.common import ERROR, IMG_EXTS, Dataset, load_config, parse_yolo_line  # noqa: E402
from pipeline.model_cache import load_cache  # noqa: E402
from pipeline.run import EXIT, qc, snapshot, write_csv  # noqa: E402
from pipeline.steps import autofix, export, gate  # noqa: E402
from pipeline.steps.report import write_report  # noqa: E402

WEB = ROOT / "work" / "web"
WORK, REP, EXP, CFG = WEB / "run", WEB / "reports", WEB / "export", WEB / "pipeline.yaml"
LAST, CONFIRMED = WEB / "last_run.json", WEB / "confirmed.json"
DS = WORK / "ds"
EXPORT_NAME = "clean_v1"

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
_preds_cache: dict = {}


# ---------------------------------------------------------------- tiện ích
def cfg_now() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else load_config()


def last_run() -> dict:
    return json.loads(LAST.read_text(encoding="utf-8")) if LAST.exists() else {}


def save_last(rec: dict):
    WEB.mkdir(parents=True, exist_ok=True)
    LAST.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def read_csv(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def confirmed() -> set[str]:
    return set(json.loads(CONFIRMED.read_text(encoding="utf-8"))) if CONFIRMED.exists() else set()


def latest_issue_file() -> Path:
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


def ds_stems() -> list[str]:
    d = DS / "images"
    return sorted(p.stem for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.exists() else []


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


def review_session() -> bool:
    """Đang có phiên duyệt của người (bắt đầu bằng bước Kiểm tra, không phải demo)."""
    return last_run().get("mode") in ("check", "finish") and (WORK / "snapshot.json").exists()


def edited_stems() -> list[str]:
    if not review_session():
        return []
    before = json.loads((WORK / "snapshot.json").read_text(encoding="utf-8"))
    now = snapshot(Dataset(DS, cfg_now()["classes"]))
    return sorted(k for k in now if now[k] != before.get(k))


def reviewed_now() -> tuple[list[str], list[str]]:
    ed = edited_stems()
    stems = set(ds_stems())
    conf = sorted(s for s in confirmed() if s in stems and s not in ed)
    return ed, conf


def agg(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        a = out.setdefault(r["code"], {"severity": r["severity"], "n": 0, "images": set()})
        a["n"] += 1
        a["images"].add(r["image"])
    return {c: {"severity": v["severity"], "n": v["n"], "images": len(v["images"])} for c, v in out.items()}


# ---------------------------------------------------------------- trang
@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/state")
def state():
    cfg = cfg_now()
    lr = last_run()
    mode = lr.get("mode")
    stages = {k: (agg(read_csv(REP / f)) if (REP / f).exists() else None)
              for k, f in [("intake", "01_intake.csv"), ("v1", "02_error_list_v1.csv"), ("v2", "04_error_list_v2.csv"), ("v3", "07_error_list_v3.csv")]}
    summary = json.loads((REP / "08_summary.json").read_text(encoding="utf-8")) if (REP / "08_summary.json").exists() else None
    fixes: dict = {}
    for f in read_csv(REP / "03_fixes.csv"):
        fixes[f["action"]] = fixes.get(f["action"], 0) + 1
    rd = raw_dir() / "images"
    n_scanned = sum(p.suffix.lower() in IMG_EXTS for p in rd.iterdir()) if rd.exists() else None
    stems = ds_stems()
    ed, conf = reviewed_now()
    unlabeled = [s for s in stems if not (DS / "labels" / f"{s}.txt").exists()]
    budget, hold = cfg["review"]["budget_images"], cfg["gate"]["warn_ratio_hold"]
    need_more = None
    if summary and summary["verdict"]["decision"] == "Hold":
        v = summary["verdict"]
        need_more = max(0, v["n_warn_images"] - (math.ceil(hold * v["n_images"]) - 1))
    reviewed_n = (len(ed) + len(conf)) if review_session() else (summary or {}).get("reviewed", 0)
    exp_manifest = EXP / EXPORT_NAME / "manifest.json"
    return jsonify(
        has_run=bool(lr), mode=mode, exit=lr.get("exit"), at=lr.get("at"), seconds=lr.get("seconds"), log=lr.get("log", ""),
        raw=lr.get("raw", "data/raw"), can_review=review_session(), finished=mode in ("finish", "demo") and summary is not None,
        summary=summary, stages=stages, fixes=fixes, queue_len=len(read_csv(REP / "05_review_queue.csv")),
        n_scanned=n_scanned, n_intake_removed=len({r["image"] for r in read_csv(REP / "01_intake.csv")}), n_ds=len(stems),
        edited=ed, confirmed=conf, reviewed=reviewed_n, unlabeled=unlabeled, need_more=need_more,
        exported=exp_manifest.exists(), n_exported=json.loads(exp_manifest.read_text(encoding="utf-8"))["n_images"] if exp_manifest.exists() else 0,
        thresholds=dict(phash_dist=cfg["duplicates"]["phash_dist"], conf=cfg["second_opinion"]["conf"]),
        budget=budget, hold_ratio=hold, classes=cfg["classes"],
    )


@app.post("/api/run")
def run_pipeline():
    """mode = check (dừng ở hàng đợi cho người duyệt) | demo (người duyệt mô phỏng, chạy hết)."""
    body = request.get_json(force=True) or {}
    mode = body.get("mode", "check")
    if mode not in ("check", "demo"):
        abort(400, "mode sai")
    raw = (body.get("raw") or last_run().get("raw") or "data/raw").strip()
    rp = Path(raw) if Path(raw).is_absolute() else ROOT / raw
    if not (rp / "images").is_dir():
        abort(400, f"Không thấy thư mục ảnh: {rp / 'images'}")
    cfg = load_config()
    cfg["duplicates"]["phash_dist"] = int(body.get("phash_dist", cfg["duplicates"]["phash_dist"]))
    cfg["second_opinion"]["conf"] = float(body.get("conf", cfg["second_opinion"]["conf"]))
    WEB.mkdir(parents=True, exist_ok=True)
    CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    CONFIRMED.unlink(missing_ok=True)
    args = [sys.executable, "-m", "pipeline.run", "--raw", raw, "--work", str(WORK), "--reports", str(REP),
            "--export-dir", str(EXP), "--export-name", EXPORT_NAME, "--config", str(CFG)]
    if mode == "check":
        args.append("--no-review")
    t0 = time.time()
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONUTF8": "1"}, timeout=1800)
    rec = dict(mode=mode, raw=raw, exit=p.returncode, seconds=round(time.time() - t0, 1), at=time.strftime("%H:%M:%S"),
               log=(p.stdout + ("\n" + p.stderr if p.stderr.strip() else ""))[-20000:])
    save_last(rec)
    ok = (mode == "check" and p.returncode == 30) or (mode == "demo" and p.returncode in (0, 10, 20))
    if not ok:
        abort(500, "Pipeline lỗi (mã " + str(p.returncode) + "): " + rec["log"][-600:])
    return jsonify(rec)


@app.post("/api/finish")
def finish():
    """Kết thúc phiên duyệt: QC vòng cuối + cửa chặn + xuất (như phần sau người duyệt của pipeline.run)."""
    if not review_session():
        abort(400, "Chưa có phiên duyệt: bấm 'Bắt đầu kiểm tra' trước.")
    t0 = time.time()
    cfg = cfg_now()
    ds = Dataset(DS, cfg["classes"])
    ed, conf = reviewed_now()
    reviewed = sorted(set(ed) | set(conf))
    budget = cfg["review"]["budget_images"]
    if len(reviewed) > budget:
        abort(400, f"Đã duyệt {len(reviewed)} ảnh > ngân sách {budget} ảnh.")
    qcodes = {r["image"]: r["codes"] for r in read_csv(REP / "05_review_queue.csv")}
    write_csv(REP / "06_review_log.csv",
              [dict(image=s, result="người sửa nhãn" if s in ed else "người xác nhận nhãn đúng", codes=qcodes.get(s, "")) for s in reviewed],
              ["image", "result", "codes"])
    stems = {p.stem for p in ds.image_files()}
    log = [f"Người duyệt: {len(ed)} ảnh sửa nhãn, {len(conf)} ảnh xác nhận đúng (ngân sách {budget})"]
    issues3 = qc(ds, cfg)
    if any(i.severity == ERROR for i in issues3):
        fixes2 = autofix.fix(ds, issues3, cfg)
        write_csv(REP / "03b_fixes_after_review.csv", fixes2, ["image", "line", "action", "before", "after"])
        log.append(f"{len(fixes2)} sửa tự động sau người duyệt")
        issues3 = qc(ds, cfg)
    verdict = gate.decide(issues3, stems, cfg, set(reviewed))
    write_report(issues3, REP / "07_error_list_v3", "Danh sách lỗi — SAU người duyệt (v3)", len(stems), verdict)
    with open(REP / "class_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_token", "boxes"])
        w.writerows(sorted(class_counts(ds).items()))
    shutil.rmtree(EXP / EXPORT_NAME, ignore_errors=True)
    exported = export.export(ds, EXP, EXPORT_NAME, verdict, cfg) if verdict["decision"] != "Rework" else None
    log.append(f"kết luận: {verdict['decision'].upper()} — {'; '.join(verdict['reasons'])}")
    log.append(f"xuất: {exported}" if exported else "KHÔNG xuất: còn lỗi bắt buộc (Rework)")
    (REP / "08_summary.json").write_text(json.dumps(dict(
        verdict=verdict, reviewed=len(reviewed), reviewed_edited=len(ed), reviewed_confirmed=len(conf), budget=budget,
        seconds=round(time.time() - t0, 1), exported=str(exported) if exported else None), indent=2, ensure_ascii=False), encoding="utf-8")
    lr = last_run()
    save_last(dict(mode="finish", raw=lr.get("raw", "data/raw"), exit=EXIT[verdict["decision"]], seconds=round(time.time() - t0, 1),
                   at=time.strftime("%H:%M:%S"), log=lr.get("log", "") + "\n--- hoàn tất duyệt (web) ---\n" + "\n".join(log)))
    return jsonify(verdict=verdict)


@app.post("/api/confirm/<stem>")
def confirm(stem):
    if not review_session():
        abort(400, "Chưa có phiên duyệt.")
    if stem not in ds_stems():
        abort(404)
    s = confirmed()
    if (request.get_json(force=True) or {}).get("ok", True):
        if not (DS / "labels" / f"{stem}.txt").exists():
            abort(400, "Ảnh chưa có nhãn: hãy vẽ box (hoặc bấm 'Ảnh không có vật nào') rồi lưu.")
        s.add(stem)
    else:
        s.discard(stem)
    CONFIRMED.write_text(json.dumps(sorted(s)), encoding="utf-8")
    ed, conf = reviewed_now()
    return jsonify(ok=True, reviewed=len(ed) + len(conf))


@app.get("/api/images")
def images():
    queue = read_csv(REP / "05_review_queue.csv")
    rank = {r["image"]: (i + 1, float(r["priority"]), r["codes"]) for i, r in enumerate(queue)}
    issues: dict = {}
    for r in read_csv(latest_issue_file()):
        issues.setdefault(r["image"], set()).add(r["code"])
    ed, conf = (set(x) for x in reviewed_now())
    sim = {r["image"] for r in read_csv(REP / "06_review_log.csv")} if last_run().get("mode") == "demo" else set()
    rows = []
    for s in ds_stems():
        rk = rank.get(s)
        rows.append(dict(image=s, rank=rk[0] if rk else None, priority=rk[1] if rk else 0, queue_codes=rk[2] if rk else "",
                         codes=sorted(issues.get(s, [])),
                         status="edited" if s in ed else "confirmed" if s in conf else "sim" if s in sim else None,
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
    ed, conf = reviewed_now()
    q = next(((i + 1, r) for i, r in enumerate(read_csv(REP / "05_review_queue.csv")) if r["image"] == stem), None)
    return jsonify(
        image=stem, in_ds=in_ds, size=size,
        labels=label_lines(DS / "labels" / f"{stem}.txt") if in_ds else None,
        raw_labels=label_lines(raw_dir() / "labels" / f"{stem}.txt"),
        preds=[dict(cls=int(p[0]), conf=float(p[1]), box=[float(v) for v in p[2:6]]) for p in preds().get(stem, [])],
        issues=rows_for(latest_issue_file().name), intake=rows_for("01_intake.csv"),
        fixes=[f for f in read_csv(REP / "03_fixes.csv") if f["image"] == stem],
        queue=dict(rank=q[0], priority=q[1]["priority"], codes=q[1]["codes"]) if q else None,
        status="edited" if stem in ed else "confirmed" if stem in conf else None,
        can_edit=in_ds and review_session(),
    )


@app.get("/img/<stem>")
def image_file(stem):
    src = find_image(DS, stem) or find_image(raw_dir(), stem)
    if src is None:
        abort(404)
    return send_file(src)


@app.post("/api/labels/<stem>")
def save_labels(stem):
    if not review_session():
        abort(400, "Chỉ sửa được nhãn trong phiên duyệt (bấm 'Bắt đầu kiểm tra' trước).")
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
    s = confirmed()
    s.discard(stem)                                   # đã sửa thì tính là "sửa", không còn là "xác nhận"
    CONFIRMED.write_text(json.dumps(sorted(s)), encoding="utf-8")
    ed, conf = reviewed_now()
    return jsonify(ok=True, reviewed=len(ed) + len(conf), budget=cfg_now()["review"]["budget_images"])


@app.get("/api/export.zip")
def export_zip():
    out = EXP / EXPORT_NAME
    man_p = out / "manifest.json"
    if not man_p.exists():
        abort(404, "Chưa có dataset xuất (kết luận Rework hoặc chưa chạy).")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in man["images"]:
            p = DS / "images" / name
            if p.exists():
                z.write(p, f"images/{name}")
        for p in sorted((out / "labels").glob("*.txt")):
            z.write(p, f"labels/{p.name}")
        z.write(out / "data.yaml", "data.yaml")
        z.write(man_p, "manifest.json")
    buf.seek(0)
    tag = man["decision"]["decision"].lower()
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"dataset_{EXPORT_NAME}_{tag}.zip")


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def err(e):
    return jsonify(error=getattr(e, "description", str(e))), getattr(e, "code", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8009))
    print(f"Mở http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
