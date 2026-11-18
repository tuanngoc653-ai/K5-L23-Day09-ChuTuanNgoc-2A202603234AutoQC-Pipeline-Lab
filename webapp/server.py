"""AutoQC web: kiểm nhãn cho dataset YOLO bất kỳ (chạy CỤC BỘ, không phải phần chấm điểm của lab).

    pip install -r webapp/requirements.txt
    python webapp/server.py            # mở http://127.0.0.1:8009
    # (tuỳ chọn) Python có ultralytics để chạy model "ý kiến thứ hai":
    #   set AUTOQC_PREDICT_PYTHON=C:\\path\\to\\venv\\Scripts\\python.exe

Luồng 3 bước:
  1. Kiểm tra   : chọn dataset (lớp lấy từ data.yaml), tuỳ chọn chạy model, rồi pipeline tới hàng đợi và dừng
  2. Duyệt ảnh  : người xem từng ảnh trong hàng đợi; sửa nhãn hoặc xác nhận "nhãn đúng"
  3. Kết quả    : QC vòng cuối + cửa chặn + xuất (đã duyệt = ảnh đã sửa ∪ ảnh xác nhận đúng)

Web dùng lại các check trong pipeline/ với một cấu hình sinh riêng cho từng dataset (work/web/pipeline.yaml).
Không bao giờ ghi vào thư mục dataset gốc: mọi thứ nằm trong work/web/.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
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
LAST, CONFIRMED, PROJECT = WEB / "last_run.json", WEB / "confirmed.json", WEB / "project.json"
STAGE, PREDS, PREDS_META, PROGRESS = WEB / "raw", WEB / "preds.json", WEB / "preds.meta.json", WEB / "progress.json"
DS = WORK / "ds"
EXPORT_NAME = "clean_v1"
LAB_RAW = (ROOT / "data" / "raw").resolve()
PREDICT_PY = Path(__file__).parent / "predict.py"

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
_preds_cache: dict = {}
JOB: dict = {"running": False}
JOB_LOCK = threading.Lock()


# ================================================================ tiện ích chung
def rjson(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def wjson(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def resolve(path: str) -> Path:
    p = Path(path.strip().strip('"'))
    return (p if p.is_absolute() else ROOT / p).resolve()


def cfg_now() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else load_config()


def project() -> dict:
    return rjson(PROJECT, {}) or {}


def last_run() -> dict:
    return rjson(LAST, {}) or {}


def read_csv(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def confirmed() -> set[str]:
    return set(rjson(CONFIRMED, []))


def latest_issue_file() -> Path:
    return REP / "07_error_list_v3.csv" if (REP / "07_error_list_v3.csv").exists() else REP / "04_error_list_v2.csv"


def preds() -> dict:
    path = cfg_now()["models"]["checker"]["cache"]
    p = resolve(path)
    key = (str(p), p.stat().st_mtime if p.exists() else 0)
    if key not in _preds_cache:
        _preds_cache.clear()
        _preds_cache[key] = load_cache(str(p))
    return _preds_cache[key]


def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS) if d.is_dir() else []


def find_image(folder: Path, stem: str) -> Path | None:
    return next((p for p in images_in(folder / "images") if p.stem == stem), None)


def raw_dir() -> Path:
    return Path(project().get("raw") or LAB_RAW)


def ds_stems() -> list[str]:
    return [p.stem for p in images_in(DS / "images")]


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
    return ed, sorted(s for s in confirmed() if s in stems and s not in ed)


def agg(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        a = out.setdefault(r["code"], {"severity": r["severity"], "n": 0, "images": set()})
        a["n"] += 1
        a["images"].add(r["image"])
    return {c: {"severity": v["severity"], "n": v["n"], "images": len(v["images"])} for c, v in out.items()}


def yaml_classes(y: Path) -> list[str]:
    try:
        d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    n = d.get("names")
    if isinstance(n, list):
        return [str(x) for x in n]
    if isinstance(n, dict):
        return [str(n[k]) for k in sorted(n, key=lambda k: int(k))]
    return []


def default_python() -> str:
    return os.environ.get("AUTOQC_PREDICT_PYTHON") or sys.executable


# ================================================================ đọc thư mục dataset
def inspect_folder(src: Path) -> dict:
    """Nhận dạng cấu trúc: 'flat' (images/ + labels/) hoặc 'split' (images/<split>/ + labels/<split>/)."""
    if not src.is_dir():
        raise ValueError(f"Không thấy thư mục: {src}")
    ymls = [src / "data.yaml"] + sorted(src.glob("*.yaml")) + sorted(src.glob("*.yml"))
    yml = next((y for y in ymls if y.exists() and yaml_classes(y)), None)
    info = dict(path=str(src), yaml=str(yml) if yml else None, classes=yaml_classes(yml) if yml else [],
                lab=src.resolve() == LAB_RAW, layout=None, splits=[])
    flat = images_in(src / "images")
    if flat:
        info.update(layout="flat", n_images=len(flat), n_labels=len(list((src / "labels").glob("*.txt"))) if (src / "labels").is_dir() else 0)
    elif (src / "images").is_dir():
        for d in sorted(p for p in (src / "images").iterdir() if p.is_dir()):
            imgs = images_in(d)
            if imgs:
                ld = src / "labels" / d.name
                info["splits"].append(dict(name=d.name, n_images=len(imgs), n_labels=len(list(ld.glob("*.txt"))) if ld.is_dir() else 0))
        if info["splits"]:
            info["layout"] = "split"
    if not info["layout"]:
        raise ValueError("Không thấy ảnh. Cần images/ (và labels/) ngay trong thư mục, hoặc images/<split>/ + labels/<split>/.")
    if info["lab"]:
        info["classes"] = load_config()["classes"]          # data.yaml của lô lab cố ý sai tên lớp
    return info


def stage_raw(p: dict) -> Path:
    """Thư mục 'raw' mà pipeline đọc: chính thư mục gốc (kiểu flat), hoặc một bản liên kết cứng (kiểu split)."""
    src = Path(p["source"])
    if p["layout"] == "flat":
        return src
    shutil.rmtree(STAGE, ignore_errors=True)
    (STAGE / "images").mkdir(parents=True)
    (STAGE / "labels").mkdir()

    def link(a: Path, b: Path):
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)
    for img in images_in(src / "images" / p["split"]):
        link(img, STAGE / "images" / img.name)
    for lb in (src / "labels" / p["split"]).glob("*.txt") if (src / "labels" / p["split"]).is_dir() else []:
        link(lb, STAGE / "labels" / lb.name)
    if p.get("yaml"):
        shutil.copy2(p["yaml"], STAGE / "data.yaml")
    return STAGE


def build_cfg(p: dict) -> dict:
    cfg = load_config()                                       # quy tắc mặc định của lab làm khung
    cfg["classes"] = p["classes"]
    cfg["class_aliases"] = {}
    cfg["intake"]["min_side_px"] = int(p.get("min_side", 250))
    cfg["intake"]["reference_dirs"] = [str(resolve(d)) for d in p.get("leak_dirs", []) if d.strip()]
    cfg["duplicates"]["phash_dist"] = int(p.get("phash_dist", 12))
    cfg["second_opinion"]["conf"] = float(p.get("conf", 0.6))
    cfg["review"]["budget_images"] = int(p.get("budget", 80))
    cfg["gate"]["warn_ratio_hold"] = float(p.get("hold_ratio", 0.10))
    mode = p.get("model_mode", "off")
    if mode == "cache":
        cfg["models"]["checker"]["cache"] = str(resolve(load_config()["models"]["checker"]["cache"]))
    else:
        cfg["models"]["checker"]["cache"] = str(PREDS)
    if mode == "off":
        cfg["second_opinion"]["iou_missing"] = 0.0            # không có dự đoán: tắt cờ "box thừa" (mọi box sẽ bị cờ)
    return cfg


# ================================================================ chạy nền
def job_log(msg: str):
    with JOB_LOCK:
        JOB.setdefault("log", []).append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def run_job(mode: str, p: dict):
    try:
        JOB.update(stage="prepare", progress=None)
        raw = stage_raw(p)
        p["raw"] = str(raw)
        wjson(PROJECT, p)
        n_img = len(images_in(raw / "images"))
        job_log(f"Dataset: {n_img} ảnh · {len(p['classes'])} lớp: {', '.join(p['classes'])}")

        mm = p.get("model_mode", "off")
        if mm == "yolo":
            sig = hashlib.sha1(json.dumps([p["source"], p.get("split"), p["classes"], p["weights"], p.get("pred_conf", 0.1),
                                            sorted(x.name for x in images_in(raw / "images"))]).encode()).hexdigest()
            meta = rjson(PREDS_META, {}) or {}
            if PREDS.exists() and meta.get("sig") == sig:
                job_log("Dùng lại dự đoán model đã chạy trước (cùng dataset, cùng model).")
            else:
                JOB.update(stage="predict", progress=dict(done=0, total=n_img))
                job_log(f"Chạy model {p['weights']} lên {n_img} ảnh…")
                PROGRESS.unlink(missing_ok=True)
                cmd = [p.get("python") or default_python(), str(PREDICT_PY), "--weights", p["weights"], "--images", str(raw / "images"),
                       "--classes", ",".join(p["classes"]), "--out", str(PREDS), "--conf", str(p.get("pred_conf", 0.1)), "--progress", str(PROGRESS)]
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        env={**os.environ, "PYTHONUTF8": "1"})
                out = []
                while proc.poll() is None:
                    pr = rjson(PROGRESS)
                    if pr:
                        JOB["progress"] = pr
                    time.sleep(0.5)
                out = (proc.stdout.read() or "").strip()
                if proc.returncode != 0:
                    raise RuntimeError("Chạy model lỗi:\n" + out[-1500:])
                last = out.splitlines()[-1] if out else "{}"
                try:
                    r = json.loads(last)
                    job_log(f"Model xong trong {r.get('seconds')}s. Lớp khớp: {', '.join(r.get('matched', [])) or '—'}"
                            + (f" · không khớp: {', '.join(r['unmatched'])}" if r.get("unmatched") else ""))
                except Exception:
                    job_log("Model xong.")
                wjson(PREDS_META, dict(sig=sig, at=time.strftime("%Y-%m-%d %H:%M:%S")))
        elif mm == "off":
            wjson(PREDS, {x.stem: [] for x in images_in(raw / "images")})
            PREDS_META.unlink(missing_ok=True)
            job_log("Ý kiến thứ hai: TẮT (chỉ kiểm quy tắc).")
        else:
            job_log("Ý kiến thứ hai: dùng dự đoán có sẵn của bài lab.")

        cfg = build_cfg(p)
        CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        CONFIRMED.unlink(missing_ok=True)
        JOB.update(stage="pipeline", progress=None)
        job_log("Chạy pipeline: cửa vào → QC → máy tự sửa → hàng đợi…")
        args = [sys.executable, "-m", "pipeline.run", "--raw", str(raw), "--work", str(WORK), "--reports", str(REP),
                "--export-dir", str(EXP), "--export-name", EXPORT_NAME, "--config", str(CFG)]
        if mode == "check":
            args.append("--no-review")
        t0 = time.time()
        pr = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONUTF8": "1"})
        log = pr.stdout + ("\n" + pr.stderr if pr.stderr.strip() else "")
        for line in pr.stdout.splitlines():
            msg = line.split("] ", 1)[-1]
            if msg.startswith("TẠM DỪNG"):                    # hướng dẫn dòng lệnh, không dành cho người dùng web
                msg = "Tạm dừng: mời người duyệt ở Bước 2."
            job_log("  " + msg)
        wjson(LAST, dict(mode=mode, raw=str(raw), exit=pr.returncode, seconds=round(time.time() - t0, 1), at=time.strftime("%H:%M:%S"), log=log[-20000:]))
        ok = (mode == "check" and pr.returncode == 30) or (mode == "demo" and pr.returncode in (0, 10, 20))
        if not ok:
            raise RuntimeError(f"Pipeline lỗi (mã {pr.returncode}):\n" + log[-1500:])
        JOB.update(stage="done", error=None)
        job_log("Xong.")
    except Exception as e:                                    # báo lỗi lên giao diện, không làm chết server
        JOB.update(stage="error", error=str(e))
        job_log("LỖI: " + str(e)[:400])
    finally:
        JOB["running"] = False


# ================================================================ API
@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/inspect")
def api_inspect():
    path = (request.get_json(force=True) or {}).get("path", "")
    try:
        return jsonify(inspect_folder(resolve(path)))
    except ValueError as e:
        abort(400, str(e))


@app.post("/api/model_check")
def api_model_check():
    b = request.get_json(force=True) or {}
    py = b.get("python") or default_python()
    classes = [c.strip() for c in b.get("classes", []) if c.strip()]
    if not classes:
        abort(400, "Chưa có danh sách lớp.")
    try:
        r = subprocess.run([py, str(PREDICT_PY), "--weights", b.get("weights", "yolo26l.pt"), "--classes", ",".join(classes), "--inspect"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env={**os.environ, "PYTHONUTF8": "1"})
    except FileNotFoundError:
        abort(400, f"Không chạy được Python: {py}")
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()
        tail = msg[-1] if msg else "lỗi không rõ"
        if "No module named 'ultralytics'" in (r.stderr or ""):
            tail = "Python này chưa cài ultralytics (pip install ultralytics), hoặc chọn Python khác."
        abort(400, tail)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return jsonify(matched=info["matched"], unmatched=info["unmatched"], n_model_classes=len(info["names"]))


@app.post("/api/start")
def api_start():
    b = request.get_json(force=True) or {}
    mode = b.get("mode", "check")
    if mode not in ("check", "demo"):
        abort(400, "mode sai")
    if JOB.get("running"):
        abort(400, "Đang có một lần chạy khác.")
    try:
        info = inspect_folder(resolve(b.get("path", "")))
    except ValueError as e:
        abort(400, str(e))
    classes = [c.strip() for c in b.get("classes", []) if c.strip()]
    if not classes:
        abort(400, "Cần ít nhất một lớp (thường lấy từ data.yaml).")
    if len(set(classes)) != len(classes):
        abort(400, "Tên lớp bị trùng.")
    if info["layout"] == "split" and b.get("split") not in [s["name"] for s in info["splits"]]:
        abort(400, "Hãy chọn phần dữ liệu (split) cần kiểm.")
    mm = b.get("model_mode", "off")
    if mm == "cache" and not info["lab"]:
        abort(400, "Dự đoán có sẵn chỉ dành cho dữ liệu bài lab.")
    if mode == "demo" and not info["lab"]:
        abort(400, "Chế độ demo (người duyệt mô phỏng) chỉ có với dữ liệu bài lab.")
    p = dict(source=info["path"], layout=info["layout"], split=b.get("split") if info["layout"] == "split" else None, yaml=info["yaml"],
             classes=classes, lab=info["lab"], leak_dirs=[d for d in b.get("leak_dirs", []) if d.strip()],
             model_mode=mm, weights=b.get("weights") or "yolo26l.pt", python=b.get("python") or default_python(), pred_conf=0.1,
             phash_dist=int(b.get("phash_dist", 12)), conf=float(b.get("conf", 0.6)), budget=int(b.get("budget", 80)),
             hold_ratio=float(b.get("hold_ratio", 0.10)), min_side=int(b.get("min_side", 250)))
    with JOB_LOCK:
        JOB.clear()
        JOB.update(running=True, mode=mode, stage="prepare", log=[], error=None, started=time.time())
    threading.Thread(target=run_job, args=(mode, p), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/job")
def api_job():
    with JOB_LOCK:
        j = {k: v for k, v in JOB.items()}
        j["log"] = list(JOB.get("log", []))[-60:]
    return jsonify(j)


@app.get("/api/state")
def state():
    cfg = cfg_now()
    lr = last_run()
    p = project()
    mode = lr.get("mode")
    stages = {k: (agg(read_csv(REP / f)) if (REP / f).exists() else None)
              for k, f in [("intake", "01_intake.csv"), ("v1", "02_error_list_v1.csv"), ("v2", "04_error_list_v2.csv"), ("v3", "07_error_list_v3.csv")]}
    summary = rjson(REP / "08_summary.json")
    fixes: dict = {}
    for f in read_csv(REP / "03_fixes.csv"):
        fixes[f["action"]] = fixes.get(f["action"], 0) + 1
    stems = ds_stems()
    ed, conf = reviewed_now()
    budget, hold = cfg["review"]["budget_images"], cfg["gate"]["warn_ratio_hold"]
    need_more = None
    if summary and summary["verdict"]["decision"] == "Hold":
        v = summary["verdict"]
        need_more = max(0, v["n_warn_images"] - (math.ceil(hold * v["n_images"]) - 1))
    exp_manifest = EXP / EXPORT_NAME / "manifest.json"
    return jsonify(
        has_run=bool(lr), mode=mode, exit=lr.get("exit"), at=lr.get("at"), log=lr.get("log", ""), project=p,
        can_review=review_session(), finished=mode in ("finish", "demo") and summary is not None,
        summary=summary, stages=stages, fixes=fixes, queue_len=len(read_csv(REP / "05_review_queue.csv")),
        n_scanned=len(images_in(raw_dir() / "images")) if p else None,
        n_intake_removed=len({r["image"] for r in read_csv(REP / "01_intake.csv")}), n_ds=len(stems),
        edited=ed, confirmed=conf, reviewed=(len(ed) + len(conf)) if review_session() else (summary or {}).get("reviewed", 0),
        unlabeled=[s for s in stems if not (DS / "labels" / f"{s}.txt").exists()], need_more=need_more,
        exported=exp_manifest.exists(), n_exported=(rjson(exp_manifest) or {}).get("n_images", 0),
        budget=budget, hold_ratio=hold, classes=cfg["classes"], second_opinion=p.get("model_mode", "off") != "off",
        thresholds=dict(phash_dist=cfg["duplicates"]["phash_dist"], conf=cfg["second_opinion"]["conf"]),
        default_python=default_python(), job_running=bool(JOB.get("running")),
    )


@app.post("/api/finish")
def finish():
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
    wjson(REP / "08_summary.json", dict(verdict=verdict, reviewed=len(reviewed), reviewed_edited=len(ed), reviewed_confirmed=len(conf),
                                        budget=budget, seconds=round(time.time() - t0, 1), exported=str(exported) if exported else None))
    lr = last_run()
    wjson(LAST, dict(mode="finish", raw=lr.get("raw"), exit=EXIT[verdict["decision"]], seconds=round(time.time() - t0, 1),
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
    wjson(CONFIRMED, sorted(s))
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
    labels = {p.stem for p in (DS / "labels").glob("*.txt")} if (DS / "labels").is_dir() else set()
    rows = []
    for s in ds_stems():
        rk = rank.get(s)
        rows.append(dict(image=s, rank=rk[0] if rk else None, priority=rk[1] if rk else 0, queue_codes=rk[2] if rk else "",
                         codes=sorted(issues.get(s, [])),
                         status="edited" if s in ed else "confirmed" if s in conf else "sim" if s in sim else None, has_label=s in labels))
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
    s.discard(stem)
    wjson(CONFIRMED, sorted(s))
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
