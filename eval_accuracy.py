"""Đo độ chính xác model đang chạy trên server, ghi trace log từng ảnh.

Đầu vào: thư mục ảnh + thư mục nhãn YOLO (mỗi ảnh 1 file .txt, dòng "class_id cx cy w h",
class_id theo thứ tự `labels` mà /config trả về). Nhãn đúng của ảnh = nhãn của box lớn nhất.

Chạy:
  python3 eval_accuracy.py --images anh/ --labels nhan/ --out ket-qua/
  python3 eval_accuracy.py --self-test     # kiểm tra phần tính số liệu, không gọi server

Kết quả trong --out: trace.jsonl (mỗi ảnh 1 dòng) và summary.json (độ chính xác theo nhãn,
bảng nhầm lẫn, quét ngưỡng 0,3–0,8, độ chính xác theo ngăn).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_URL = "https://toxingau1000--smart-bin-api.us-east.modal.run"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


class Client:
    """Giữ 1 kết nối HTTPS (keep-alive) cho mọi request, giống cách ESP32 sẽ làm."""

    def __init__(self, base_url: str, token: str):
        u = urlparse(base_url)
        self.conn = http.client.HTTPSConnection(u.netloc, timeout=30)
        self.headers = {"X-Bin-Token": token} if token else {}

    def call(self, method: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
        headers = dict(self.headers, **({"Content-Type": "image/jpeg"} if body else {}))
        self.conn.request(method, path, body=body, headers=headers)
        res = self.conn.getresponse()
        data = res.read()  # phải đọc hết phản hồi thì lần sau mới dùng lại được kết nối
        try:
            return res.status, json.loads(data)
        except ValueError:
            return res.status, {"detail": data[:200].decode(errors="replace")}


def ground_truth(label_file: Path, names: list[str]) -> str | None:
    # Nhãn của ảnh = nhãn của box có diện tích lớn nhất
    best, best_area = None, -1.0
    for line in label_file.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            area = float(parts[3]) * float(parts[4])
            if area > best_area:
                best, best_area = names[int(parts[0])], area
    return best


def summarize(records: list[dict], label_to_bin: dict, default_bin: int, report_thr: float) -> dict:
    scored = [r for r in records if r.get("gt") and r.get("status") == 200]
    sweep = []
    for thr in THRESHOLDS:
        c = Counter()
        for r in scored:
            decided = r["pred"] is not None and r["pred_conf"] >= thr
            c["abstained" if not decided else "correct" if r["pred"] == r["gt"] else "wrong"] += 1
        sweep.append({"threshold": thr, **c})

    per_class, confusion, bin_ok = {}, {}, 0
    for r in scored:
        decided = r["pred"] is not None and r["pred_conf"] >= report_thr
        pred = r["pred"] if decided else "(không chắc)"
        confusion.setdefault(r["gt"], Counter())[pred] += 1
        s = per_class.setdefault(r["gt"], Counter())
        s["n"] += 1
        s["correct"] += pred == r["gt"]
        # Ngăn thật thùng sẽ đổ: không chắc -> ngăn mặc định (Khác)
        pred_bin = label_to_bin.get(pred, default_bin)
        bin_ok += pred_bin == label_to_bin.get(r["gt"], default_bin)
    for s in per_class.values():
        s["accuracy"] = round(s["correct"] / s["n"], 3)
    return {
        "images_scored": len(scored), "report_threshold": report_thr,
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "bin_accuracy": round(bin_ok / len(scored), 3) if scored else None,
        "threshold_sweep": sweep,
    }


def self_test() -> None:
    recs = [
        {"gt": "paper", "pred": "paper", "pred_conf": 0.9, "status": 200},
        {"gt": "paper", "pred": "paper_cup", "pred_conf": 0.6, "status": 200},
        {"gt": "tissue", "pred": None, "pred_conf": 0.0, "status": 200},
        {"gt": "plastic_bottle", "pred": "plastic_container", "pred_conf": 0.4, "status": 200},
    ]
    l2b = {"paper": 2, "paper_cup": 4, "tissue": 4, "plastic_bottle": 1, "plastic_container": 1}
    s = summarize(recs, l2b, 4, 0.5)
    assert s["per_class"]["paper"] == {"n": 2, "correct": 1, "accuracy": 0.5}, s["per_class"]
    # tissue không chắc -> Khác = đúng ngăn; chai nhựa 0,4 < 0,5 -> Khác = sai ngăn
    assert s["bin_accuracy"] == 0.5, s["bin_accuracy"]
    assert s["threshold_sweep"][0] == {"threshold": 0.3, "correct": 1, "wrong": 2, "abstained": 1}
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path, help="thư mục nhãn YOLO; bỏ trống thì chỉ ghi dự đoán")
    ap.add_argument("--out", type=Path, default=Path("eval-out"))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default=os.environ.get("BIN_TOKEN", ""), help="mặc định lấy từ biến BIN_TOKEN")
    ap.add_argument("--threshold", type=float, help="ngưỡng cho bảng theo nhãn; mặc định lấy min_conf của /config")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.images or not args.images.is_dir():
        sys.exit("Cần --images là thư mục ảnh")

    client = Client(args.url, args.token)
    status, cfg = client.call("GET", "/config")
    if status != 200:
        sys.exit(f"Không lấy được /config: HTTP {status} {cfg}")
    names, thr = cfg["labels"], args.threshold or cfg["min_conf"]

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    with open(args.out / "trace.jsonl", "w", encoding="utf-8") as trace:
        for i, img in enumerate(images, 1):
            label_file = args.labels / f"{img.stem}.txt" if args.labels else None
            gt = ground_truth(label_file, names) if label_file and label_file.exists() else None
            t0 = time.perf_counter()
            try:
                status, res = client.call("POST", "/detect", img.read_bytes())
            except (OSError, http.client.HTTPException) as err:  # mất mạng: ghi lỗi rồi mở lại kết nối
                status, res = 0, {"detail": str(err)}
                client = Client(args.url, args.token)
            objects = res.get("objects", [])
            rec = {
                "file": str(img.relative_to(args.images)), "gt": gt, "status": status,
                "pred": objects[0]["label"] if objects else None,
                "pred_conf": objects[0]["conf"] if objects else 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "infer_ms": res.get("infer_ms"), "request_id": res.get("request_id"),
                "model": res.get("model"), "objects": objects, "error": res.get("detail"),
            }
            records.append(rec)
            trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mark = "" if gt is None else (" ✓" if rec["pred"] == gt and rec["pred_conf"] >= thr else " ✗")
            print(f"[{i}/{len(images)}] {rec['file']}: {rec['pred']} {rec['pred_conf']:.2f}"
                  f" (đúng: {gt}){mark} {rec['total_ms']} ms" + (f" LỖI {status}" if status != 200 else ""))

    summary = {"model": cfg["model"], "config_version": cfg["config_version"], "images": len(images),
               "errors": sum(r["status"] != 200 for r in records),
               **summarize(records, cfg["label_to_bin"], cfg["default_bin"], thr)}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐộ chính xác theo nhãn ở ngưỡng {thr} (mốc build plan: mỗi loại ≥ 85%):")
    for name, s in summary["per_class"].items():
        print(f"  {name:18} {s['correct']:>3}/{s['n']:<3} = {s['accuracy']:.0%}" + ("" if s["accuracy"] >= 0.85 else "  < 85%"))
    print(f"Đổ đúng ngăn: {summary['bin_accuracy']}  ·  ghi chi tiết ở {args.out}/")


if __name__ == "__main__":
    main()
