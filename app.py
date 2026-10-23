"""API phát hiện vật thể YOLO26n trên Modal: ESP32-CAM POST ảnh JPEG, nhận lại JSON danh sách vật.

Model phải có sẵn trong Volume: chạy `modal run export_yolo.py` một lần trước khi deploy.
"""
from __future__ import annotations

import modal

MODEL_DIR = "/models"
MAX_BYTES = 500_000  # ảnh ESP32-CAM ở VGA thường nhỏ hơn nhiều
IMG_SIZE = 640  # model được export với imgsz=640
CONF_THRESHOLD = 0.25

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "onnxruntime", "pillow", "numpy", "fastapi[standard]"
)
app = modal.App("phan-loai", image=image)
vol = modal.Volume.from_name("phan-loai-models", create_if_missing=True)

# Chỉ import trong container; máy local không cần cài các thư viện này
with image.imports():
    import io
    import json

    import numpy as np
    import onnxruntime as ort
    from fastapi import HTTPException, Request
    from PIL import Image


@app.cls(
    cpu=(0.25, 0.5),
    memory=(1024, 1536),
    min_containers=0,  # đang phát triển: tự tắt cho rẻ. Tuần demo đổi thành 1 (~$14/tháng)
    scaledown_window=600,  # giữ máy thêm 10 phút sau request cuối
    max_containers=1,  # chặn chi phí phát sinh nếu bị spam
    volumes={MODEL_DIR: vol},
    timeout=60,
)
@modal.concurrent(max_inputs=4)
class Detector:
    @modal.enter()
    def load(self):
        self.session = ort.InferenceSession(f"{MODEL_DIR}/yolo26n.onnx")
        self.input_name = self.session.get_inputs()[0].name
        with open(f"{MODEL_DIR}/yolo26n-names.json") as f:
            self.names = json.load(f)

    def letterbox(self, img: Image.Image):
        # Thu nhỏ giữ nguyên tỉ lệ rồi pad viền xám cho đủ 640x640, để vật không bị méo
        w, h = img.size
        r = min(IMG_SIZE / w, IMG_SIZE / h)
        nw, nh = round(w * r), round(h * r)
        pad_x, pad_y = (IMG_SIZE - nw) / 2, (IMG_SIZE - nh) / 2
        canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (round(pad_x - 0.1), round(pad_y - 0.1)))
        # YOLO chỉ chia 255, KHÔNG trừ mean/std như MobileNet
        x = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        return x, r, pad_x, pad_y

    @modal.fastapi_endpoint(method="POST")
    async def detect(self, request: Request):
        body = await request.body()
        if not body or len(body) > MAX_BYTES:
            raise HTTPException(400, "Ảnh rỗng hoặc quá lớn")
        try:
            img = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Không đọc được ảnh JPEG")

        x, r, pad_x, pad_y = self.letterbox(img)
        # Đầu ra NMS-free: [1, 300, 6], mỗi dòng = x1, y1, x2, y2, conf, class
        rows = self.session.run(None, {self.input_name: x})[0][0]
        w, h = img.size
        objects = []
        for x1, y1, x2, y2, conf, cls in rows[rows[:, 4] >= CONF_THRESHOLD]:
            # Đổi toạ độ từ ảnh 640x640 đã pad về ảnh gốc
            box = [(x1 - pad_x) / r, (y1 - pad_y) / r, (x2 - pad_x) / r, (y2 - pad_y) / r]
            box = [round(min(max(v, 0), lim)) for v, lim in zip(box, (w, h, w, h))]
            objects.append({"label": self.names[int(cls)], "conf": round(float(conf), 3), "box": box})
        return {"count": len(objects), "objects": objects, "image_size": [w, h]}
