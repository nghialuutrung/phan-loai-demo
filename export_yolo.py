"""Chạy 1 lần: export YOLO26n sang ONNX trên Modal và lưu vào Volume.

Chạy: modal run export_yolo.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV (ultralytics cần) đòi 2 thư viện hệ thống này
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("ultralytics", "onnx", "onnxslim", "onnxruntime")
)
app = modal.App("export-yolo26n", image=image)
vol = modal.Volume.from_name("smart-bin-models", create_if_missing=True)


@app.function(volumes={"/models": vol}, timeout=600, cpu=2.0, memory=4096)
def export():
    import json
    import shutil
    import urllib.request

    import onnxruntime as ort
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    onnx_path = model.export(format="onnx", nms=False, imgsz=640)  # nms=False: model tự lọc khung

    sess = ort.InferenceSession(onnx_path)
    meta = sess.get_modelmeta().custom_metadata_map
    info = {
        "inputs": [(i.name, i.shape) for i in sess.get_inputs()],
        "outputs": [(o.name, o.shape) for o in sess.get_outputs()],
        "meta_keys": sorted(meta),
        "stride": meta.get("stride"),
        "imgsz": meta.get("imgsz"),
        "end2end": meta.get("end2end"),
    }

    # Kết quả tham chiếu của chính Ultralytics, để so với code tự viết trên server
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "/models/bus.jpg")
    result = model.predict("/models/bus.jpg", conf=0.25, verbose=False)[0]
    info["reference"] = [
        {"label": model.names[int(b.cls)], "conf": round(float(b.conf), 3),
         "box": [round(v) for v in b.xyxy[0].tolist()]}
        for b in result.boxes
    ]

    shutil.copy(onnx_path, "/models/yolo26n.onnx")
    with open("/models/yolo26n-names.json", "w") as f:
        json.dump([model.names[i] for i in range(len(model.names))], f)
    vol.commit()
    return info


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(export.remote(), indent=2, ensure_ascii=False))
