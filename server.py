"""Flask dashboard: MJPEG feed + live label counts.

Run:
  python server.py              # fake source (default)
  python server.py --source oak # Oak-D camera
"""
import argparse
import math
import os
import threading
import time
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from render import Detection, FPSCounter, draw

try:
    import depthai as dai
except ImportError:
    dai = None


class FakeSkySource:
    """Synthetic sky background + two moving fake drones. No camera needed."""

    def __init__(self, w: int = 960, h: int = 540):
        self.w = w
        self.h = h
        self.start = time.monotonic()
        top = np.array([180, 130, 70], dtype=np.float32)
        bot = np.array([70, 40, 20], dtype=np.float32)
        grad = np.linspace(top, bot, h).astype(np.uint8)
        self._bg = np.broadcast_to(grad[:, None, :], (h, w, 3)).copy()

    def next(self):
        t = time.monotonic() - self.start
        frame = self._bg.copy()

        def box(cx, cy, size):
            return cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2

        x1, y1, x2, y2 = box(0.30 + 0.12 * math.sin(t), 0.40 + 0.10 * math.cos(t), 0.14)
        a = Detection(x1, y1, x2, y2, "airplane", 0.87)

        x1, y1, x2, y2 = box(
            0.70 + 0.15 * math.cos(t * 1.3),
            0.55 + 0.10 * math.sin(t * 1.3),
            0.09,
        )
        b = Detection(x1, y1, x2, y2, "bird", 0.62)

        c = None
        if math.sin(t * 0.5) > 0.3:
            x1, y1, x2, y2 = box(0.5 + 0.05 * math.sin(t * 2), 0.25, 0.10)
            c = Detection(x1, y1, x2, y2, "airplane", 0.71)

        dets = [a, b] + ([c] if c else [])
        return frame, dets


class OakDSource:
    """Oak-D S2 running YOLOv6-nano on the Myriad X VPU."""

    MODEL_SLUG = "yolov6-nano"
    CONF_THRESHOLD = 0.25
    FPS_TARGET = 30
    WANT = {
        "airplane", "bird", "surfboard", "cell phone",
        "mouse", "snowboard", "skateboard", "remote",
    }

    def __init__(self):
        if dai is None:
            raise RuntimeError("depthai is not installed. `pip install depthai`")

        os.environ.setdefault(
            "DEPTHAI_ZOO_CACHE_PATH",
            os.path.expanduser("~/.depthai_cache"),
        )
        os.makedirs(os.environ["DEPTHAI_ZOO_CACHE_PATH"], exist_ok=True)

        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        model_desc = dai.NNModelDescription(self.MODEL_SLUG, platform="RVC2")
        nn = self.pipeline.create(dai.node.DetectionNetwork).build(
            cam, model_desc, fps=self.FPS_TARGET
        )
        nn.setConfidenceThreshold(self.CONF_THRESHOLD)
        self.class_names = nn.getClasses() or []
        self.det_queue = nn.out.createOutputQueue()
        self.frame_queue = nn.passthrough.createOutputQueue()
        self.pipeline.start()

    def next(self):
        img = self.frame_queue.get()
        dets = self.det_queue.get()
        if img is None or dets is None:
            return None, []

        frame = img.getCvFrame()
        detections = []
        for det in dets.detections:
            cls_id = int(det.label)
            cls_name = (
                self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            ).lower()
            if cls_name not in self.WANT:
                continue
            detections.append(Detection(
                xmin=det.xmin, ymin=det.ymin,
                xmax=det.xmax, ymax=det.ymax,
                label=cls_name, confidence=det.confidence,
            ))
        return frame, detections


app = Flask(__name__)

_state = {
    "frame_jpg": None,
    "labels": {},
    "classifications": {},
}
_lock = threading.Lock()


def _run_loop(source):
    fps = FPSCounter()
    while True:
        frame, dets = source.next()
        if frame is None:
            time.sleep(0.01)
            continue
        draw(frame, dets, fps.tick())
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        labels = dict(Counter(d.label for d in dets))
        classifications: dict[str, float] = {}
        for d in dets:
            if d.confidence > classifications.get(d.label, 0.0):
                classifications[d.label] = float(d.confidence)
        with _lock:
            _state["frame_jpg"] = buf.tobytes()
            _state["labels"] = labels
            _state["classifications"] = classifications
        time.sleep(1 / 30)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with _lock:
                buf = _state["frame_jpg"]
            if buf is None:
                time.sleep(0.01)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            )
            time.sleep(1 / 30)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections():
    with _lock:
        return jsonify({
            "labels": _state["labels"],
            "classifications": _state["classifications"],
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["fake", "oak"], default="fake")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.source == "oak":
        print("Initializing Oak-D...")
        source = OakDSource()
    else:
        source = FakeSkySource()

    threading.Thread(target=_run_loop, args=(source,), daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
