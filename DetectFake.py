"""GUI iteration without Oak-D: Mac webcam + scripted fake detections.

Uses the same `render.draw` as Detect.py, so visual tweaks in render.py
apply to both the real and fake pipelines.
"""
import math
import time

import cv2

from render import Detection, FPSCounter, draw


def fake_detections(t: float):
    def box(cx, cy, size):
        return cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2

    x1, y1, x2, y2 = box(0.3 + 0.1 * math.sin(t), 0.4 + 0.1 * math.cos(t), 0.15)
    a = Detection(x1, y1, x2, y2, "airplane", 0.87)

    x1, y1, x2, y2 = box(
        0.7 + 0.15 * math.cos(t * 1.3),
        0.6 + 0.1 * math.sin(t * 1.3),
        0.10,
    )
    b = Detection(x1, y1, x2, y2, "bird", 0.62)

    return [a, b]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Is another app using it?")
        return

    print("Fake detection running. Press 'q' to quit.")
    fps_counter = FPSCounter()
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            dets = fake_detections(time.monotonic() - start)
            fps = fps_counter.tick()
            draw(frame, dets, fps)

            cv2.imshow("Drone Detection (Fake - Webcam)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
