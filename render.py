"""Shared drawing code for drone detection GUI. Edit here to restyle."""
from dataclasses import dataclass
import time
import cv2


@dataclass
class Detection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str
    confidence: float


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self._last = time.monotonic()
        self._ema = 0.0
        self._smoothing = smoothing

    def tick(self) -> float:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return self._ema
        inst = 1.0 / dt
        s = self._smoothing
        self._ema = inst if self._ema == 0 else (s * self._ema + (1 - s) * inst)
        return self._ema


BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
HUD_COLOR = (0, 255, 255)


def draw(frame, detections, fps):
    h, w = frame.shape[:2]
    for det in detections:
        x1 = int(det.xmin * w)
        y1 = int(det.ymin * h)
        x2 = int(det.xmax * w)
        y2 = int(det.ymax * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            BOX_COLOR,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        HUD_COLOR,
        2,
    )
