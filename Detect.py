"""
Live Drone Detection using YOLO26
Detects drones from a live camera feed and draws bounding boxes around them.
"""
#gello
import cv2
from ultralytics import YOLO

# ---------- Configuration ----------
MODEL_PATH = "yolo26n.pt"        # Path to your YOLO26 model weights
CAMERA_INDEX = 1                  # 0 for default webcam; change if using another camera
CONF_THRESHOLD = 0.35             # Minimum confidence to display a detection
IOU_THRESHOLD = 0.45              # IoU threshold for non-max suppression
IMG_SIZE = 640                    # Inference image size
DEVICE = "mps"                     # 0 for GPU (CUDA), "cpu" for CPU, "mps" for Apple Silicon

# Class names that represent a drone. YOLO26n is pretrained on COCO, which has
# "airplane" (class 4) as the closest analog. If you later fine-tune on a
# drone-specific dataset, replace this with your drone class name(s).
DRONE_CLASS_NAMES = {"airplane", "drone", "uav", "quadcopter"}


def main():
    # Load the YOLO26 model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names  # dict: {class_id: class_name}

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # Optional: set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Allow camera to warm up
    import time
    time.sleep(1.0)

    print("Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break

        # Run inference on the current frame
        # stream=False returns a list of Results; we take the first (and only) one
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, str(cls_id)).lower()
                
                if cls_name != "phone" or cls_name != "airplane":
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label with confidence
                label = f"DRONE ({cls_name}) {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        # Show FPS in the corner
        fps = 1000.0 / max(results.speed.get("inference", 1e-3), 1e-3)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Drone Detection (YOLO26)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()