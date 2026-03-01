import cv2
from ultralytics import YOLO


class YOLODetector:
    """Wrapper around a YOLOv8 model for person detection.

    The detector keeps a video capture internally and can annotate frames
    while also reporting a person count.  It is designed to be used from
    `main.py` so that gameplay and detection can run side-by-side.
    """

    def __init__(self, source: int | str = 0, conf: float = 0.6, iou: float = 0.2):
        # model download happens on first instantiation
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(source)
        self.conf = conf
        self.iou = iou

    def read(self):
        """Read a frame from the camera. Returns (ret, frame)."""
        return self.cap.read()

    def annotate(self, frame):
        """Run the detector on a frame.

        Returns a tuple ``(annotated, person_count, boxes)`` where ``annotated``
        is the image with boxes drawn by the model, ``person_count`` is the
        number of person-class boxes, and ``boxes`` is a list of the box
        coordinates in ``(x1, y1, x2, y2)`` format (all ints).  The boxes list
        can be empty when no one is detected.
        """
        results = self.model(frame, classes=[0], conf=self.conf, iou=self.iou)
        annotated = results[0].plot()
        boxes = []
        for box in results[0].boxes:
            # ultralytics box.xyxy returns tensor; convert to ints
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
        person_count = len(boxes)
        return annotated, person_count, boxes

    def release(self):
        self.cap.release()


# simple convenience entry point for standalone usage
if __name__ == "__main__":
    detector = YOLODetector()
    while True:
        ret, frame = detector.read()
        if not ret:
            break
        annotated, count, boxes = detector.annotate(frame)
        cv2.putText(annotated, "AIRBUD", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(annotated, f"People: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Airbud", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    detector.release()
    cv2.destroyAllWindows()
