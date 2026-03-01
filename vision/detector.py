import cv2
from ultralytics import YOLO

# Load YOLO model (downloads automatically on first run)
model = YOLO("yolov8n.pt")  # 'n' = nano, smallest/fastest

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    # class 0 = person only
    # conf - reducing false human detections; raise
    # iou - suppress overlap of boxes; lower
    results = model(frame, classes=[0], conf=0.6, iou=0.2)

    # Draw results on frame
    annotated_frame = results[0].plot()

    person_count = len(results[0].boxes)
    # TODO: enforce person count depending on game mode
    cv2.putText(
        annotated_frame,
        "AIRBUD",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Airbud", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
