# live_person_detection_clean.py
from ultralytics import YOLO
import cv2
import time

# 1️⃣ Load your trained YOLOv8 model
model = YOLO("runs/detect/person_small_run/weights/best.pt")  # path to your trained model

# 2️⃣ Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

# 3️⃣ Track FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 4️⃣ Run YOLO detection (verbose=False disables CMD spam)
    results = model(frame, conf=0.5, verbose=False)

    # 5️⃣ Annotate frame with bounding boxes
    annotated_frame = results[0].plot()

    # 6️⃣ Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 7️⃣ Display FPS on frame
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # 8️⃣ Show webcam frame
    cv2.imshow("YOLOv8 Person Detection", annotated_frame)

    # 9️⃣ Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# 🔟 Release camera and close windows
cap.release()
cv2.destroyAllWindows()
