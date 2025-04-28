import torch
from ultralytics import YOLO
import cv2
import time

# Carica il modello YOLO
model = YOLO("../yolo11n.pt")
print(model.names)

cap = cv2.VideoCapture("../Video/video1.mp4")

# Ottieni le dimensioni originali del video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crea una finestra fullscreen o ridimensionabile
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", original_width, original_height)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Esegui la detection
    with torch.no_grad():
        results = model(frame, imgsz=320)
    annotated_frame = results[0].plot()

    # Calcola FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostra il frame ridimensionato
    cv2.imshow("Object Detection", annotated_frame)

    # Opzioni di controllo
    key = cv2.waitKey(1)
    if key == ord('q'):  # Esci
        break
    elif key == ord('f'):  # Toggle fullscreen
        cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN,
                              not cv2.getWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN))

cap.release()
cv2.destroyAllWindows()