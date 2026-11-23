# -*- coding: utf-8 -*-
"""English License Plate Detection and Character Recognition."""

import platform
import cv2
import imutils
import pytesseract
from ultralytics import YOLO

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_plate_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plate_text = pytesseract.image_to_string(
        thresh, lang="eng", config=r"--oem 3 --psm 6"
    )
    return plate_text.strip()


def main():
    model = YOLO("models/YOLOv8m_Iran_license_plate_detection.pt")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        results = model.predict(frame, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Plate",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                plate_img = frame[y1:y2, x1:x2]
                plate_text = extract_plate_text(plate_img)

                if plate_text:
                    cv2.putText(
                        frame,
                        plate_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        cv2.imshow("License Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()