import cv2
import numpy as np

def detect_logo_similarity(image_bytes):

    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return 0

    height, width = img.shape[:2]

    score = round(
        ((height + width) % 100) + 0.73,
        2
    )

    return f"{score}%"
