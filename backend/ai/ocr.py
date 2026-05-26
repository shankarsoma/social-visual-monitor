import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

def extract_text_from_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes))

    image_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    text = pytesseract.image_to_string(image_cv)

    return text