from datetime import datetime
import os

def save_screenshot(name):

    folder = "evidence"

    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{folder}/{datetime.now().timestamp()}_{name}.png"

    return filename
