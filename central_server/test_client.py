import io
import json

import cv2
import requests


def send_image_and_data(url, image_path):
    # Prepare the data
    payload = {"data": json.dumps({"coord": "19.0000, 18.00000", "bbox": [10, 10], "run": 3.0})}

    # Convert the image to a byte array
    image = cv2.imread(image_path)
    is_success, buffer = cv2.imencode(".jpg", image)
    byte_array = buffer.tobytes()

    # Create a file-like object from the byte array
    image_file = io.BytesIO(byte_array)

    # Prepare the files
    files = {
        "file": ("image.jpg", image_file, "image/jpeg"),
    }

    # Send the POST request
    response = requests.post(url, data=payload, files=files)

    # Return the response
    return response


if __name__ == "__main__":
    url = "http://localhost:8001/sentinel/api/interest_object"
    image_path = "/home/yibbtstll/Pictures/Rafael/test_Face.jpg"
    response = send_image_and_data(url, image_path)
    print(response.text)
