import io
import json
from pathlib import Path
from time import sleep

import cv2
import numpy as np
import requests
from rich.progress import track

SEQUENCE_PATH = "/home/yibbtstll/Projects/sentinel/data/VisDrone/VisDrone2019-VID-val/sequences/uav0000117_02622_v/"
CENTRAL_SERVER_URL = "http://localhost:8001/sentinel/api/interest_object"

session = requests.Session()


def generate_random_coordinates():
    """Generate random coordinates."""
    return f"{np.random.uniform(0, 90):.4f}, {np.random.uniform(0, 90):.4f}"


def iter_dataset_folder(dataset_folder):
    dataset_path = Path(dataset_folder)

    all_image_names = [img_name.name for img_name in dataset_path.iterdir()]
    all_image_names.sort(key=str.lower)

    for image_name in all_image_names:
        image_path = dataset_path / image_name
        yield image_path.as_posix()


def send_image_and_data():
    # listing all images
    image_paths = list(iter_dataset_folder(SEQUENCE_PATH))

    # iterating over the dataset
    for image_path in track(
        image_paths,
        total=len(image_paths),
        description=f"Sending {len(image_paths)} Events to SENTINEL Cloud",
    ):
        # Prepare the data
        payload = {
            "data": json.dumps(
                {
                    "coord": generate_random_coordinates(),
                    "mission_name": "Alpha Mission - 2024-05-12",
                    "sentinel_id": "sentinel-001",
                }
            )
        }

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
        response = session.post(CENTRAL_SERVER_URL, data=payload, files=files)

        sleep(1)


if __name__ == "__main__":
    response = send_image_and_data()
    session.close()
