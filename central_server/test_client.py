import io
import json
import logging
from datetime import date
from pathlib import Path
from typing import List

import bentoml
import cv2
import numpy as np
import requests
from rich.progress import track

SEQUENCE_PATH = "/home/yibbtstll/Projects/sentinel/data/YouTube/carrier_1/frames"
CENTRAL_SERVER_URL = "http://localhost:8001"

"""
This is a simulation of a SENTINEL, which sends images and data to the central server. The object
detector will be running on the Edge, but here, we will be calling the Cloud service for
object detection. For demo purposes only.
"""

INFERENCER_SERVER_URL = "http://localhost:3000"
inference_client = bentoml.SyncHTTPClient(INFERENCER_SERVER_URL)
session = requests.Session()

logger = logging.getLogger(__name__)

# once implmented in the edge, this classes will be updates from Sentinel Cloud
OBJECTS_OF_INTEREST = session.get(
    f"{CENTRAL_SERVER_URL}/sentinel/api/interest_objects_list"
).json()["objects_list"]


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


def detect_objects(cv2_image: np.ndarray) -> List[dict]:
    # get object detection bboxes
    object_dets = inference_client.detect_objects_bboxes(image=cv2_image)

    logger.debug("%s obects were detected.", len(object_dets))

    return object_dets


def filter_objects_of_interest(object_dets: List[dict]) -> List[dict]:
    interest_objects = [obj for obj in object_dets if obj["class"] in OBJECTS_OF_INTEREST]

    return interest_objects


def send_image_and_data():
    # listing all images
    image_paths = list(iter_dataset_folder(SEQUENCE_PATH))

    # iterating over the dataset
    for image_path in track(
        image_paths,
        total=len(image_paths),
        description=f"Sending {len(image_paths)} Events to SENTINEL Cloud",
    ):
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

        # extracting objects
        object_dets = detect_objects(cv2_image=image)
        interest_detections = filter_objects_of_interest(object_dets)
        logger.debug(
            "Objects of interest %s found in frame, sending to Sentinel Cloud.", OBJECTS_OF_INTEREST
        )

        # TODO: send only the frames with objects of interest
        # Prepare the additional data
        payload = {
            "data": json.dumps(
                {
                    "coord": generate_random_coordinates(),
                    "mission_name": f"Alpha Mission - {date.today().isoformat()}",
                    "sentinel_id": "sentinel-001",
                    "object_dets": interest_detections,
                }
            )
        }
        # Send the POST request
        response = session.post(
            f"{CENTRAL_SERVER_URL}/sentinel/api/interest_object", data=payload, files=files
        )
        logger.debug("Response received from Setinel Cloud: %s.", response.json())


if __name__ == "__main__":
    response = send_image_and_data()
    session.close()
