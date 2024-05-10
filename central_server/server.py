import json
from typing import Annotated

import bentoml
import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, UploadFile
from PIL import Image

sentinel = FastAPI()

INFERENCER_SERVER_URL = "http://localhost:3000"


@sentinel.post("/sentinel/api/interest_object")
async def register_object(
    file: Annotated[UploadFile, File()], data: Annotated[str, Form] = Body(...)
):
    # Decode the data
    additional_data = json.loads(data)

    # Read the image file
    image_bytes = await file.read()

    # Convert the bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array to an OpenCV image
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # log obect to the visualizer
    log_interest_object(cv2_image=cv_image, additional_data=additional_data)

    # Return a response
    return {"message": f"Object registered successfully."}


def log_interest_object(cv2_image: np.ndarray, additional_data: dict):
    # get face detection bboxes
    with bentoml.SyncHTTPClient(INFERENCER_SERVER_URL) as client:
        print(f"Sending image to inference server...")
        result = client.detect_faces_bboxes(image=cv2_image)

    # log the object
    print(f"Face bboxes: {result}")
