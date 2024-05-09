import json
from typing import Annotated

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, UploadFile

sentinel = FastAPI()


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

    # Return a response
    return {"message": f"Object registered successfully."}
