import json
import logging
from typing import Annotated, Any, Optional

import bentoml
import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, File, Form, UploadFile

logger = logging.getLogger(__name__)

from visualizer import SentinelVisualizer

INFERENCER_SERVER_URL = "http://localhost:3000"


class SentinelCentralServer(FastAPI):
    def __init__(
        self,
        inferencer_server_url: Optional[str] = INFERENCER_SERVER_URL,
        visualizer_kwargs: dict = {},
        **extra: Any,
    ):
        super().__init__(**extra)

        self.inference_client = bentoml.SyncHTTPClient(inferencer_server_url)
        self.visualizer = SentinelVisualizer(**visualizer_kwargs)

        # define api routes
        self.add_api_route("/sentinel/api/interest_object", self.register_object, methods=["POST"])

        logger.info("Sentinel Central Server initialized")

    async def register_object(
        self, file: Annotated[UploadFile, File()], data: Annotated[str, Form] = Body(...)
    ):
        # Decode the data
        additional_data = json.loads(data)

        # Read the image file
        image_bytes = await file.read()

        # Convert the bytes to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode the numpy array to an OpenCV image
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        logger.debug(
            "Received a request to register a frame with size: %s and metadata: %s",
            cv_image.shape,
            additional_data.keys(),
        )

        # log obect to the visualizer
        self.log_interest_object(cv2_image=cv_image, additional_data=additional_data)

        # Return a response
        return {"message": f"Object registered successfully."}

    def log_interest_object(self, cv2_image: np.ndarray, additional_data: dict):
        # log to the visualizer
        logger.debug("Logging object to the Cloud Visualizer")
        # logging the frame
        self.visualizer.log_interest_frame(cv2_image)

        # extracting object detection from additional data
        object_dets = additional_data["object_dets"]
        interest_classes = [detection["class"] for detection in object_dets]
        logger.debug("%s objects of interest detected.", len(interest_classes))

        # objects to be logged
        interest_bboxes = [detection["bbox"] for detection in object_dets]

        # face detection if persons are detected
        if "person" in interest_classes:
            # get face detection bboxes
            face_dets = self.inference_client.detect_faces_bboxes(image=cv2_image)
            logger.debug("Face Detection: %s faces", len(face_dets))

            # logging the face bboxes
            face_bboxes = [detection["bbox"] for detection in face_dets]
            interest_bboxes.extend(face_bboxes)
            interest_classes.extend(["face"] * len(face_bboxes))

        self.visualizer.log_interest_bboxes(boxes=interest_bboxes, labels=interest_classes)

        # log the metadata to the visualizer
        self.visualizer.log_interest_metadata(additional_data)


if __name__ == "__main__":
    app = SentinelCentralServer(
        title="Sentinel Central Server",
        description=f"Central server for the Sentinel project. This server is responsible for registering objects and sending them to the Cloud Visualizer.",
    )
    uvicorn.run(app, host="127.0.0.1", port=8001)
