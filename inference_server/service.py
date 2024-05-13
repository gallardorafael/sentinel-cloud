from __future__ import annotations

from typing import Dict, List

import bentoml
import cv2
import numpy as np
from detectors import YOLOv6Detector, YOLOv6FaceDetector
from PIL import Image
from utils.postprocess import draw_bounding_boxes


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class ObjectDetectionService:
    def __init__(self):
        self.object_detector = YOLOv6Detector()
        self.face_detector = YOLOv6FaceDetector()

    @bentoml.api
    def detect_objects_bboxes(self, image: np.ndarray) -> List[Dict]:
        """Detects objects in the input image and returns the bounding boxes.

        Args:
            image (np.ndarray): The input image. Assumed to be in cv2 format.

        Returns:
            List[Dict]: The detected objects as a list of dictionaries, each containing the bounding box coordinates (format x1y1 ... xNyN),
            confidence score and class label.
        """
        preprocessed_image = self.object_detector._preprocess(image)

        detections = self.object_detector.predict(preprocessed_image)[0]

        formatted_detections = self.object_detector._postprocess(detections, image.shape[:2])

        return formatted_detections

    @bentoml.api
    def detect_faces_bboxes(self, image: np.ndarray) -> List[Dict]:
        """Detects faces in the input image and returns the bounding boxes.

        Args:
            image (np.ndarray): The input image. Assumed to be in cv2 format.

        Returns:
            List[Dict]: The detected faces as a list of dictionaries, each containing the bounding box coordinates (format x1y1 ... xNyN),
            confidence score, class label, and landmarks.
        """
        preprocessed_image = self.face_detector._preprocess(image)

        detections = self.face_detector.predict(preprocessed_image)[0]

        formatted_detections = self.face_detector._postprocess(detections, image.shape[:2])

        return formatted_detections

    @bentoml.api
    def detect_plot_faces(self, image: np.ndarray) -> Image.Image:
        """Detects faces in the input image and returns the bounding boxes.

        Args:
            image (np.ndarray): The input image. Assumed to be in cv2 format.
        Returns:
            PILImage: The input image with the detected faces plotted.
        """
        preprocessed_image = self.face_detector._preprocess(image)

        detections = self.face_detector.predict(preprocessed_image)[0]

        formatted_detections = self.face_detector._postprocess(detections, image.shape[:2])

        # converting to PIL image and going back to RGB
        postprocessed_image = Image.fromarray(
            draw_bounding_boxes(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                [det["bbox"] for det in formatted_detections],
            )
        )

        return postprocessed_image
