from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL.Image import Image as PILImage
from utils.postprocess import non_max_suppression, scale_coords
from utils.preprocess import letterbox_yolov6

import bentoml


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class FaceDetection:
    def __init__(self, model_tag: Optional[str] = "yolov6n_face") -> None:
        """Initialize the FaceDetection service.

        Args:
            model_tag (Optional[str]): The tag of the ONNX model, which must be registered into the BentoML model store.
        """
        # Load the ONNX model
        try:
            self.model = bentoml.onnx.get(model_tag)
        except Exception as e:
            raise ValueError(f"Failed to initialize the model: {e}")

    def preprocess(self, image: PILImage) -> np.ndarray:
        """Preprocesses the input image for face detection.

        Args:
            image (PILImage): The input image.

        Returns:
            np.ndarray: The preprocessed image as a NumPy array.
        """
        preprocessed_image = np.array(image)

        # letterboxing square image to 640, 640
        preprocessed_image = letterbox_yolov6(image, new_shape=self.input_shape, auto=False)[0]

        # HWC to CHW, BGR to RGB
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))[::-1].astype(np.half)

        # 0 - 255 to 0.0 - 1.0
        preprocessed_image /= 255

        # adding batch dim
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        return preprocessed_image

    def _postprocess(
        self,
        detections,
        original_img_shape: Tuple[int, int],
        confidence_thresh: float = 0.3,
    ) -> List[Dict]:
        """Postprocesses the face detection results.

        Args:
            detections: The raw detection results.
            original_img_shape (Tuple[int, int]): The shape of the original image. This assumes a vector with shape [1,8400,16].
                This class is not prepared to work with a model that already includes the NMS step (end2end).
            confidence_thresh (float, optional): The confidence threshold for filtering detections. Defaults to 0.3.

        Returns:
            List[Dict]: The formatted face detection results as a list of dictionaries.
        """
        nms_results = non_max_suppression(
            detections,
            conf_thres=confidence_thresh,
            agnostic=False,
            extra_data=10,
            max_det=100,
        )[0]

        # reescaling face bboxes
        nms_results[:, :4] = scale_coords(self.input_shape, nms_results[:, :4], original_img_shape)

        # reescaling face landmarks
        nms_results[:, 6:] = scale_coords(self.input_shape, nms_results[:, 6:], original_img_shape)

        formatted_detections = []
        for det in nms_results:
            if len(det):
                formatted_detections.append(
                    {"bbox": det[:4], "confidence": det[4], "class": det[5], "landmarks": det[6:]}
                )

        return formatted_detections

    @bentoml.api
    def detect_faces_bboxes(self, image: PILImage) -> List[Dict]:
        """Detects faces in the input image and returns the bounding boxes.

        Args:
            image (PILImage): The input image.

        Returns:
            List[Dict]: The detected faces as a list of dictionaries, each containing the bounding box coordinates, confidence score, class label, and landmarks.
        """
        preprocessed_image = self.preprocess(image)

        detections = self.model(preprocessed_image)

        formatted_detections = self._postprocess(detections, image.size)

        return formatted_detections
