from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import bentoml
import cv2
import numpy as np
from PIL import Image
from utils.postprocess import draw_bounding_boxes, non_max_suppression, scale_coords
from utils.preprocess import letterbox_yolov6


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class FaceDetection:
    def __init__(
        self,
        model_tag: Optional[str] = "yolov6n_face:latest",
        input_shape: Tuple[int, int] = (640, 640),
    ) -> None:
        """Initialize the FaceDetection service.

        Args:
            model_tag (Optional[str]): The tag of the ONNX model, which must be registered into the BentoML model store.
        """
        # Load the ONNX model
        try:
            # This loads the ONNX and returns a onnxruntime.InferenceSession object
            self.model = bentoml.onnx.load_model(model_tag)
        except Exception as e:
            raise ValueError(f"Failed to initialize the model: {e}")

        self.input_shape = input_shape

    def predict(self, preprocessed_image: np.ndarray):
        """Detects faces in the input image and returns the bounding boxes.

        This only works for ONNX models.
        """
        outputs = self.model.run(["outputs"], {"images": preprocessed_image})

        # Compute ONNX Runtime output prediction
        ort_inputs = {self.model.get_inputs()[0].name: preprocessed_image}

        # ONNX Runtime will return a list of outputs
        outputs = self.model.run(None, ort_inputs)

        return outputs

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Performs all preprocessing steps before predicting over an image. Preprocessing done
        according to:
        https://github.com/meituan/YOLOv6/blob/yolov6-face/yolov6/core/inferer.py#L180.

        Args:
            image: Image to perform preprocessing on. This assumes an input with cv2 format
                HWC and will be converted to CHW, also assumed BGR to be converted to RGB

        Returns:
            preprocessed_image: An object representing the image as required by the actual model.
        """
        # explicit typing as uint8, to avoid resize issues in cv2
        image = image.astype(np.uint8)

        # letterboxing square image to 640, 640
        preprocessed_image = letterbox_yolov6(image, new_shape=self.input_shape, auto=False)[0]

        # HWC to CHW, BGR to RGB
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))[::-1].astype(np.float32)

        # 0 - 255 to 0.0 - 1.0
        preprocessed_image = preprocessed_image / 255.0

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
            List[Dict]: The formatted face detection results as a list of dictionaries. The dictionary contains
                the bounding box coordinates (format x1y1 ... xNyN), confidence score, class label, and landmarks.
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
            # Note: cast to int to avoid serialization issues with Pydantic
            if len(det):
                formatted_detections.append(
                    {
                        "bbox": [round(c.item()) for c in det[:4]],
                        "confidence": det[4].item(),
                        "class": det[5].item(),
                        "landmarks": [round(c.item()) for c in det[6:]],
                    }
                )

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
        preprocessed_image = self._preprocess(image)

        detections = self.predict(preprocessed_image)[0]

        formatted_detections = self._postprocess(detections, image.shape[:2])

        return formatted_detections

    @bentoml.api
    def detect_faces_plot(self, image: np.ndarray) -> Image.Image:
        """Detects faces in the input image and returns the bounding boxes.

        Args:
            image (np.ndarray): The input image. Assumed to be in cv2 format.
        Returns:
            PILImage: The input image with the detected faces plotted.
        """
        preprocessed_image = self._preprocess(image)

        detections = self.predict(preprocessed_image)[0]

        formatted_detections = self._postprocess(detections, image.shape[:2])

        # converting to PIL image and going back to RGB
        postprocessed_image = Image.fromarray(
            draw_bounding_boxes(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                [det["bbox"] for det in formatted_detections],
            )
        )

        return postprocessed_image
