from typing import Dict, List, Optional, Tuple

import bentoml
import numpy as np
from utils.postprocess import get_coco_idx_to_class, non_max_suppression, scale_coords
from utils.preprocess import letterbox_yolov6


class YOLOv6Detector:
    def __init__(
        self,
        model_tag: Optional[str] = "yolov6s:latest",
        input_shape: Optional[Tuple[int, int]] = (640, 640),
        max_detections: Optional[int] = 300,
    ) -> None:
        """Initialize the FaceDetection service.

        Args:
            model_tag (str): The tag of the ONNX model, which must be registered into the BentoML model store.
            input_shape (Tuple[int, int]): The input shape of the model.
        """
        # Load the ONNX model
        try:
            # This loads the ONNX and returns a onnxruntime.InferenceSession object
            self.model = bentoml.onnx.load_model(model_tag)
        except Exception as e:
            raise ValueError(f"Failed to initialize the model: {e}")

        self.input_shape = input_shape
        self.idx_to_class = get_coco_idx_to_class()
        self.max_detections = max_detections

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
        confidence_thresh: float = 0.2,
        return_class_names: bool = True,
    ) -> List[Dict]:
        """Postprocesses the face detection results.

        Args:
            detections: The raw detection results.
            original_img_shape (Tuple[int, int]): The shape of the original image. This assumes a vector with shape [1,8400,16].
                This class is not prepared to work with a model that already includes the NMS step (end2end).
            confidence_thresh (float, optional): The confidence threshold for filtering detections. Defaults to 0.3.

        Returns:
            List[Dict]: The formatted face detection results as a list of dictionaries. The dictionary contains
                the bounding box coordinates (format x1y1 ... xNyN), confidence score and class label.
        """
        nms_results = non_max_suppression(
            detections,
            conf_thres=confidence_thresh,
            agnostic=False,
            extra_data=0,
            max_det=self.max_detections,
        )[0]

        if nms_results.shape[0] == 0:
            return []

        # reescaling face bboxes
        nms_results[:, :4] = scale_coords(self.input_shape, nms_results[:, :4], original_img_shape)

        formatted_detections = []
        for det in nms_results:
            # Note: cast to int to avoid serialization issues with Pydantic
            if len(det):
                object_class = det[5].item()

                if return_class_names:
                    object_class = self.idx_to_class[object_class]

                formatted_detections.append(
                    {
                        "bbox": [round(c.item()) for c in det[:4]],
                        "confidence": det[4].item(),
                        "class": object_class,
                    }
                )

        return formatted_detections


class YOLOv6FaceDetector(YOLOv6Detector):
    def __init__(
        self,
        model_tag: Optional[str] = "yolov6s_face:latest",
        input_shape: Optional[Tuple[int, int]] = (640, 640),
    ) -> None:
        super().__init__(model_tag, input_shape)

    def _postprocess(
        self,
        detections,
        original_img_shape: Tuple[int, int],
        confidence_thresh: float = 0.3,
        return_class_names: bool = True,
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
            max_det=self.max_detections,
        )[0]

        if nms_results.shape[0] == 0:
            return []

        # reescaling face bboxes
        nms_results[:, :4] = scale_coords(self.input_shape, nms_results[:, :4], original_img_shape)

        # reescaling face landmarks
        nms_results[:, 6:] = scale_coords(self.input_shape, nms_results[:, 6:], original_img_shape)

        formatted_detections = []
        for det in nms_results:
            # Note: cast to int to avoid serialization issues with Pydantic
            if len(det):
                object_class = det[5].item()

                if return_class_names:
                    object_class = "face"

                formatted_detections.append(
                    {
                        "bbox": [round(c.item()) for c in det[:4]],
                        "confidence": det[4].item(),
                        "class": object_class,
                        "landmarks": [round(c.item()) for c in det[6:]],
                    }
                )

        return formatted_detections
