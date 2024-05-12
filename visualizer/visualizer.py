import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr

# Ensure the logging gets written to stderr:
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel("DEBUG")

DESCRIPTION = """
# SENTINEL Cloud Visualizer

This is a visualizer for the SENTINEL Cloud project. Provides a customizable
interface for visualizing multimodal data and annotations returned by the
deployed Sentinels. A single Cloud Visualizer can manage the streams of
multiple Sentinels.
"""


class SentinelVisualizer:
    def __init__(
        self,
        app_id: Optional[str] = "default",
        recording_id: Optional[uuid.UUID] = uuid.uuid4(),
        log_file: Optional[Path] = None,
    ) -> None:

        # init a session in rerun
        rr.init(app_id, recording_id=recording_id)

        # connecting to the session
        rr.connect()

        # setting output log file
        if log_file:
            rr.save(log_file)

        # setup logging
        self._setup_logging()

        # registered classes
        self.class_to_index = {}

        # total frames register
        self.current_frame = 0

        # logging app description
        rr.log(
            "app/description",
            rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
            timeless=True,
        )

    def log_interest_frame(self, cv2_image: np.ndarray) -> None:
        rr.set_time_sequence("frame", self.current_frame)
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        rr.log("interest_object/frame", rr.Image(rgb).compress(jpeg_quality=85))

        self.current_frame += 1

    def log_interest_bboxes(self, boxes: npt.NDArray[np.float32], class_names=List[str]) -> None:

        rr.set_time_sequence("frame", self.current_frame)
        for class_name in class_names:
            if class_name not in self.class_to_index:
                self.class_to_index[class_name] = len(self.class_to_index)

        class_ids = [self.class_to_index[class_name] for class_name in class_names]

        rr.log(
            "interest_object/frame/objects",
            rr.Boxes2D(
                array=boxes,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=class_ids,
            ),
        )

    def log_interest_metadata(self, additional_data: dict) -> None:
        rr.log(
            "interest_object/frame/metadata", rr.TextDocument(json.dumps(additional_data, indent=4))
        )

    def _setup_logging(self) -> None:
        logger = logging.getLogger()
        rerun_handler = rr.LoggingHandler("logs")
        rerun_handler.setLevel(-1)
        logger.addHandler(rerun_handler)
