import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr

from inference_server.utils.preprocess import letterbox_yolov6

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

CLASS_COLORS = {"face": [255, 0, 0], "person": [0, 255, 0]}


class SentinelVisualizer:
    def __init__(
        self,
        app_id: Optional[str] = "default",
        recording_id: Optional[uuid.UUID] = uuid.uuid4(),
        log_file: Optional[Path] = None,
        frame_size: Optional[Tuple[int, int]] = (640, 640),
    ) -> None:

        # init a session in rerun
        rr.init(app_id, recording_id=recording_id)

        # connecting to the session
        rr.connect()

        # setting output log file
        if log_file:
            rr.save(log_file)

        # setup logging
        self.logger = self._setup_logging()

        # registered classes
        self.class_to_index = {}

        # total frames register
        self.current_frame = 0

        self.frame_size = frame_size

        # logging app description
        rr.log(
            "app/description",
            rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
            timeless=True,
        )

        # creating stats dict
        self.sentinel_stats = defaultdict(list)

    def log_interest_frame(self, cv2_image: np.ndarray) -> None:
        # destination timeline
        rr.set_time_sequence("frame", self.current_frame)

        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        rr.log("interest_object/frame", rr.Image(rgb))

        self.current_frame += 1

    def log_interest_bboxes(self, boxes: npt.NDArray[np.float32], labels: List[str]) -> None:
        # destination timeline
        rr.set_time_sequence("frame", self.current_frame)

        rr.log(
            "interest_object/frame/objects",
            rr.Boxes2D(
                array=boxes,
                array_format=rr.Box2DFormat.XYXY,
                labels=labels,
                colors=[CLASS_COLORS[cls] for cls in labels],
            ),
        )

    def log_interest_metadata(self, additional_data: dict) -> None:
        # destination timeline
        rr.set_time_sequence("frame", self.current_frame)

        # logging metadata
        rr.log(
            "interest_object/frame/metadata", rr.TextDocument(json.dumps(additional_data, indent=4))
        )
        # creating crowd count series
        rr.log(
            "interest_object/stats/crowd",
            rr.SeriesLine(color=[255, 0, 0], name="Crowd count", width=2),
            timeless=True,
        )

        # logging stats (crowd count)
        if "n_persons" in additional_data:
            # saving record of number of faces
            self.sentinel_stats["crowd"].append(additional_data["n_persons"])

            rr.log("interest_object/stats/crowd", rr.Scalar(additional_data["n_persons"]))

    def _setup_logging(self) -> None:
        logger = logging.getLogger()
        rerun_handler = rr.LoggingHandler("logs")
        rerun_handler.setLevel(-1)
        logger.addHandler(rerun_handler)

        return logger
