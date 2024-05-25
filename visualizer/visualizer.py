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

CLASS_COLORS = {
    "face": [157, 50, 168],
    "person": [207, 142, 64],
    "car": [77, 108, 143],
    "truck": [30, 87, 18],
    "airplane": [209, 182, 61],
    "boat": [245, 66, 215],
}

SERIES_CONFIGS = [
    {"route": "interest_object/frame/personnel", "color": [207, 142, 64], "name": "Personnel"},
    {"route": "interest_object/frame/airplanes", "color": [209, 182, 61], "name": "Airplanes"},
    {"route": "interest_object/frame/boats", "color": [245, 66, 215], "name": "Boats"},
    {"route": "interest_object/frame/trucks", "color": [30, 87, 18], "name": "Trucks"},
    {"route": "interest_object/frame/cars", "color": [77, 108, 143], "name": "Cars"},
]


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

        # creating counter series
        self._create_counter_series(SERIES_CONFIGS)

    def _create_counter_series(self, series_configs: List[dict]) -> None:
        """Create counter series for logging according to the series_configs.

        This method creates counter series for different objects
        using the `rr.log` function from the `rr` module. Each series is configured
        with a specific route, color, name, and width.

        Args:
            series_configs: A list of dictionaries containing the series configurations, must include
            the route, color, and name of the series.
        """
        for series_config in series_configs:
            rr.log(
                series_config["route"],
                rr.SeriesLine(
                    color=series_config["color"],
                    name=series_config["name"],
                    width=2,
                ),
                timeless=True,
            )

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

        # getting object counts
        detected_classes = [detection["class"] for detection in additional_data["object_dets"]]
        n_persons = detected_classes.count("person")
        n_boats = detected_classes.count("boat")
        n_cars = detected_classes.count("car")
        n_trucks = detected_classes.count("truck")
        n_airplanes = detected_classes.count("airplane")

        print(
            f"Detected: {n_persons} persons, {n_boats} boats, {n_cars} cars, {n_trucks} trucks, {n_airplanes} airplanes"
        )

        # logging stats (counters)
        rr.log("interest_object/stats/personnel", rr.Scalar(n_persons))
        rr.log("interest_object/stats/boats", rr.Scalar(n_boats))
        rr.log("interest_object/stats/cars", rr.Scalar(n_cars))
        rr.log("interest_object/stats/trucks", rr.Scalar(n_trucks))
        rr.log("interest_object/stats/airplanes", rr.Scalar(n_airplanes))

    def _setup_logging(self) -> None:
        logger = logging.getLogger()
        rerun_handler = rr.LoggingHandler("logs")
        rerun_handler.setLevel(-1)
        logger.addHandler(rerun_handler)

        return logger
