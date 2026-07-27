import inspect

from config import load_config
from pipeline.detector import ShipDetector


def test_detection_threshold_defaults_are_consistent():
    config = load_config()
    signature = inspect.signature(ShipDetector.__init__)

    assert config["yolo"]["confidence"] == 0.5
    assert config["yolo"]["iou"] == 0.1
    assert config["pipeline"]["conf_threshold"] == 0.5
    assert config["pipeline"]["iou_threshold"] == 0.1
    assert signature.parameters["conf_threshold"].default == 0.5
    assert signature.parameters["iou_threshold"].default == 0.1
