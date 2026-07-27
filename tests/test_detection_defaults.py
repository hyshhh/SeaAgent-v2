import inspect

from config import load_config
from pipeline.detector import ShipDetector


def test_detection_threshold_defaults_are_consistent():
    config = load_config()
    signature = inspect.signature(ShipDetector.__init__)

    assert config["yolo"]["confidence"] == 0.5
    assert config["yolo"]["iou"] == 0.5
    assert config["pipeline"]["conf_threshold"] == 0.5
    assert config["pipeline"]["iou_threshold"] == 0.5
    assert signature.parameters["conf_threshold"].default == 0.5
    assert signature.parameters["iou_threshold"].default == 0.5


def test_tracking_defaults_are_consistent():
    config = load_config()
    tracker = config["yolo"]["tracker_params"]
    appearance = config["yolo"]["appearance_tracking"]

    assert tracker["track_high_thresh"] == 0.5
    assert tracker["track_low_thresh"] == 0.1
    assert tracker["new_track_thresh"] == 0.5
    assert tracker["track_buffer"] == 90
    assert tracker["match_thresh"] == 0.8
    assert tracker["fuse_score"] is True
    assert config["pipeline"]["max_stale_frames"] == 120
    assert config["pipeline"]["appearance_tracking"] == appearance
    assert appearance["enabled"] is False
    assert appearance["appearance_thresh"] == 0.8
    assert appearance["proximity_thresh"] == 0.5
