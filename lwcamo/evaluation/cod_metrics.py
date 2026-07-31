"""Standard COD metric evaluation using grayscale probability maps."""

from pathlib import Path
import cv2
from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure


def evaluate_directories(prediction_directory, mask_directory):
    prediction_directory, mask_directory = Path(prediction_directory), Path(mask_directory)
    paths = sorted(path for path in prediction_directory.iterdir() if path.is_file())
    if not paths:
        raise ValueError(f"No predictions in {prediction_directory}")
    metrics = [Smeasure(), Emeasure(), WeightedFmeasure(), MAE()]
    for prediction_path in paths:
        mask_path = mask_directory / prediction_path.name
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if prediction is None or mask is None or prediction.shape != mask.shape:
            raise ValueError(f"Invalid or mismatched pair: {prediction_path.name}")
        for metric in metrics:
            metric.step(pred=prediction, gt=mask)
    sm, em, wfm, mae = (metric.get_results() for metric in metrics)
    return {
        "S_alpha": float(sm["sm"]),
        "E_phi_adaptive": float(em["em"]["adp"]),
        "F_beta_weighted": float(wfm["wfm"]),
        "MAE": float(mae["mae"]),
        "images": len(paths),
    }
