from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DETECT_DATA_DIR = DATA_DIR / "detect"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CROP_OUTPUT_DIR = OUTPUTS_DIR / "crop"
PREDICTIONS_OUTPUT_DIR = OUTPUTS_DIR / "predictions"
LOG_OUTPUT_DIR = OUTPUTS_DIR / "logs"
FIGURE_OUTPUT_DIR = OUTPUTS_DIR / "figure"
OCR_OUTPUT_DIR = OUTPUTS_DIR / "ocr"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
DETECTOR_WEIGHTS_DIR = WEIGHTS_DIR / "detector"
DETECTOR_CHECKPOINT_DIR = DETECTOR_WEIGHTS_DIR / "best.pth"
DEEPSEEK_OCR_WEIGHTS_DIR = WEIGHTS_DIR / "deepseek_ocr" / "deepseek_ocr_merged"
