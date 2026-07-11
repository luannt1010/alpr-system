import os
import json
import argparse
from alpr.detector.helper_utils import crop_image
from alpr.detector.inference import Detector
from alpr.deepseek_ocr.inference import OCR
from alpr.paths import CROP_OUTPUT_DIR, PREDICTIONS_OUTPUT_DIR, DEEPSEEK_OCR_WEIGHTS_DIR

class PIPELINE:
    def __init__(self, yolo=False):
        self.detector = Detector(yolo)
        self.ocr = OCR(DEEPSEEK_OCR_WEIGHTS_DIR)

    def predict(self, img_path):
        results = self.detector.detect(img_path)
        if results is None:
            return None

        preds_path = PREDICTIONS_OUTPUT_DIR
        preds_path.mkdir(parents=True, exist_ok=True)

        img_cropped = crop_image(img_path, results["BBox"])
        CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        crop_name_sp = "cropped"+"_"+f"{len(os.listdir(preds_path))}.png"
        crop_sp = os.path.join(CROP_OUTPUT_DIR, crop_name_sp)
        img_cropped.save(crop_sp)

        text = self.ocr.inference(crop_sp)
        results["Text"] = text

        results_save_sp = "results"+"_"+f"{len(os.listdir(preds_path))}.json"
        pred_sp = os.path.join(preds_path, results_save_sp)
        with open(pred_sp, "w") as f:
            json.dump(results, f)

        return results

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo", type=bool, default=False)
    parser.add_argument("--img_path", type=str, default="")

    return parser.parse_args()

def main():
    args = get_args()

    is_yolo = args.yolo
    img_path = args.img_path
    pipeline = PIPELINE(is_yolo)
    results = pipeline.predict(img_path)

    if results is not None:
        print(results)
    else:
        print("Not found license plate in the image.")

if __name__ == "__main__":
    main()

