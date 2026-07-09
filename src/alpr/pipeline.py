import os
import json
import argparse
from alpr.detector.config import MODEL_OBJECT
from alpr.detector.inference import Detector
from alpr.deepseek_ocr.inference import OCR
from alpr.paths import CROP_OUTPUT_DIR, PREDICTIONS_OUTPUT_DIR, DETECTOR_CHECKPOINT_DIR, DEEPSEEK_OCR_WEIGHTS_DIR

class PIPELINE:
    def __init__(self, model_obj, num_classes, weight_path, ocr_cp_path):
        self.detector = Detector(model_obj, num_classes, weight_path)
        self.ocr = OCR(ocr_cp_path)

    def predict(self, img_path):
        results = self.detector.detect(img_path)
        if results is None:
            return None

        img_cropped = self.detector.crop_image(img_path, results["BBox"])
        CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        crop_sp = os.path.join(CROP_OUTPUT_DIR, os.path.basename(img_path))
        img_cropped.save(crop_sp)

        text = self.ocr.inference(crop_sp)
        results["Text"] = text
        preds_path = PREDICTIONS_OUTPUT_DIR
        preds_path.mkdir(parents=True, exist_ok=True)
        pred_sp = os.path.join(preds_path, "results"+"_"+f"{len(os.listdir(preds_path))}.json")
        with open(pred_sp, "w") as f:
            json.dump(results, f)

        return results

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=list(MODEL_OBJECT.keys()), default="mobile_net_large")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--weight_path", type=str, default=DETECTOR_CHECKPOINT_DIR)
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--ocr_cp", type=str, default=DEEPSEEK_OCR_WEIGHTS_DIR)

    return parser.parse_args()

def main():
    args = get_args()

    model_obj = args.model
    num_classes = args.num_classes
    weight_path = args.weight_path
    img_path = args.img_path
    ocr_cp = args.ocr_cp

    model = MODEL_OBJECT[model_obj]
    pipeline = PIPELINE(model, num_classes, weight_path, ocr_cp)

    results = pipeline.predict(img_path)

    if results is not None:
        print(results)
    else:
        print("Not found license plate in the image.")

if __name__ == "__main__":
    main()

