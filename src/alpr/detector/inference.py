import cv2
import argparse
from alpr.detector.helper_utils import read_img, draw_bbox, crop_image
import numpy as np
import torch
from torchvision import transforms
from alpr.paths import FASTRCNN_CHECKPOINT_DIR, YOLO_CHECKPOINT_DIR
from alpr.detector.helper_utils import load_faster_rcnn_state_dict, load_yolo_state_dict
from alpr.detector.config import MODEL_OBJECT

class FasterRCNNDetector:
    def __init__(self, model_obj, num_classes, weight_path):
        self.model = load_faster_rcnn_state_dict(model_obj, num_classes, weight_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def detect(self, img):
        orig_img = read_img(img)
        trans = transforms.ToTensor()
        img = trans(orig_img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img)[0]
        scores, labels, boxes = predictions["scores"], predictions["labels"], predictions["boxes"]
        if 0 in (len(scores), len(labels), len(boxes)):
            return None
        # Only get one license when vehicle go into parking area
        max_score_idx = torch.argmax(scores)
        score = scores[max_score_idx]
        label = labels[max_score_idx]
        bbox = boxes[max_score_idx].cpu().numpy().astype(int).tolist()
        return {"Score": score.item(), "Label": label.item(), "BBox": bbox}


class YOLODetector:
    def __init__(self, wp):
        self.model = load_yolo_state_dict(wp)

    def detect(self, img):
        results = self.model(img)
        outputs = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            outputs.append(boxes)
            outputs.append(scores)
            outputs.append(class_ids)

        boxes, scores, class_ids = outputs
        is_license_plate = class_ids == 2
        max_score_idx = np.argmax(scores[is_license_plate])
        final_boxes = boxes[max_score_idx].astype(int)
        final_class_ids = class_ids[max_score_idx]
        final_scores = scores[max_score_idx]
        return {"Score": final_scores.item(), "Label": final_class_ids.astype(int).item(), "BBox": final_boxes.tolist()}

class Detector:
    def __init__(self, yolo=False):
        if yolo:
            self.detector = YOLODetector(YOLO_CHECKPOINT_DIR)
        else:
            self.detector = FasterRCNNDetector(MODEL_OBJECT["mobile_net_large"], 3, FASTRCNN_CHECKPOINT_DIR)
    def detect(self, img):
        return self.detector.detect(img)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=bool, default=False)
    parser.add_argument("--img_path", type=str, default="")
    return parser.parse_args()

def main():
    args = get_args()
    is_yolo = args.yolo
    img_path = args.img_path

    detector = Detector(is_yolo)
    results = detector.detect(img_path)
    if results == None:
        print("Not found license plate in the image.")
    else:
        print(results)
        img_cropped = crop_image(img_path, results["BBox"])
        img_cropped.show()

if __name__ == "__main__":
    main()