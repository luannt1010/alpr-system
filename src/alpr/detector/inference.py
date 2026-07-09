import cv2
import argparse
import torch
from PIL import Image
from torchvision import transforms
from alpr.paths import DETECTOR_CHECKPOINT_DIR
from alpr.detector.helper_utils import load_model_state_dict
from alpr.detector.config import MODEL_OBJECT

class Detector:
    def __init__(self, model_obj, num_classes, weight_path):
        self.model = load_model_state_dict(model_obj, num_classes, weight_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def detect(self, img_path):
        orig_img = Image.open(img_path).convert("RGB")
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
        max_score_idx = torch.argmax(scores)
        score = scores[max_score_idx]
        label = labels[max_score_idx]
        bbox = boxes[max_score_idx].cpu().numpy().astype(int).tolist()
        return {"Score": score.item(), "Label": label.item(), "BBox": bbox}

    def crop_image(self, img_path, bbox):
        image = Image.open(img_path).convert("RGB")
        return image.crop(bbox)

    def draw_bbox(self, img_path, bbox, label, score):
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = bbox
        img = cv2.rectangle(orig_img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=3)
        img = cv2.putText(img, f"{label, round(score,2)}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return img


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=list(MODEL_OBJECT.keys()), default="mobile_net_large")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--weight_path", type=str, default=DETECTOR_CHECKPOINT_DIR)
    parser.add_argument("--img_path", type=str, default="")

    return parser.parse_args()

def main():
    args = get_args()
    model_obj = args.model
    num_classes = args.num_classes
    weight_path = args.weight_path
    img_path = args.img_path

    detector = Detector(MODEL_OBJECT[model_obj], num_classes, weight_path)
    results = detector.detect(img_path)
    if results == None:
        print("Not found license plate in the image.")
    else:
        print(results)
        img_cropped = detector.crop_image(img_path, results["BBox"])
        img_cropped.show()

if __name__ == "__main__":
    main()

