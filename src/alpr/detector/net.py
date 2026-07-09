import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
class Model(nn.Module):
    def __init__(self, model_obj, num_classes):
        super().__init__()
        self.model = model_obj(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, image, target=None):
        return self.model(image, target)