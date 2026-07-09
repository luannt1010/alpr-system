import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.ops import box_convert

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_file_dir = os.path.join(os.path.join(self.root_dir, "annotations"), "_annotations.coco.json")
        self.images = sorted(os.listdir(self.img_dir))
        self.transforms = transforms
        with open(self.ann_file_dir, "r") as f:
                self.coco_file = json.load(f)
        self.image_name_idx = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.retrieve_image(idx)
        target = self.retrieve_target()
        if self.transforms is not None:
            image = self.transforms(image)
        return image, target

    def retrieve_image(self, idx):
        img_name = self.images[idx]
        self.image_name_idx = img_name
        image_path = os.path.join(self.img_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        return image
    
    def retrieve_target(self):
        if self.image_name_idx is not None:
            annotations_coco = self.coco_file["annotations"]
            images_coco = self.coco_file["images"]
            boxes, labels = [], []
            for image in images_coco:
                if image["file_name"] == self.image_name_idx:
                    image_id = image["id"]
                    for ann in annotations_coco:
                        if ann["image_id"] == image_id:
                            boxes.append(ann["bbox"])
                            labels.append(int(ann["category_id"]))
            boxes = box_convert(torch.tensor(boxes, dtype=torch.float32), in_fmt="xywh", out_fmt="xyxy")
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
            return target
        else:
            raise ValueError("Name of image is invalid!")
        

class SubsetLicensePlate(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, target = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target
    
