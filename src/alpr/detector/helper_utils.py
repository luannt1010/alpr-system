import os
import json
import time
import torch
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from alpr.detector.net import Model
from alpr.paths import LOG_OUTPUT_DIR, FIGURE_OUTPUT_DIR, FASTRCNN_WEIGHTS_DIR
from torch.utils.data import random_split, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import transforms


# def draw_bbox(image, target, class_map):
#     boxes = target["boxes"]
#     tar_labels = target["labels"]
#     labels = [class_map[int(i)] for i in tar_labels]
#     result = vutils.draw_bounding_boxes(image=image,
#                                         boxes=boxes,
#                                         labels=labels,
#                                         colors="red",
#                                         width=4, font_size=30)
#     result = to_pil_image(result)
#     result.show()


def split_dataset(datasets, val_factor, test_factor):
    total_size = len(datasets)
    val_size = int(total_size * val_factor)
    test_size = int(total_size * test_factor)
    train_size = total_size - (val_size + test_size)
    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def check_path(img):
    return isinstance(img, (str, Path))

def read_img(img):
    if check_path(img):
        return Image.open(img).convert("RGB")

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

def crop_image(img, bbox):
    image = read_img(img)
    return image.crop(bbox)

def draw_bbox(img, bbox, label, score):
    orig_img = np.asarray(read_img(img))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = bbox
    img = cv2.rectangle(orig_img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=3)
    img = cv2.putText(img, f"{label, round(score,2)}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return img

def create_dataset_splits(dataset, val_factor, test_factor):
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_factor, test_factor)
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)),
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)),
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)),
                             pin_memory=True)
    return train_loader, val_loader, test_loader


def define_transforms():
    train_transform = transforms.Compose([transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.ToTensor()])
    return train_transform, val_transform

def load_yolo_state_dict(weight_path):
    return YOLO(weight_path)

def load_faster_rcnn_state_dict(model_obj, num_classes, state_dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_obj=model_obj, num_classes=num_classes)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    return model

def evaluate(model, data_loader, test=False, metric=None):
    if not test:
        if metric is None:
            raise TypeError("Only torchmetrics.MeanAveragePrecision")
        else:
            if not isinstance(metric, MeanAveragePrecision):
                raise TypeError("Only torchmetrics.MeanAveragePrecision")
    else:
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', extended_summary=True)
    metric.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_pbar = tqdm(data_loader, desc="[EVALUATING]", leave=True if test else False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, targets in data_pbar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
            preds, gts = [], []
            for i in range(len(images)):
                preds_boxes = outputs[i]["boxes"].cpu()
                preds_labels = outputs[i]["labels"].cpu()
                preds_scores = outputs[i]["scores"].cpu()
                gts_boxes = targets[i]["boxes"].cpu()
                gts_labels = targets[i]["labels"].cpu()
                preds.append({"boxes": preds_boxes, "scores": preds_scores, "labels": preds_labels})
                gts.append({"boxes": gts_boxes, "labels": gts_labels})
                metric.update(preds, gts)

    results = metric.compute()
    map50 = results["map_50"].item()
    map5095 = results["map"].item()
    return map50, map5095

def save_history(history):
    history_save_path = os.path.join(LOG_OUTPUT_DIR, "history.json")
    with open(history_save_path, "w") as f:
        json.dump(history, f)
    print(f"Saved history at {history_save_path}.")

def train(model, train_loader, val_loader, optimizer, num_epochs, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {"training_loss": [], "map50": [], "map5095": [], "total_time": 0}
    best_map_all = 0.0
    model = model.to(device)
    scaler = GradScaler()

    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FASTRCNN_WEIGHTS_DIR, exist_ok=True)
    best_save_path = os.path.join(FASTRCNN_WEIGHTS_DIR, "best.pth")
    best_last_path = os.path.join(FASTRCNN_WEIGHTS_DIR, "last.pth")

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', extended_summary=True)
    for epoch in range(num_epochs):
        model.train()
        start = time.perf_counter()
        epoch_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=False)
        for images, targets in train_pbar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)  # {loss1: tensor1, loss2:tensor2,..}
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += losses.item() * len(images)
        avg_loss = epoch_loss / len(train_loader.dataset)

        map50, map5095 = evaluate(model, val_loader, test=False, metric=metric)

        end = time.perf_counter()
        train_time = ((end - start) / 60)
        history["total_time"] += train_time

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(map5095)
            else:
                scheduler.step()

        history["training_loss"].append(avg_loss)
        history["map50"].append(map50)
        history["map5095"].append(map5095)
        print(f"Epoch {epoch + 1}/{num_epochs} - {train_time:.2f}m: Loss={avg_loss:.4f} | mAP@50={map50:.4f} | mAP@50:95={map5095:.4f}")

        checkpoints = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch}
        if map5095 > best_map_all:
            best_map_all = map5095
            torch.save(checkpoints, best_save_path)
        torch.save(checkpoints, best_last_path)

    print("Training successfully!")
    print(f"Spent {history['total_time']:.4f} minutes to train.")
    print(f"Model weight is saved at directory {FASTRCNN_WEIGHTS_DIR}.")
    return history


def visualize_metrics(history, figsize=(12, 6), sp=None):
    train_loss = history["training_loss"]
    map50 = history["map50"]
    map5095 = history["map5095"]
    nums_epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(nums_epochs, train_loss, linewidth=3, color="red")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss Values")

    ax[1].plot(nums_epochs, map50, linewidth=3, color="red")
    ax[1].plot(nums_epochs, map5095, linewidth=3, color="blue")
    ax[1].set_title("mAP@50 & map50:95")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("mAP Values")
    ax[1].legend(["mAP@50", "mAP@50:95"])

    if sp is not None:
        fig.savefig(os.path.join(sp, "report_figure.png"))
    plt.tight_layout()
    plt.show()


        

                
