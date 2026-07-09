import torch
import argparse
from alpr.detector.net import Model
from alpr.detector import helper_utils
from alpr.detector.dataset import LicensePlateDataset
from alpr.detector.config import MODEL_OBJECT
from alpr.paths import DETECT_DATA_DIR, FIGURE_OUTPUT_DIR, DETECTOR_CHECKPOINT_DIR

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=str(DETECT_DATA_DIR / "train"))
    parser.add_argument("--val_path", type=str, default=str(DETECT_DATA_DIR / "valid"))
    parser.add_argument("--test_path", type=str, default=str(DETECT_DATA_DIR / "test"))
    parser.add_argument("--model", type=str, choices=list(MODEL_OBJECT.keys()), default="mobile_net_large")

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()

    lr = args.lr
    num_classes = args.num_classes
    num_epochs = args.epochs
    momentum = args.momentum
    weight_decay = args.weight_decay
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    batch_size = args.batch_size

    train_transform, val_transform = helper_utils.define_transforms()
    train_dataset = LicensePlateDataset(train_path, train_transform)
    val_dataset = LicensePlateDataset(val_path, val_transform)
    test_dataset = LicensePlateDataset(test_path, val_transform)
    train_loader, val_loader, test_loader = helper_utils.create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)
    print(f"Training set size:        {len(train_dataset)}")
    print(f"Validation set size:      {len(val_dataset)}")
    print(f"Test set size:            {len(test_dataset)}")

    model_object = MODEL_OBJECT[args.model]
    print(f"Model {args.model} is training on {device}.")
    model = Model(model_object, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    history = helper_utils.train(model=model, train_loader=train_loader, val_loader=val_loader,
                                 optimizer=optimizer, num_epochs=num_epochs, scheduler=scheduler)

    model_best = helper_utils.load_model_state_dict(model_object, num_classes, DETECTOR_CHECKPOINT_DIR)
    test_map_50, test_map_5095 = helper_utils.evaluate(model_best, test_loader, test=True)
    history["test_map_50"] = test_map_50
    history["test_map_5095"] = test_map_5095
    helper_utils.save_history(history)
    helper_utils.visualize_metrics(history, sp=FIGURE_OUTPUT_DIR)

if __name__ == "__main__":
    main()


