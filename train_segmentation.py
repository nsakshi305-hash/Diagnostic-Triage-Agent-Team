import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import json
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

ROOT = Path("Kvasir-SEG/Kvasir-SEG")
IMG_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"
TRAIN_TXT = Path("train.txt")
VAL_TXT = Path("val.txt")
OUT_DIR = Path("outputs/trained_segmentation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "deeplabv3_mobilenetv3_kvasir.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS_FROZEN = 4
EPOCHS_FINETUNE = 3
NUM_WORKERS = 0
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def read_split(path):
    if path.exists():
        names = [x.strip() for x in path.read_text(errors="ignore").splitlines() if x.strip()]
        names = [Path(x).name for x in names]
        names = [x for x in names if (IMG_DIR / x).exists()]
        if names:
            return names
    return sorted(x.name for x in IMG_DIR.glob("*.jpg"))

class KvasirSegDataset(Dataset):
    def __init__(self, names, train=False):
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = cv2.imread(str(IMG_DIR / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(MASK_DIR / name), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Could not load pair: {name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        mask = torch.from_numpy((mask > 0).astype(np.int64))
        return image, mask

def build_model():
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    # Keep pretrained DeepLab feature/ASPP layers; replace only final classifier output.
    model.classifier = DeepLabHead(960, 2)
    return model

def dice_iou(logits, target):
    pred = torch.argmax(logits, dim=1)
    pred_b = pred.bool()
    target_b = target.bool()
    inter = (pred_b & target_b).sum().item()
    pred_sum = pred_b.sum().item()
    target_sum = target_b.sum().item()
    union = (pred_b | target_b).sum().item()
    dice = (2 * inter + 1e-7) / (pred_sum + target_sum + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice, iou

def run_epoch(model, loader, optimizer, criterion, train):
    model.train(train)
    total_loss = 0.0
    dices, ious = [], []

    for images, masks in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)

        out = model(images)["out"]
        loss = criterion(out, masks)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        d, i = dice_iou(out.detach(), masks)
        dices.append(d)
        ious.append(i)

    return (
        total_loss / max(1, len(loader)),
        float(np.mean(dices)),
        float(np.mean(ious)),
    )

def main():
    train_names = read_split(TRAIN_TXT)
    val_names = read_split(VAL_TXT)

    # If train.txt/val.txt are absent or unusable, make a deterministic 80/20 split.
    if not TRAIN_TXT.exists() or not VAL_TXT.exists():
        names = sorted(x.name for x in IMG_DIR.glob("*.jpg"))
        rng = random.Random(SEED)
        rng.shuffle(names)
        cut = int(0.8 * len(names))
        train_names, val_names = names[:cut], names[cut:]

    print(f"Training images: {len(train_names)}")
    print(f"Validation images: {len(val_names)}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Loading pretrained MobileNetV3-DeepLabV3 weights...")

    train_ds = KvasirSegDataset(train_names, train=True)
    val_ds = KvasirSegDataset(val_names, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]))

    # Phase 1: train the newly initialized final segmentation layer.
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_dice = -1.0
    history = []

    for epoch in range(1, EPOCHS_FROZEN + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, True)
        va = run_epoch(model, val_loader, optimizer, criterion, False)
        row = {"phase":"head", "epoch":epoch, "train_loss":tr[0], "val_loss":va[0], "val_dice":va[1], "val_iou":va[2]}
        history.append(row)
        print(f"[Head {epoch}/{EPOCHS_FROZEN}] loss={tr[0]:.4f} val_loss={va[0]:.4f} Dice={va[1]:.4f} IoU={va[2]:.4f}")
        if va[1] > best_dice:
            best_dice = va[1]
            torch.save({"model_state": model.state_dict(), "image_size": IMAGE_SIZE, "best_dice": best_dice, "best_iou": va[2]}, CHECKPOINT)

    # Phase 2: fine-tune the last part of the pretrained backbone.
    for p in model.backbone.parameters():
        p.requires_grad = True
    if hasattr(model.backbone, "features"):
        for p in model.backbone.features[:-3].parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, True)
        va = run_epoch(model, val_loader, optimizer, criterion, False)
        row = {"phase":"finetune", "epoch":epoch, "train_loss":tr[0], "val_loss":va[0], "val_dice":va[1], "val_iou":va[2]}
        history.append(row)
        print(f"[Fine {epoch}/{EPOCHS_FINETUNE}] loss={tr[0]:.4f} val_loss={va[0]:.4f} Dice={va[1]:.4f} IoU={va[2]:.4f}")
        if va[1] > best_dice:
            best_dice = va[1]
            torch.save({"model_state": model.state_dict(), "image_size": IMAGE_SIZE, "best_dice": best_dice, "best_iou": va[2]}, CHECKPOINT)

    (OUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
