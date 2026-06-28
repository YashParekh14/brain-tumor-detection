"""
STEP : MODEL TRAINING

This trains an EfficientNet model on brain MRI images.
It classifies tumors into 4 categories:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

HOW TO RUN:
  python model/train.py

WHAT IT DOES:
  1. Loads MRI images from /data folder
  2. Applies augmentation (flips, rotations) to improve accuracy
  3. Fine-tunes EfficientNet (pretrained on ImageNet)
  4. Saves the best model to model/best_model.pth
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─── CONFIG ──────────
DATA_DIR    = "data/brain_tumor_dataset"   # Kaggle dataset goes here
MODEL_SAVE  = "model/best_model.pth"
NUM_CLASSES = 4
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ─── STEP 1: DATA LOADING & AUGMENTATION ───────────
# Augmentation = artificially creating more training examples
# by flipping, rotating images slightly
# This stops the model from overfitting (memorising instead of learning)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize all images to 224x224
    transforms.RandomHorizontalFlip(),       # Randomly flip image left-right
    transforms.RandomRotation(15),           # Randomly rotate by up to 15 degrees
    transforms.ColorJitter(brightness=0.2),  # Slightly change brightness
    transforms.ToTensor(),                   # Convert image to tensor (numbers)
    transforms.Normalize(                    # Normalize pixel values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_data():
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "Training"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "Testing"),
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ Training samples:   {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Classes: {train_dataset.classes}")

    return train_loader, val_loader


# ─── STEP 2: MODEL SETUP ──────────
# EfficientNet was pretrained on ImageNet (1.2M images, 1000 classes)
# We replace its final layer to output 4 classes instead of 1000
# This is called Transfer Learning - reusing a model trained on similar data

def build_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # Load pretrained weights

    # Freeze early layers (they already learned useful features like edges, shapes)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classifier for our 4-class problem
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model.to(DEVICE)


# ─── STEP 3: TRAINING LOOP ────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()          # Clear previous gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()               # Backpropagation
        optimizer.step()              # Update weights

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():  # No gradient calculation needed for validation
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy, all_preds, all_labels


# ─── STEP 4: SAVE CONFUSION MATRIX ────────────────

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png")
    print("✅ Confusion matrix saved to model/confusion_matrix.png")


# ─── MAIN ───────────────
def main():
    print(f"\n🧠 Brain Tumor Classifier Training")
    print(f"📱 Device: {DEVICE}\n")

    train_loader, val_loader = load_data()
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc, preds, labels = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  💾 New best model saved! Val Acc: {val_acc:.4f}")

    print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    plot_confusion_matrix(labels, preds)


if __name__ == "__main__":
    main()
