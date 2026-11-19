#-----------------------Mounting
# from google.colab import drive
# drive.mount('/content/drive')

#----------------------Dataset Extraction
# import zipfile
# import os
# zip_path = '/content/drive/MyDrive/Classification Dataset (Split).zip'   # <-- your zip file path
# extract_path = '/content/Classification_Dataset'
# os.makedirs(extract_path, exist_ok=True)
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)
# print("✅ Dataset extracted to:", extract_path)

#-------------------Installations
# !nvidia-smi
# !pip install torch torchvision torchaudio --quiet
# !pip install scikit-learn matplotlib
# !pip install seaborn --quiet

#-----------------Device Check
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("✅ Using device:", device)

#-----------------------Dataset Paths
# root = "/content/Classification_Dataset"
# print("Root folder contents:", os.listdir(root))
# for f in os.listdir(root):
#     path = os.path.join(root, f)
#     if os.path.isdir(path):
#         print(f"Contents of {f}:", os.listdir(path))

# # After confirming structure, set dataset paths
# base_dir = os.path.join(root, "Classification Dataset (Split)")  # adjust name if needed
# train_dir = os.path.join(base_dir, "train")
# val_dir   = os.path.join(base_dir, "val")
# test_dir  = os.path.join(base_dir, "test")

# save_dir  = "/content/drive/MyDrive/Classification_Results"
# os.makedirs(save_dir, exist_ok=True)

#--------------------------------------------------------------------------------------------

# =============================================
# BLOCK 1 — SWIN-T TRAINING AND VALIDATION
# =============================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import os

# ---------------- CONFIGURATION ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = 2
batch_size = 32
num_epochs = 30
learning_rate = 1e-4

# ---------------- TRANSFORMS ----------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- DATASETS ----------------
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data   = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# ---------------- MODEL ----------------
model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------- TRAINING LOOP ----------------
patience = 5
no_improve = 0

train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_correct, train_total, running_loss = 0, 0, 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = running_loss / len(train_data)
    train_acc = train_correct / train_total

    # ---- VALIDATION ----
    model.eval()
    val_correct, val_total, val_running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / len(val_data)
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] — "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    # ---- SAVE BEST MODEL + EARLY STOPPING TRACKER ----
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), os.path.join(save_dir, "best_swin_t.pth"))
        best_val_acc = val_acc
        no_improve = 0   # reset counter since model improved
    else:
        no_improve += 1  # no improvement this epoch

    # ---- EARLY STOPPING CONDITION ----
    if no_improve >= patience:
        print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
        break

print("\nTraining complete. Best validation accuracy:", best_val_acc)