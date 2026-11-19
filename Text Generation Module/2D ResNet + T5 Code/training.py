# from google.colab import drive
# drive.mount('/content/drive')
# -------------------------------------------------------------------------------------------
# import zipfile

# # Update with your actual path if different
# zip_path = '/content/drive/MyDrive/Anomaly & Text Dataset (Split).z'
# extract_path = '/content/Text_Gen_Dataset/'  # You can change this if you like

# # Unzip
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

# print("Files extracted to:", extract_path)
# ------------------------------------------------------------------------------------------
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# ------------------------------------------------------------------------------------------
# !pip install rouge-score
# ------------------------------------------------------------------------------------------
import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MedicalScanDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer, max_len=32, is_train=True):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

        # -------------------- Load CSV --------------------
        self.annotations = pd.read_csv(annotation_file).to_dict(orient="records")

        # -------------------- Transforms --------------------
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        img_path = os.path.join(self.image_dir, item["Image"])
        image = Image.open(img_path).convert("L")     # grayscale
        image = image.convert("RGB")                  # replicate to 3 channels
        image = self.transform(image)

        encoding = self.tokenizer(
            item["Description"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return image, encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()

class ResNet2DEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = torch.flatten(self.backbone(x), 1)  # Flattens (B, 512, 1, 1) -> (B, 512)
        return self.fc(x)

class VisualProjector(nn.Module):
    def __init__(self, visual_dim=512, hidden_dim=512, seq_len=8):
        super().__init__()
        self.fc = nn.Linear(visual_dim, hidden_dim * seq_len)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.fc(x)
        return x.view(x.size(0), self.seq_len, self.hidden_dim)

from torch.utils.data import DataLoader

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# -------------------- Train Dataset --------------------
train_dataset = MedicalScanDataset(
    image_dir="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/train/Infected",
    annotation_file="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/train/annotations.csv",
    tokenizer=tokenizer,
    is_train=True
)

# -------------------- Validation Dataset --------------------
val_dataset = MedicalScanDataset(
    image_dir="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/val/Infected",
    annotation_file="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/val/annotations.csv",
    tokenizer=tokenizer,
    is_train=False
)

# -------------------- DataLoaders --------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


encoder = ResNet2DEncoder().to(device)
projector = VisualProjector().to(device)
decoder = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(projector.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
bleu_scores, rouge_l_scores = [], []

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

SAVE_DIR = "/content/drive/MyDrive/Medical_Image_Text_Model"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")

best_val_loss = float("inf")
epochs = 30

# ----------------- Setup -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure projector hidden_dim matches T5 embedding
projector = VisualProjector(visual_dim=512, hidden_dim=decoder.config.d_model, seq_len=32).to(device)

encoder = ResNet2DEncoder(feature_dim=512).to(device)
decoder = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(projector.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)

# Metric storage
train_losses, val_losses = [], []
train_accs, val_accs = [], []
bleu_scores, rouge_l_scores = [], []

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

SAVE_DIR = "/content/drive/MyDrive/Medical_Image_Text_Model"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
best_val_loss = float("inf")

# Early stopping
patience = 5
no_improve_epochs = 0

# ----------------- Training Loop -----------------
epochs = 30

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # ----- TRAIN -----
    encoder.train()
    projector.train()
    decoder.train()

    total_loss, correct, total = 0, 0, 0

    for images, input_ids, attention_mask in train_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        mask = input_ids != tokenizer.pad_token_id

        features = encoder(images)
        visual_tokens = projector(features)

        outputs = decoder(inputs_embeds=visual_tokens, labels=input_ids, attention_mask=mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.logits.argmax(-1)
        correct += ((preds == input_ids) & mask).sum().item()
        total += mask.sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ----- VALIDATION -----
    encoder.eval()
    projector.eval()
    decoder.eval()

    val_loss, correct, total = 0, 0, 0
    refs, hyps = [], []

    with torch.no_grad():
        for images, input_ids, attention_mask in val_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            mask = input_ids != tokenizer.pad_token_id

            features = encoder(images)
            visual_tokens = projector(features)

            outputs = decoder(inputs_embeds=visual_tokens, labels=input_ids, attention_mask=mask)
            val_loss += outputs.loss.item()

            preds = outputs.logits.argmax(-1)
            correct += ((preds == input_ids) & mask).sum().item()
            total += mask.sum().item()

            # Decode for BLEU/ROUGE
            for i in range(input_ids.size(0)):
                pred_tokens = preds[i][:input_ids.size(1)]
                refs.append(tokenizer.decode(input_ids[i], skip_special_tokens=True))
                hyps.append(tokenizer.decode(pred_tokens, skip_special_tokens=True))

    val_loss /= len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Compute BLEU and ROUGE-L
    bleu = np.mean([sentence_bleu([r.split()], h.split(), smoothing_function=smoothie)
                    for r, h in zip(refs, hyps)])
    rouge_l = np.mean([rouge.score(r, h)["rougeL"].fmeasure
                       for r, h in zip(refs, hyps)])

    bleu_scores.append(bleu)
    rouge_l_scores.append(rouge_l)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
    print(f"BLEU: {bleu:.4f} | ROUGE-L: {rouge_l:.4f}")

    # ----- SAVE BEST MODEL -----
    if val_loss < best_val_loss:
        if os.path.exists(BEST_MODEL_PATH):
            os.remove(BEST_MODEL_PATH)

        torch.save({
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": epoch+1,
            "val_loss": val_loss
        }, BEST_MODEL_PATH)

        best_val_loss = val_loss
        no_improve_epochs = 0
        print("✔ Best model saved")
    else:
        no_improve_epochs += 1
        print(f"⚠ No improvement for {no_improve_epochs} epoch(s)")

    # ----- EARLY STOPPING CHECK -----
    if no_improve_epochs >= patience:
        print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
        break

