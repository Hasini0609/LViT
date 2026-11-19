import torch
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Paths --------------------
SAVE_DIR = "/content/drive/MyDrive/Medical_Image_Text_Model/Test_Evaluation"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = "/content/drive/MyDrive/Medical_Image_Text_Model/best_model.pth"

# -------------------- Load Dataset --------------------
test_dataset = MedicalScanDataset(
    image_dir="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/test/Infected",
    annotation_file="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/test/annotations.csv",
    tokenizer=tokenizer,
    is_train=False
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize models (MUST match training exactly)
encoder = ResNet2DEncoder(feature_dim=512).to(device)

decoder = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

projector = VisualProjector(
    visual_dim=512,
    hidden_dim=decoder.config.d_model,  # MUST match T5
    seq_len=32                          # MUST match training
).to(device)

# Load best saved model
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
projector.load_state_dict(checkpoint["projector"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
projector.eval()
decoder.eval()

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

test_losses = []
total_correct = 0
total_tokens = 0

references = []
hypotheses = []

with torch.no_grad():
    for images, input_ids, attention_mask in test_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)

        mask = input_ids != tokenizer.pad_token_id

        features = encoder(images)
        visual_tokens = projector(features)

        outputs = decoder(
            inputs_embeds=visual_tokens,
            labels=input_ids,
            attention_mask=mask
        )

        test_losses.append(outputs.loss.item())

        preds = outputs.logits.argmax(-1)
        total_correct += ((preds == input_ids) & mask).sum().item()
        total_tokens += mask.sum().item()

        for i in range(input_ids.size(0)):
            ref = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            pred_tokens = preds[i][:input_ids.size(1)]
            hyp = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            references.append(ref)
            hypotheses.append(hyp)

# Final metrics
test_loss = np.mean(test_losses)
test_acc = total_correct / total_tokens
test_bleu = np.mean([
    sentence_bleu([r.split()], h.split(), smoothing_function=smoothie)
    for r, h in zip(references, hypotheses)
])
test_rouge_l = np.mean([
    rouge.score(r, h)["rougeL"].fmeasure
    for r, h in zip(references, hypotheses)
])

print(f"\n✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Token Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test BLEU Score: {test_bleu:.4f}")
print(f"✅ Test ROUGE-L Score: {test_rouge_l:.4f}")

# Convert lists to arrays
test_losses_arr = np.array(test_losses)
test_accs_arr = np.array([test_acc] * len(test_losses_arr))
bleu_arr = np.array([test_bleu] * len(test_losses_arr))
rouge_l_arr = np.array([test_rouge_l] * len(test_losses_arr))

batches = np.arange(1, len(test_losses_arr) + 1)

# ---------------- Loss Plot ----------------
plt.figure(figsize=(8,5))
plt.plot(batches, test_losses_arr, label='Test Loss', marker='o')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Test Loss per Batch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_loss.png"))
plt.show()

# ---------------- Token Accuracy ----------------
plt.figure(figsize=(8,5))
plt.plot(batches, test_accs_arr * 100, label='Test Token Accuracy', marker='o')
plt.xlabel("Batch")
plt.ylabel("Accuracy (%)")
plt.title("Test Token Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_accuracy.png"))
plt.show()

# ---------------- BLEU & ROUGE ----------------
plt.figure(figsize=(8,5))
plt.plot(batches, bleu_arr, label='BLEU Score', marker='o')
plt.plot(batches, rouge_l_arr, label='ROUGE-L Score', marker='o')
plt.xlabel("Batch")
plt.ylabel("Score")
plt.title("Test BLEU & ROUGE-L Scores")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_bleu_rouge.png"))
plt.show()

np.savez(
    os.path.join(SAVE_DIR, "test_metrics.npz"),
    test_losses=test_losses_arr,
    test_accuracy=test_accs_arr,
    bleu_scores=bleu_arr,
    rouge_l_scores=rouge_l_arr
)

print("✅ Test plots and metrics saved to Google Drive.")


import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
from transformers import T5ForConditionalGeneration

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load Dataset --------------------
test_dataset = MedicalScanDataset(
    image_dir="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/test/Infected",
    annotation_file="/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/test/annotations.csv",
    tokenizer=tokenizer,
    is_train=False
)

# -------------------- Load Best Model --------------------
encoder = ResNet2DEncoder(feature_dim=512).to(device)
# Make sure projector matches training: hidden_dim=512, seq_len=32
projector = VisualProjector(visual_dim=512, hidden_dim=512, seq_len=32).to(device)
decoder = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

checkpoint = torch.load(
    "/content/drive/MyDrive/Medical_Image_Text_Model/best_model.pth",
    map_location=device
)
encoder.load_state_dict(checkpoint["encoder"])
projector.load_state_dict(checkpoint["projector"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
projector.eval()
decoder.eval()

print("✅ Model loaded successfully and ready for testing")


def generate_text(image_tensor, max_len=32):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim
        features = encoder(image_tensor)
        visual_tokens = projector(features)
        generated_ids = decoder.generate(
            inputs_embeds=visual_tokens,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text

sample_indices = random.sample(range(len(test_dataset)), 10)

for idx in sample_indices:
    image, input_ids, _ = test_dataset[idx]

    orig_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    gen_text = generate_text(image)

    # Convert tensor to image for display
    img = image.permute(1, 2, 0).cpu().numpy()  # ✅ FIXED
    img = (img * 0.5 + 0.5)                     # Denormalize
    img = (img * 255).astype("uint8")

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Original: {orig_text}\nGenerated: {gen_text}",
        fontsize=10
    )
    plt.show()

import torch
import pandas as pd
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Paths --------------------
SAVE_DIR = "/content/drive/MyDrive/Medical_Image_Text_Model/Generated_Annotations"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = "/content/drive/MyDrive/Medical_Image_Text_Model/best_model.pth"

import torch
from transformers import T5ForConditionalGeneration

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load Models --------------------
encoder = ResNet2DEncoder(feature_dim=512).to(device)
projector = VisualProjector(visual_dim=512, hidden_dim=512, seq_len=32).to(device)  # Must match training
decoder = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# -------------------- Load Checkpoint --------------------
BEST_MODEL_PATH = "/content/drive/MyDrive/Medical_Image_Text_Model/best_model.pth"
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

encoder.load_state_dict(checkpoint['encoder'])
projector.load_state_dict(checkpoint['projector'])
decoder.load_state_dict(checkpoint['decoder'])

# -------------------- Set Eval Mode --------------------
encoder.eval()
projector.eval()
decoder.eval()

print("✅ Model loaded successfully")

def generate_text(image_tensor, max_len=32):
    """Generate text from image tensor using trained model."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim
        features = encoder(image_tensor)
        visual_tokens = projector(features)
        generated_ids = decoder.generate(
            inputs_embeds=visual_tokens,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text

def generate_annotations_csv(dataset, csv_filename):
    results = []
    for i in tqdm(range(len(dataset)), desc=f"Generating {csv_filename}"):
        image_tensor, _, _ = dataset[i]
        img_name = dataset.annotations[i]['Image']        # uppercase I
        generated_text = generate_text(image_tensor)       # your function
        results.append({"Image": img_name, "Generated_Description": generated_text})

    df = pd.DataFrame(results)
    csv_path = os.path.join(SAVE_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_filename} to {SAVE_DIR}")

# Assuming you have already loaded these datasets
train_dataset = MedicalScanDataset("/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/train/Infected", "/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/train/annotations.csv", tokenizer, is_train=False)
val_dataset = MedicalScanDataset("/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/val/Infected", "/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/val/annotations.csv", tokenizer, is_train=False)
test_dataset = MedicalScanDataset("/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/test/Infected", "/content/Text_Gen_Dataset/Anomaly & Text Dataset (Split)/test/annotations.csv", tokenizer, is_train=False)

# Generate CSVs
generate_annotations_csv(train_dataset, "train_generated_annotations.csv")
generate_annotations_csv(val_dataset, "val_generated_annotations.csv")
generate_annotations_csv(test_dataset, "test_generated_annotations.csv")
