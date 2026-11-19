# =============================================
# BLOCK 3 — TESTING
# =============================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

test_data = datasets.ImageFolder(test_dir, transform=val_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model.load_state_dict(torch.load(os.path.join(save_dir, "best_swin_t.pth")))
model.eval()

all_labels, all_preds = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
f1s  = f1_score(all_labels, all_preds)

print("\n=== TEST RESULTS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1s:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
plt.close()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=test_data.classes)
with open(os.path.join(save_dir, "test_classification_report.txt"), "w") as f:
    f.write(report)
print(report)