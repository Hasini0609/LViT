# =============================================
# BLOCK 2 — METRICS AND VISUALIZATION
# =============================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import numpy as np

model.load_state_dict(torch.load(os.path.join(save_dir, "best_swin_t.pth")))
model.eval()

# ---------------- FINAL VALIDATION EVALUATION ----------------
all_labels, all_preds = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

precision = precision_score(all_labels, all_preds)
recall    = recall_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds)
acc       = np.mean(np.array(all_preds) == np.array(all_labels))

print("\nFINAL VALIDATION METRICS:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ---------------- GRAPHS ----------------
# Accuracy vs Epoch
plt.figure()
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(val_acc_list, label="Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Accuracy vs Epoch")
plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
plt.close()

# Loss vs Epoch
plt.figure()
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Loss vs Epoch")
plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
plt.close()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

# Individual bar charts
plt.figure()
sns.barplot(x=["F1 Score"], y=[f1], palette="pastel")
plt.ylim(0,1); plt.title("Validation F1 Score")
plt.savefig(os.path.join(save_dir, "f1_score_bar.png"))
plt.close()

plt.figure()
sns.barplot(x=["Precision"], y=[precision], palette="muted")
plt.ylim(0,1); plt.title("Validation Precision")
plt.savefig(os.path.join(save_dir, "precision_bar.png"))
plt.close()

plt.figure()
sns.barplot(x=["Recall"], y=[recall], palette="Set2")
plt.ylim(0,1); plt.title("Validation Recall")
plt.savefig(os.path.join(save_dir, "recall_bar.png"))
plt.close()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=train_data.classes)
with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write(report)
print("\nClassification Report:\n", report)