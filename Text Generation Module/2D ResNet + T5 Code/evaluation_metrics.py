import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- Save Directory --------------------
SAVE_DIR = "/content/drive/MyDrive/Medical_Image_Text_Model"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- Convert to arrays --------------------
train_losses_arr = np.array(train_losses)
val_losses_arr = np.array(val_losses)
train_accs_arr = np.array(train_accs)
val_accs_arr = np.array(val_accs)
bleu_arr = np.array(bleu_scores)
rouge_l_arr = np.array(rouge_l_scores)

# -------------------- Epochs --------------------
epochs = np.arange(1, len(train_losses) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, train_losses_arr, label='Train Loss', marker='o')
plt.plot(epochs, val_losses_arr, label='Val Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs, train_accs_arr*100, label='Train Token Accuracy', marker='o')
plt.plot(epochs, val_accs_arr*100, label='Val Token Accuracy', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Token Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "token_accuracy_plot.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs, bleu_arr, label='BLEU Score', marker='o')
plt.plot(epochs, rouge_l_arr, label='ROUGE-L Score', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("BLEU & ROUGE-L Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bleu_rouge_plot.png"))
plt.show()

np.savez(
    os.path.join(SAVE_DIR, "training_metrics.npz"),
    train_losses=train_losses_arr,
    val_losses=val_losses_arr,
    train_accs=train_accs_arr,
    val_accs=val_accs_arr,
    bleu_scores=bleu_arr,
    rouge_l_scores=rouge_l_arr
)

print("✅ All plots and metrics saved to Google Drive.")

