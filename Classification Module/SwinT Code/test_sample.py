# =============================================
# BLOCK 4 — RANDOM TEST SAMPLE VISUALIZATION
# =============================================
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Make sure model is in evaluation mode
model.eval()

# Class names (update these according to your dataset folder names)
class_names = test_data.classes

# Randomly select 5 samples from the test set
num_samples = 5
indices = random.sample(range(len(test_data)), num_samples)

# Disable gradient computation for inference
with torch.no_grad():
    for i, idx in enumerate(indices):
        image, label = test_data[idx]
        input_img = image.unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass
        output = model(input_img)
        _, pred = torch.max(output, 1)

        # Convert tensors for display
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize for display

        # Plot the image and predictions
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(
            f"True: {class_names[label]} | Predicted: {class_names[pred.item()]}",
            fontsize=12, color="blue" if label == pred.item() else "red"
        )
        plt.show()