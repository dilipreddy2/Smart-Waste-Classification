
import os
import matplotlib.pyplot as plt
from PIL import Image

# Dataset location
dataset_path = "dataset"

# Categories
categories = [
    "trash",
    "plastic",
    "paper",
    "metal",
    "glass",
    "cardboard"
]

# Create visualization
plt.figure(figsize=(14, 8))

for i, category in enumerate(categories):

    folder_path = os.path.join(dataset_path, category)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if len(images) > 0:

        image_path = os.path.join(folder_path, images[0])

        img = Image.open(image_path).convert("RGB")

        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(category.capitalize())
        plt.axis("off")

plt.tight_layout()
plt.show()
