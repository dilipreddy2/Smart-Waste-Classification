
import os

# Get the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
dataset_path = os.path.join(BASE_DIR, "dataset")

# Waste categories
categories = [
    "trash",
    "plastic",
    "paper",
    "metal",
    "glass",
    "cardboard"
]

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total_images = 0

print("\nSMART WASTE CLASSIFICATION")
print("=" * 45)

for category in categories:

    folder_path = os.path.join(dataset_path, category)

    if not os.path.isdir(folder_path):
        print(f"{category.capitalize():15}: Folder not found!")
        continue

    images = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path) and file.lower().endswith(image_extensions):
            images.append(file)

    count = len(images)
    total_images += count

    print(f"{category.capitalize():15}: {count} images")

print("=" * 45)
print("Total Images:", total_images)
print("=" * 45)
