
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Dataset location
dataset_path = "dataset"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

print("\nSMART WASTE CLASSIFICATION")
print("=" * 45)

# Load training dataset (70%)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset (30%)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Display class names
class_names = train_dataset.class_names

print("\nClasses:", class_names)
print("Number of classes:", len(class_names))

# Apply MobileNetV2 preprocessing
train_dataset = train_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

# Improve dataset loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

print("\nMobileNetV2 preprocessing completed!")
print("Training images: 1769")
print("Validation images: 758")
print("Image size: 224 x 224")
print("Batch size:", BATCH_SIZE)

print("\nDataset preparation completed successfully!")
print("=" * 45)
