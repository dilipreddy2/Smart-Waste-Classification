
import os
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras import layers, models

# ----------------------------------
# 1. SETTINGS
# ----------------------------------

DATASET_PATH = "dataset"
MODEL_PATH = "model/best_waste_model_backup.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

os.makedirs("model", exist_ok=True)

print("\nSMART WASTE CLASSIFICATION")
print("IMPROVED MODEL TRAINING")
print("=" * 50)

# ----------------------------------
# 2. LOAD TRAINING DATASET
# ----------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ----------------------------------
# 3. LOAD VALIDATION DATASET
# ----------------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Number of classes:", num_classes)

# ----------------------------------
# 4. DATA AUGMENTATION
# ----------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# ----------------------------------
# 5. PREPROCESS IMAGES
# ----------------------------------

def prepare_training(images, labels):
    images = data_augmentation(images, training=True)
    images = preprocess_input(images)
    return images, labels


def prepare_validation(images, labels):
    images = preprocess_input(images)
    return images, labels


train_dataset = train_dataset.map(
    prepare_training,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    prepare_validation,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# ----------------------------------
# 6. CALCULATE CLASS WEIGHTS
# ----------------------------------

print("\nCalculating class weights...")

all_labels = []

for _, labels in tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
):
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=all_labels
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

print("Class weights:", class_weights)

# ----------------------------------
# 7. LOAD PREVIOUS MODEL
# ----------------------------------

print("\nLoading existing model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Existing model loaded successfully!")

# ----------------------------------
# 8. FIND MOBILENETV2 BASE MODEL
# ----------------------------------

base_model = next(
    (
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    ),
    None
)

if base_model is None:
    raise ValueError("MobileNetV2 base model was not found!")

# ----------------------------------
# 9. ENABLE FINE-TUNING
# ----------------------------------

base_model.trainable = True

fine_tune_at = max(0, len(base_model.layers) - 40)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("Fine-tuning enabled!")

print(
    "Trainable layers:",
    sum(layer.trainable for layer in base_model.layers)
)

# ----------------------------------
# 10. COMPILE MODEL
# ----------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.000005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------
# 11. CALLBACKS
# ----------------------------------

checkpoint = ModelCheckpoint(
    "model/best_waste_model_improved.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# ----------------------------------
# 12. TRAIN MODEL
# ----------------------------------

print("\nStarting improved model training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ----------------------------------
# 13. SAVE FINAL MODEL
# ----------------------------------

model.save("model/waste_classifier_improved.keras")

# ----------------------------------
# 14. COMPLETION
# ----------------------------------

print("\nImproved training completed!")

print("Best model:")
print("model/best_waste_model_improved.keras")

print("Final model:")
print("model/waste_classifier_improved.keras")

print("\nAll six categories were included in training.")
print("=" * 50)
