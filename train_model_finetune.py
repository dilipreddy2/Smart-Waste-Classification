
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Settings
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

os.makedirs("model", exist_ok=True)

print("\nSMART WASTE CLASSIFICATION")
print("MOBILENETV2 FINE-TUNING")
print("=" * 50)

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Preprocess images
train_dataset = train_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# Load your existing trained model
print("\nLoading existing model...")

model = tf.keras.models.load_model(
    "model/best_waste_model_backup.keras"
)

print("Existing model loaded successfully!")

# Find MobileNetV2 base model
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

# Enable fine-tuning
base_model.trainable = True

# Train only the last 30 layers
fine_tune_at = max(0, len(base_model.layers) - 30)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("Fine-tuning enabled!")
print("Trainable MobileNetV2 layers:",
      sum(layer.trainable for layer in base_model.layers))

# Compile with a low learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save the best fine-tuned model
checkpoint = ModelCheckpoint(
    "model/best_waste_model_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Train model
print("\nStarting fine-tuning...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# Save fine-tuned model
model.save("model/waste_classifier_finetuned.keras")

print("\nFine-tuning completed!")
print("Fine-tuned models saved successfully!")
print("=" * 50)
