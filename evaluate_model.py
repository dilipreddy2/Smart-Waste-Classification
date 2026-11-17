
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

print("\nSMART WASTE CLASSIFICATION")
print("IMPROVED MODEL EVALUATION")
print("=" * 50)

# Load improved model
model = tf.keras.models.load_model(
    "model/best_waste_model_improved.keras"
)

# Load class names
with open("model/class_names.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

print("Classes:", class_names)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.30,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=16,
    shuffle=True
)

# Apply MobileNetV2 preprocessing
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

validation_dataset = validation_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

# Collect actual labels and predictions together
actual_labels = []
predicted_labels = []

for images, labels in validation_dataset:
    predictions = model.predict(images, verbose=0)

    actual_labels.extend(labels.numpy())
    predicted_labels.extend(np.argmax(predictions, axis=1))

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)

print("\nOverall Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nCLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

print("\nCONFUSION MATRIX")
print(cm)

# Display confusion matrix
plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Improved Model - Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()

plt.savefig("model/confusion_matrix_improved.png")
plt.show()

print("\nConfusion matrix saved successfully!")
print("=" * 50)
