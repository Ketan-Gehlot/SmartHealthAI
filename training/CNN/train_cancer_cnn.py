# ==========================================================
# SmartHealthAI - Breast Cancer Image CNN Training
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_DIR = os.path.join(
    BASE_DIR, "data", "raw", "breast_cancer_images"
)

MODEL_SAVE_DIR = os.path.join(
    BASE_DIR, "models", "cnn_models"
)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "breast_cancer_cnn.h5")

# ==========================================================
# IMAGE PARAMETERS
# ==========================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30

# ==========================================================
# DATA AUGMENTATION & LOADING
# ==========================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ==========================================================
# CNN MODEL ARCHITECTURE
# ==========================================================

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall()]
)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)

# ==========================================================
# TRAINING
# ==========================================================

print("\n[INFO] Starting CNN training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

model.save(MODEL_PATH)

print("\n[SUCCESS] CNN Model saved at:")
print(MODEL_PATH)

# ==========================================================
# TRAINING VISUALIZATION
# ==========================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
