import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
import numpy as np
import os

# ---------------- Configuration ----------------
IMG_SIZE = 224       # Target image size for the model
BATCH_SIZE = 32      # Number of images per batch
MODEL_PATH = "flower_model.h5"  # Path to save/load the trained model

# ---------------- Load Dataset ----------------
# Load Oxford Flowers 102 dataset and split into train, validation, test
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    "oxford_flowers102",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],  # 80% train, 10% val, 10% test
    with_info=True,  # Get dataset metadata
    as_supervised=True  # Return (image, label) pairs
)

# ---------------- Data Augmentation ----------------
# Random transformations to reduce overfitting
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip horizontally
    tf.keras.layers.RandomRotation(0.1),       # Random rotation by ±10%
    tf.keras.layers.RandomZoom(0.1),           # Random zoom ±10%
])

# ---------------- Preprocessing ----------------
def preprocess(image, label):
    """
    Resize, cast to float32, and normalize images using ResNet preprocessing
    """
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Normalize to match ResNet's expected input
    return image, label

# Apply preprocessing and augmentation to training data
ds_train = (
    ds_train.map(preprocess)  # Resize and normalize
            .map(lambda x, y: (data_augmentation(x), y))  # Apply augmentation
            .batch(BATCH_SIZE)  # Batch the data
            .shuffle(1000)     # Shuffle to make training robust
            .prefetch(tf.data.AUTOTUNE)  # Optimize data loading
)

# Validation dataset only needs preprocessing (no augmentation)
ds_val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------- Train or Load Model ----------------
if os.path.exists(MODEL_PATH):
    print("Model already exists.")  # If model is already trained, skip training
else:
    print("Training model...")

    # ---------------- Load Pretrained ResNet50V2 ----------------
    base_model = ResNet50V2(
        weights="imagenet",      # Load weights trained on ImageNet
        include_top=False,       # Exclude final classification layer
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # Freeze base model initially

    # ---------------- Add Custom Layers ----------------
    x = GlobalAveragePooling2D()(base_model.output)  # Reduce feature maps to vector
    x = Dropout(0.3)(x)                              # Prevent overfitting
    output = Dense(102, activation="softmax")(x)     # 102 flower classes

    # Define the full model
    model = Model(inputs=base_model.input, outputs=output)

    # ---------------- Compile Model ----------------
    model.compile(
        optimizer=Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Labels are integers
        metrics=["accuracy"]
    )

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,                # Stop if val_loss doesn't improve for 3 epochs
        restore_best_weights=True  # Keep the best model
    )

    # ---------------- Train the Top Layers ----------------
    model.fit(ds_train, validation_data=ds_val, epochs=10, callbacks=[early_stop])

    # ---------------- Fine-tuning ----------------
    base_model.trainable = True  # Unfreeze base model for fine-tuning
    for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
        layer.trainable = False

    # Compile with a smaller learning rate for fine-tuning
    model.compile(
        optimizer=Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Train the whole model (last layers of base + top layers)
    model.fit(ds_train, validation_data=ds_val, epochs=10, callbacks=[early_stop])

    # ---------------- Save Model ----------------
    model.save(MODEL_PATH)
    print("Model saved!")
