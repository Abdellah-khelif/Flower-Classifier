import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
import numpy as np
import os

# -------------------- Settings --------------------
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "flower_model.h5"

# -------------------- Dataset Loading --------------------
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

# -------------------- Data Augmentation --------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

ds_train = ds_train.map(preprocess).map(lambda x, y: (data_augmentation(x), y)).batch(BATCH_SIZE).shuffle(1000).prefetch(tf.data.AUTOTUNE)
ds_val   = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------- Train or Load --------------------
if os.path.exists(MODEL_PATH):
    print("Model already exists. Delete to retrain.")
else:
    print("Training model...")

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(102, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(ds_train, validation_data=ds_val, epochs=10, callbacks=[early_stop])

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds_train, validation_data=ds_val, epochs=10, callbacks=[early_stop])

    model.save(MODEL_PATH)
    print("Model saved!")
