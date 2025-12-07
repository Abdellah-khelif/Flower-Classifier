import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

# -------------------- Load dataset --------------------
# Load Oxford Flowers 102 dataset (images + labels)
(ds_train, ds_test, ds_valid), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train', 'test', 'validation'],
    with_info=True,
    as_supervised=True
)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = ds_info.features['label'].num_classes

# -------------------- Preprocessing functions --------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Apply preprocess + batching
train_ds = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_ds = ds_valid.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------- Build Model (Transfer Learning) --------------------
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------- Train Model --------------------
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# -------------------- Save Model --------------------
model.save("flower_model.h5")

print("Model training completed and saved as flower_model.h5")
