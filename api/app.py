from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow_datasets as tfds
from io import BytesIO

# -------------------- Load class names --------------------
_, ds_info = tfds.load(
    'oxford_flowers102',
    split=['train[:1%]'],
    with_info=True,
    as_supervised=True
)
class_names = ds_info.features['label'].names

# -------------------- Load model --------------------
MODEL_PATH = "flower_model.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = 224

# -------------------- FastAPI --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def prepare_image(bytes_data):
    img = image.load_img(BytesIO(bytes_data), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()

    img = prepare_image(content)
    preds = model.predict(img)
    idx = np.argmax(preds, axis=1)[0]
    flower = class_names[idx]

    return {"prediction": flower}
