from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow_datasets as tfds
from io import BytesIO  # To handle image bytes

# -------------------- Load class names --------------------
# Load Oxford Flowers 102 dataset metadata to get class names
# We only load 1% of the train split, just to get the info (not actual images)
_, ds_info = tfds.load(
    'oxford_flowers102',
    split=['train[:1%]'],
    with_info=True,
    as_supervised=True
)
# Get the list of flower class names (0-101)
class_names = ds_info.features['label'].names

# -------------------- Load trained model --------------------
MODEL_PATH = "flower_model.h5"
model = load_model(MODEL_PATH)  # Load the trained Keras model

IMG_SIZE = 224  # Model input size

# -------------------- FastAPI setup --------------------
app = FastAPI()  # Create FastAPI app instance

# Enable CORS (Cross-Origin Resource Sharing) so frontend can call API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any domain
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# -------------------- Helper function --------------------
def prepare_image(bytes_data):
    """
    Convert uploaded image bytes into model-ready input:
    1. Load image from bytes
    2. Resize to model input size
    3. Convert to array
    4. Expand dims to make batch size = 1
    5. Preprocess for ResNet50
    """
    img = image.load_img(BytesIO(bytes_data), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Normalize input like ResNet expects

# -------------------- API endpoint --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict flower type from uploaded image
    - Receives an image file
    - Converts it to model input
    - Runs prediction
    - Returns the predicted flower class
    """
    content = await file.read()  # Read uploaded file as bytes

    img = prepare_image(content)       # Preprocess image
    preds = model.predict(img)         # Get predictions
    idx = np.argmax(preds, axis=1)[0] # Get index of highest probability
    flower = class_names[idx]          # Map index to class name

    return {"prediction": flower}      # Return JSON response
