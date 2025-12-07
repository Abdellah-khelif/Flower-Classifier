# app.py
import os
from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# -------------------- Suppress TF warnings (optional) --------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

# -------------------- Model & Class Names --------------------
MODEL_PATH = "flower_model.h5"
IMG_SIZE = 224  # Model input size

class_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barberton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"
]

# Load the trained model
model = load_model(MODEL_PATH)

# -------------------- FastAPI Setup --------------------
app = FastAPI(title="Flower Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Image Preprocessing --------------------
def prepare_image(bytes_data):
    img = image.load_img(BytesIO(bytes_data), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# -------------------- Prediction Endpoint --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = prepare_image(content)
    preds = model.predict(img)
    idx = np.argmax(preds, axis=1)[0]
    flower = class_names[idx]
    return {"prediction": flower}

# -------------------- Run the App (for Render) --------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render dynamically sets PORT
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
