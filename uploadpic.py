from google.colab import files
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Step 1: Upload image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Step 2: Load trained model
model = load_model("final_mobilenet_asd_model.h5")

# Step 3: Preprocess and predict
IMG_SIZE = 240

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

img_array = preprocess_image(img_path)
prediction = model.predict(img_array)[0][0]

# Step 4: Display result
if prediction >= 0.5:
    print("🧠 Prediction: Non-Autistic")
else:
    print("🧠 Prediction: Autistic")
