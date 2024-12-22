import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Sequential

# Dataset path
DATASET_PATH = "./static/data/"

# Load dataset
df = pd.read_csv(DATASET_PATH + "styles.csv", on_bad_lines='skip')

# Helper functions

def img_path(img):
    return os.path.join(DATASET_PATH, "images", img)

def load_image(img, resized_fac=1):
    path = img_path(img)
    img = cv2.imread(path)
    if img is None:
        print(f"Image {path} could not be loaded.")
        return None  # Return None if image can't be loaded
    w, h, _ = img.shape
    return cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_CUBIC)

# Image embeddings with pre-trained ResNet50
img_width, img_height = 224, 224
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def get_embedding(model, img_name):
    try:
        path = img_path(img_name)
        img = image.load_img(path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x, verbose=0).reshape(-1)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")
        return None  # Return None if any error occurs

# Compute embeddings for the entire dataset
def precompute_embeddings():
    # Initialize list for storing embeddings
    embeddings = []

    # Loop through each image in the dataset
    for img_name in df['image']:
        print(f"Processing image: {img_name}")
        embedding = get_embedding(model, img_name)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"Skipping image {img_name} due to an error.")
    
    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings)
    
    # Save embeddings to a .npy file
    if embeddings_array.size > 0:
        np.save("image_embeddings.npy", embeddings_array)
        print(f"Embeddings saved to image_embeddings.npy")
    else:
        print("No valid embeddings to save.")

if __name__ == "__main__":
    # Call the function to precompute and save embeddings
    precompute_embeddings()
