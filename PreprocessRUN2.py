import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(img_path):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Noise removal
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Normalize
    img_norm = img.astype(np.float32)/255.0

    # Add batch dimension for model
    img_array = np.expand_dims(img_norm, axis=0)

    return img, img_array  # original + normalized for model
print("preprocess completed💗")