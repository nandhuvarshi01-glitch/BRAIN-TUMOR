# =====================================
# IMPORTS
# =====================================
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# =====================================
# CONSTANTS
# =====================================
IMG_DET = 224
IMG_SEG = 128
classes = ["Glioma", "Meningioma", "NoTumor", "Pituitary"]

# =====================================
# MODELS (UNCHANGED)
# =====================================
def build_detector():
    base = MobileNetV3Small(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_DET, IMG_DET, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(base.input, output)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

def build_classifier():
    base = MobileNetV3Small(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_DET, IMG_DET, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(len(classes), activation="softmax")(x)

    model = tf.keras.Model(base.input, output)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

def build_segmenter():
    inputs = Input(shape=(IMG_SEG, IMG_SEG, 3))

    x = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = UpSampling2D()(x)
    outputs = Conv2D(1, 1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

# Build models
print("🔧 Building models...")
detector = build_detector()
classifier = build_classifier()
segmenter = build_segmenter()

# =====================================
# HELPER FUNCTIONS
# =====================================
def preprocess(image, size):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def extract_brain_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return (mask // 255).astype(np.uint8)

# =====================================
# GUI FUNCTION
# =====================================
def analyze_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Invalid Image")
        return

    det_score = detector.predict(
        preprocess(img, (IMG_DET, IMG_DET)), verbose=0
    )[0][0]

    if det_score < 0.5:
        result_label.config(text="No Tumor Detected\nTSS: 0%\nSeverity: None")
        return

    cls_pred = classifier.predict(
        preprocess(img, (IMG_DET, IMG_DET)), verbose=0
    )[0]

    tumor_type = classes[np.argmax(cls_pred)]
    confidence = np.max(cls_pred)

    seg = segmenter.predict(
        preprocess(img, (IMG_SEG, IMG_SEG)), verbose=0
    )[0, :, :, 0]

    tumor_mask = (seg > 0.5).astype(np.uint8)

    brain_mask = extract_brain_mask(img)
    brain_mask = cv2.resize(brain_mask, (IMG_SEG, IMG_SEG))

    tumor_area = np.sum(tumor_mask)
    brain_area = np.sum(brain_mask)

    if brain_area == 0:
        tss = 0
    else:
        tumor_area = min(tumor_area, brain_area)
        tss = (tumor_area / brain_area) * 100

    tss = np.clip(tss, 0, 100)

    if tss <= 5:
        severity = "Mild"
    elif tss <= 30:
        severity = "Moderate"
    else:
        severity = "High"

    result_text = f"""
Tumor Type : {tumor_type}
Confidence : {confidence:.2%}
TSS        : {tss:.2f}%
Severity   : {severity}
"""
    result_label.config(text=result_text)

    # Display image with overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_big = cv2.resize(
        tumor_mask,
        (img_rgb.shape[1], img_rgb.shape[0])
    )

    overlay = img_rgb.copy()
    overlay[mask_big == 1] = [255, 0, 0]
    result_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    display_img = Image.fromarray(result_img)
    display_img = display_img.resize((300, 300))
    tk_img = ImageTk.PhotoImage(display_img)

    image_label.config(image=tk_img)
    image_label.image = tk_img

# =====================================
# CREATE WINDOW
# =====================================
root = tk.Tk()
root.title("Brain Tumor Detection System")
root.geometry("500x600")

title = tk.Label(root, text="Brain Tumor Analysis", font=("Arial", 18))
title.pack(pady=10)

btn = tk.Button(root, text="Upload MRI Image", command=analyze_image)
btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()