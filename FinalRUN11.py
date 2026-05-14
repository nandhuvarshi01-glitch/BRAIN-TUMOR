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
import os

# =====================================
# CONSTANTS
# =====================================
IMG_DET = 224
IMG_SEG = 128
classes = ["Glioma", "Meningioma", "NoTumor", "Pituitary"]

IMAGE_PATH = "test.jpg"   # 🔥 Change this to your image path

# =====================================
# 1️⃣ DETECTOR MODEL
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

# =====================================
# 2️⃣ CLASSIFIER MODEL (FIXED → 4 classes)
# =====================================
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

# =====================================
# 3️⃣ SIMPLE SEGMENTER
# =====================================
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

# =====================================
# PREPROCESS
# =====================================
def preprocess(image, size):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# =====================================
# BRAIN MASK (OTSU)
# =====================================
def extract_brain_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return (mask // 255).astype(np.uint8)

# =====================================
# LOAD MODELS
# =====================================
print("🔧 Building models...")
detector = build_detector()
classifier = build_classifier()
segmenter = build_segmenter()

# ⚠ IMPORTANT:
# If you trained models, load them like this:
# detector.load_weights("detector.h5")
# classifier.load_weights("classifier.h5")
# segmenter.load_weights("segmenter.h5")

# =====================================
# LOAD IMAGE (VS CODE VERSION)
# =====================================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("❌ Image file not found")

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("❌ Invalid image file")

print("✅ Image Loaded:", img.shape)

# =====================================
# STEP 1: DETECTION
# =====================================
det_score = detector.predict(
    preprocess(img, (IMG_DET, IMG_DET)), verbose=0
)[0][0]

if det_score < 0.5:
    print("\n✅ No Tumor Detected")
    print("TSS (%) : 0.00")
    print("Severity: No Tumor")

else:
    print("\n🧠 Tumor Detected")

    # =====================================
    # STEP 2: CLASSIFICATION
    # =====================================
    cls_pred = classifier.predict(
        preprocess(img, (IMG_DET, IMG_DET)), verbose=0
    )[0]

    tumor_type = classes[np.argmax(cls_pred)]
    confidence = np.max(cls_pred)

    # =====================================
    # STEP 3: SEGMENTATION
    # =====================================
    seg = segmenter.predict(
        preprocess(img, (IMG_SEG, IMG_SEG)), verbose=0
    )[0, :, :, 0]

    tumor_mask = (seg > 0.5).astype(np.uint8)

    # =====================================
    # STEP 4: TSS
    # =====================================
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

    # =====================================
    # SEVERITY
    # =====================================
    if tss <= 5:
        severity = "Mild"
    elif tss <= 30:
        severity = "Moderate"
    else:
        severity = "High"

    # =====================================
    # OUTPUT
    # =====================================
    print("\n" + "="*50)
    print("🧠 BRAIN TUMOR ANALYSIS")
    print("="*50)
    print("Tumor Type     :", tumor_type)
    print("Confidence     :", f"{confidence:.2%}")
    print("Tumor Area(px) :", tumor_area)
    print("Brain Area(px) :", brain_area)
    print("TSS (%)        :", f"{tss:.2f}")
    print("Severity       :", severity)
    print("="*50)

    # =====================================
    # VISUALIZATION
    # =====================================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_big = cv2.resize(
        tumor_mask,
        (img_rgb.shape[1], img_rgb.shape[0])
    )

    overlay = img_rgb.copy()
    overlay[mask_big == 1] = [255, 0, 0]

    result = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original MRI")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(result)
    plt.title(f"{tumor_type} | {severity} | TSS {tss:.2f}%")
    plt.axis("off")

    plt.show()

print("\n✅ Pipeline executed successfully")