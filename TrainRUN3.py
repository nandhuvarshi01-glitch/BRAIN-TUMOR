import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32   # Better batch size for stability

# ---------------- AUTO DETECT TRAIN FOLDER ----------------
TRAIN_FOLDER = None
for f in os.listdir(DATASET_PATH):
    if f.lower() == "train":
        TRAIN_FOLDER = f
        break

if TRAIN_FOLDER is None:
    raise FileNotFoundError(f"No folder named 'Train' found in {DATASET_PATH}")

TRAIN_PATH = os.path.join(DATASET_PATH, TRAIN_FOLDER)

# ---------------- IMAGE DATA GENERATOR ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 80% train, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# ---------------- TRAIN GENERATOR ----------------
train_gen = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',   # ✅ correct value
    shuffle=True
)

# ---------------- VALIDATION GENERATOR ----------------
val_gen = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  # ✅ correct value
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes

print("✅ Classes:", train_gen.class_indices)
print("✅ Training samples:", train_gen.samples)
print("✅ Validation samples:", val_gen.samples)