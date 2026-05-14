# ===================== RESNET + MOBILENET HIGH ACCURACY =====================

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV3Small
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ---------------- PARAMETERS ----------------
DATA_PATH = r"Dataset\Train"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30

# ===================== RESNET DATA =====================
resnet_datagen = ImageDataGenerator(
    preprocessing_function=resnet_preprocess,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_resnet = resnet_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_resnet = resnet_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_resnet.num_classes

labels = train_resnet.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# ===================== RESNET MODEL =====================
resnet_base = ResNet50(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = GlobalAveragePooling2D()(resnet_base.output)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

resnet = Model(resnet_base.input, output)

resnet_base.trainable = False
resnet.compile(optimizer=Adam(1e-4),
               loss="categorical_crossentropy",
               metrics=["accuracy"])

resnet.fit(train_resnet,
           validation_data=val_resnet,
           epochs=10,
           class_weight=class_weights)

resnet_base.trainable = True
resnet.compile(optimizer=Adam(1e-5),
               loss="categorical_crossentropy",
               metrics=["accuracy"])

resnet.fit(train_resnet,
           validation_data=val_resnet,
           epochs=EPOCHS,
           callbacks=[EarlyStopping(patience=6, restore_best_weights=True),
                      ReduceLROnPlateau(patience=3, factor=0.2)],
           class_weight=class_weights)

resnet.save("resnet_model.h5")

# ===================== MOBILENET DATA =====================
mobile_datagen = ImageDataGenerator(
    preprocessing_function=mobilenet_preprocess,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_mobile = mobile_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_mobile = mobile_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ===================== MOBILENET MODEL =====================
mobile_base = MobileNetV3Small(weights="imagenet", include_top=False,
                               input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = GlobalAveragePooling2D()(mobile_base.output)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

mobilenet = Model(mobile_base.input, output)

mobile_base.trainable = False
mobilenet.compile(optimizer=Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

mobilenet.fit(train_mobile,
              validation_data=val_mobile,
              epochs=10,
              class_weight=class_weights)

mobile_base.trainable = True
mobilenet.compile(optimizer=Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

mobilenet.fit(train_mobile,
              validation_data=val_mobile,
              epochs=EPOCHS,
              callbacks=[EarlyStopping(patience=6, restore_best_weights=True),
                         ReduceLROnPlateau(patience=3, factor=0.2)],
              class_weight=class_weights)

mobilenet.save("mobilenet_model.h5")

print("✅ Both Models Trained Successfully")