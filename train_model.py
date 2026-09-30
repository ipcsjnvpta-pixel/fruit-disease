import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json

# Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# Dataset location
DATASET_PATH = "dataset"

# Load and prepare images
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Save class names
class_indices = train_data.class_indices

with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("Classes found:", len(class_indices))
print("Training images:", train_data.samples)
print("Validation images:", validation_data.samples)

# Load MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Add our classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(len(class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training Started...")

# Train
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# Save model
model.save("fruit_disease_model.h5")

print("Training Complete!")
print("Model saved as fruit_disease_model.h5")