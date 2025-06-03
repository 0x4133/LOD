from __future__ import annotations

import argparse
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the MobileNetV2 model without the top layer (since we're adding our own for classification)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top for our specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification, change as needed

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare data
train_datagen = ImageDataGenerator(rescale=1./255)  # Add more data augmentation parameters as needed

def train_model(data_dir: str) -> Model:
    train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')

    model.fit(train_generator, epochs=10)
    return model

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 classifier")
    parser.add_argument("--data-dir", default="data", help="Directory with training images")
    parser.add_argument("--output", default="my_model.h5", help="Path to save the model")
    args = parser.parse_args()

    train_model(args.data_dir)
    model.save(args.output)


if __name__ == "__main__":
    main()

