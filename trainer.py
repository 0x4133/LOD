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

train_generator = train_datagen.flow_from_directory(
        'path/to/your/data',  # This is the target directory
        target_size=(224, 224),  # All images will be resized
        batch_size=32,
        class_mode='binary')  # Use 'categorical' for more than two classes

# Train the model
model.fit(train_generator, epochs=10)  # Adjust epochs as needed

# Save the model
model.save('my_model.h5')
