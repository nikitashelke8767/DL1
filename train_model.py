import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_dir = "dataset/train"
test_dir = "dataset/test"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir, target_size=(48,48),
    color_mode="grayscale", batch_size=64,
    class_mode="categorical"
)

print("train_data.num_classes:", train_data.num_classes)

test_data = datagen.flow_from_directory(
    test_dir, target_size=(48,48),
    color_mode="grayscale", batch_size=64,
    class_mode="categorical"
)

print("test_data.num_classes:", test_data.num_classes)

if train_data.num_classes != test_data.num_classes:
    raise ValueError(
        f"Train/test class count mismatch: {train_data.num_classes} != {test_data.num_classes}. "
        "Verify your dataset folders and ensure both splits contain the same classes."
    )

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=25, validation_data=test_data)

model.save("model/emotion_model.h5")