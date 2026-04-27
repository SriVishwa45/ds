import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==========================================
# CHANGE ONLY THESE SETTINGS
# ==========================================
dataset_path = "PlantVillage"      # your main dataset folder
has_train_test_split = False  # True if dataset has train/test folders
# ==========================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

if has_train_test_split:
    train_data = datagen.flow_from_directory(
        f"{dataset_path}/train",
        target_size=(64,64),
        batch_size=10,
        class_mode='categorical'
    )

    test_data = datagen.flow_from_directory(
        f"{dataset_path}/test",
        target_size=(64,64),
        batch_size=10,
        class_mode='categorical'
    )

else:
    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(64,64),
        batch_size=10,
        class_mode='categorical',
        subset='training'
    )

    test_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(64,64),
        batch_size=10,
        class_mode='categorical',
        subset='validation'
    )

cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(train_data, epochs=3)

accuracy = cnn.evaluate(test_data)[1]
print("Accuracy:", accuracy)