import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report


# ==============================
# 1. DATASET PATHS
# ==============================

train_dir = "dataset/train"
test_dir = "dataset/test"


# ==============================
# 2. PARAMETERS
# ==============================

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 7


# ==============================
# 3. PREPROCESSING
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


# ==============================
# 4. TRAINING DATA
# ==============================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True
)


# ==============================
# 5. VALIDATION DATA
# ==============================

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False
)


# ==============================
# 6. TEST DATA
# ==============================

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nClass indices:")
print(train_generator.class_indices)


# ==============================
# 7. CNN MODEL
# ==============================

model = models.Sequential([

    layers.Input(shape=(48, 48, 1)),

    # Convolution Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convolution Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convolution Layer 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Flatten
    layers.Flatten(),

    # Dense Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==============================
# 8. MODEL SUMMARY
# ==============================

model.summary()


# ==============================
# 9. COMPILE
# ==============================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# 10. EARLY STOPPING
# ==============================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ==============================
# 11. TRAIN
# ==============================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[early_stopping]
)


# ==============================
# 12. ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# ==============================
# 13. LOSS GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# ==============================
# 14. TESTING
# ==============================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\nTest Loss:", test_loss)

print(
    "Test Accuracy:",
    test_accuracy * 100,
    "%"
)


# ==============================
# 15. PREDICTIONS
# ==============================

predictions = model.predict(test_generator)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes


# ==============================
# 16. CLASSIFICATION REPORT
# ==============================

class_names = list(
    test_generator.class_indices.keys()
)

print("\nClassification Report:")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# ==============================
# 17. CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Facial Emotion Recognition - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.show()


# ==============================
# 18. SAVE MODEL
# ==============================

model.save("emotion_model.keras")

print("\nModel saved successfully!")