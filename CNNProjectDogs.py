import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping

# === Ścieżki do danych ===
base_dir = os.path.join(os.getcwd())  
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# === Wczytanie danych ===
train_images = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32
)

test_images = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32
)

# === Klasy i liczba klas ===
class_names = train_images.class_names
ile_clas = len(class_names)
print("Klasy:", class_names)
print("Liczba klas:", ile_clas)

for images, labels in test_images.take(1):
    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(f"{class_names[labels[i].numpy()]}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Test Images", fontsize=16)
    plt.tight_layout()
    plt.show()


for images, labels in train_images.take(1):
    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(f"{class_names[labels[i].numpy()]}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Test Images", fontsize=16)
    plt.tight_layout()
    plt.show()

  # === Normalizacja danych i shuffle/prefetch ===
normalization_layer = tf.keras.layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE

train_images = train_images.map(lambda x, y: (normalization_layer(x), y))
test_images = test_images.map(lambda x, y: (normalization_layer(x), y))

train_images = train_images.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_images = test_images.prefetch(buffer_size=AUTOTUNE)

class SoftRandomContrast(tf.keras.layers.Layer):
    def __init__(self, strength=0.01, prob=1.0, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.prob = prob

    def call(self, images, training=True):
        def apply_contrast():
            contrast_factor = tf.random.uniform([], 1 - self.strength, 1 + self.strength)
            mean = tf.reduce_mean(images, axis=[1, 2], keepdims=True)
            return tf.clip_by_value((images - mean) * contrast_factor + mean, 0.0, 1.0)

        def skip():
            return images

        return tf.cond(
            tf.logical_and(training, tf.random.uniform([]) < self.prob),
            apply_contrast,
            skip
        )

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    SoftRandomContrast(strength=0.8, prob=0.5)  
])

# === MODEL ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
   data_augmentation,

    # Block 1
    tf.keras.layers.Conv2D(32, 3, padding='same'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(),

    # Block 2
    tf.keras.layers.Conv2D(64, 3, padding='same'),
     tf.keras.layers.Dropout(0.10),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(),

    # Block 3
    tf.keras.layers.Conv2D(128, 3, padding='same'),
   
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(ile_clas, activation='softmax')
])



model.summary()


# 1. Pobierz jeden batch z train_images
example_batch = next(iter(train_images))
images, labels = example_batch

# 2. Przepuść przez augmentację
augmented_images = data_augmentation(images)

plt.figure(figsize=(12, 6))
for i in range(8):
    # Oryginał
    ax = plt.subplot(2, 8, i + 1)
    plt.imshow(images[i].numpy())
    plt.title("Oryg.", fontsize=8)
    plt.axis("off")

    # Augmentacja
    ax = plt.subplot(2, 8, i + 9)
    aug = augmented_images[i].numpy()
    if aug.max() > 1.0:
        aug = aug.astype("uint8") * 255.0
    plt.imshow(aug)
    plt.title("Augm.", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()

# === Kompilacja ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # BO etykiety są liczbowe
    metrics=['accuracy']
)

# === Early stopping ===
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

# === Trening ===
history = model.fit(
    train_images,
    validation_data=test_images,
    epochs=80,
    callbacks=[early_stop]
)

# === Wykres dokładności ===
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.show()

# === Predykcje na testowych ===
images, labels = next(iter(test_images))
preds = model.predict(images)
predicted_classes = np.argmax(preds, axis=1)
true_classes = labels.numpy().astype("int32")

# === Pokazanie wyników ===
plt.figure(figsize=(12, 8))
for i in range(15):
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(images[i].numpy())
    plt.axis("off")
    true_label = class_names[true_classes[i]]
    pred_label = class_names[predicted_classes[i]]
    ax.set_title(f"Prawda: {true_label}\nPrzewidziane: {pred_label}", fontsize=9)

plt.tight_layout()
plt.show()