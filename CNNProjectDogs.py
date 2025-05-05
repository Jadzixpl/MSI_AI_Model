import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

images_dir = os.path.join(os.getcwd(), "Images")  

#przeskalowanie zdjęć z podziełem na tstowe i treningowe dzięki bibliotece ImageDataGenerator
data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

#dane treningowe
train_images = data.flow_from_directory(
    images_dir,
    target_size=(224,224),
    batch_size=64,
    subset='training'
)

#dane testowe
test_images = data.flow_from_directory(
    images_dir,
    target_size=(224,224),
    batch_size=64,
    subset='validation'
)

ile_clas = len(train_images.class_indices)
print(ile_clas)

plt.figure(figsize=(12, 12))

images, labels = next(train_images)

raw_class_names = {v: k for k, v in
train_images.class_indices.items()}

class_names = {i: name[10:] for i, name in raw_class_names.items()}

for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])  
        plt.axis("off")
        class_index = np.argmax(labels[i])
        class_label = class_names[class_index]
        ax.set_title(class_label)

plt.tight_layout()
plt.show()

model = tf.keras.Sequential([
    # 1. Blok 1
    tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    # 2. Blok 2
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # 3. Blok 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # 4. Spłaszczanie i gęste warstwy
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(ile_clas, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    validation_data=test_images,
    epochs=6
)

plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność testowa')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.show()

