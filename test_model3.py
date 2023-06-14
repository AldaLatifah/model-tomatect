import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import legacy
from sklearn.metrics import classification_report, confusion_matrix

dataset_dir = 'data/tomatect'
base_dir = 'data/splits'
train_pct = 0.8
validation_pct = 0.15
test_pct = 0.05
img_width = 256
img_height = 256
batch_size = 32

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# Define the generators for the training data
train_generators = []

data_dir = 'data/splits/train/'
class_names = os.listdir(data_dir)


train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.2, 1.0],
        rotation_range=90,
        fill_mode = 'nearest',
    )

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
model =Sequential()
#  tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(n_classes, activation='softmax'),
model.add(Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,  kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,  kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,  (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,  (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,  (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(train_generator.num_classes,activation="softmax"))

# #BATAS
# model.add(Conv2D(32,(3,3),input_shape=(img_width, img_height, 3),activation="relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=1024,activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=train_generator.num_classes,activation="softmax"))

model.compile(optimizer=legacy.Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=8000//batch_size,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps =1500//batch_size,
                    verbose=1)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()