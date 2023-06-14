import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import legacy
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

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    shuffle=True,
    image_size=(img_height,img_width),
    batch_size=batch_size
)

dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    seed=123,
    shuffle=True,
    image_size=(img_height,img_width),
    batch_size=batch_size
)


dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=123,
    shuffle=True,
    image_size=(img_height,img_width),
    batch_size=batch_size
)

class_names = dataset_train.class_names

dataset_train = dataset_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
dataset_validation = dataset_validation.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
dataset_test = dataset_test.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

#keep size , normalize and data ogmantuation
data_scaling = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(img_height, img_width),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
   tf.keras.layers.experimental.preprocessing.RandomRotation(0.9),
   tf.keras.layers.experimental.preprocessing.RandomContrast(0.5),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
  tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
  tf.keras.layers.experimental.preprocessing.RandomWidth(0.2)    
])

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#now  let us apply data augmentation
train_ds = dataset_train.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

test_ds = dataset_test.map(
    lambda x, y: (data_scaling(x, training=False), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


#now  let us apply data augmentation
# test_ds = dataset_test.map(
#     lambda x, y: (data_augmentation(x, training=True), y)
# ).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (batch_size, img_width, img_height, 3)
n_classes = 10

model = tf.keras.models.Sequential([
    data_scaling,
  #  data_augmentation,
    tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    dataset_train,
    batch_size=32,
    #steps_per_epoch=len(dataset_test)// batch_size,
    validation_data=dataset_validation,
    verbose=1,
    epochs=20,
)

scores = model.evaluate(dataset_test)
print(f'Test Loss: {scores[0]:.3f}')
print(f'Test Accuracy: {scores[1]:.3f}')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Get the class labels and their corresponding indices
# class_labels = list(test_generator.class_indices.keys())
# class_indices = test_generator.class_indices

# Get the true labels
# y_true = dataset_test.classes
# y_true = np.array([class_labels[idx] for idx in y_true])

# Get the predicted labels
# y_pred = model.predict(dataset_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_pred = np.array([class_labels[idx] for idx in y_pred])

# # Generate the confusion matrix
# cm = confusion_matrix(y_true, y_pred, labels=class_labels)

# # Print the confusion matrix
# print(cm)




# Get the true labels
y_true = test_generator.classes

# Get the predicted labels
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)