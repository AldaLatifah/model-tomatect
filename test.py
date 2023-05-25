import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

train_dir = os.path.join('data/splits', 'train')
# os.makedirs(train_dir)

validation_dir = os.path.join('data/splits', 'validation')
# os.makedirs(validation_dir)

test_dir = os.path.join('data/splits', 'test')
# os.makedirs(test_dir)

# for class_name in os.listdir(dataset_dir):
#     class_dir = os.path.join(dataset_dir, class_name)
    
#     train_class_dir = os.path.join(train_dir, class_name)
#     os.makedirs(train_class_dir)
    
#     validation_class_dir = os.path.join(validation_dir, class_name)
#     os.makedirs(validation_class_dir)
    
#     test_class_dir = os.path.join(test_dir, class_name)
#     os.makedirs(test_class_dir)

# for class_name in os.listdir(dataset_dir):
#     class_dir = os.path.join(dataset_dir, class_name)
    
#     train_class_dir = os.path.join(train_dir, class_name)
#     validation_class_dir = os.path.join(validation_dir, class_name)
#     test_class_dir = os.path.join(test_dir, class_name)
    
#     all_files = os.listdir(class_dir)
#     num_files = len(all_files)
    
#     num_train = int(num_files * train_pct)
#     num_validation = int(num_files * validation_pct)
#     num_test = int(num_files * test_pct)
    
#     random.shuffle(all_files)
    
#     train_files = all_files[:num_train]
#     validation_files = all_files[num_train:num_train+num_validation]
#     test_files = all_files[-num_test:]
    
#     for file_name in train_files:
#         src_file = os.path.join(class_dir, file_name)
#         dst_file = os.path.join(train_class_dir, file_name)
#         shutil.copyfile(src_file, dst_file)
    
#     for file_name in validation_files:
#         src_file = os.path.join(class_dir, file_name)
#         dst_file = os.path.join(validation_class_dir, file_name)
#         shutil.copyfile(src_file, dst_file)
    
#     for file_name in test_files:
#         src_file = os.path.join(class_dir, file_name)
#         dst_file = os.path.join(test_class_dir, file_name)
#         shutil.copyfile(src_file, dst_file)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

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

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(train_generator.num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=legacy.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()


epochs = 50

history = model.fit(train_generator,
                    steps_per_epoch=50,
                    epochs=epochs,
                    validation_data=validation_generator,
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