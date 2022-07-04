import os
import functools
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D

IMG_SIZE = (48,48)
BATCH_SIZE = 32
EPOCHS = 100

BASE_DIR = '/content/drive/MyDrive/ia-vcomp/'
TRAIN_DATA_DIR = BASE_DIR + 'fer2013/train/'
TEST_DATA_DIR = BASE_DIR + 'fer2013/validation/'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.3, zoom_range=0.3, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR, color_mode='grayscale', target_size=(IMG_SIZE[0], IMG_SIZE[1]), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
validation_generator = validation_datagen.flow_from_directory(TEST_DATA_DIR, color_mode='grayscale', target_size=(IMG_SIZE[0], IMG_SIZE[1]), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_images_length = functools.reduce(lambda a, path: a + len(path[2]), os.walk(TRAIN_DATA_DIR), 0)
test_images_length = functools.reduce(lambda a, path: a + len(path[2]), os.walk(TEST_DATA_DIR), 0)

history = model.fit(train_generator, steps_per_epoch=train_images_length//BATCH_SIZE, epochs=EPOCHS, validation_data=validation_generator, validation_steps=test_images_length//BATCH_SIZE)

model.save('emotion_detection_model_100epochs.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()