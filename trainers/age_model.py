import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

ROOT_DIR = '/home/dr-joel/drjoel-projects/babysitAI/'
MODELS_DIR = ROOT_DIR + 'models/'
UTKFACE_DIR = ROOT_DIR + 'data/UTKFace/'

images = []
ages = []

for img in os.listdir(UTKFACE_DIR):
  age = img.split('_')[0]
  img = cv2.imread(UTKFACE_DIR + str(img))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images.append(np.array(img))
  ages.append(np.array(age))

images = np.array(images)
ages = np.array(ages, dtype=np.int64)

x_train, x_test, y_train, y_test = train_test_split(images, ages, random_state=42)

model = Sequential()
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPool2D(pool_size=3, strides=2))

model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))
              
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))

model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='linear', name='age'))
              
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

model.save(MODELS_DIR + 'h5s/age_model_50epochs.h5')

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