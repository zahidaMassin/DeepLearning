#Training Cnn with Mnist datasets by AliZahid

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import  Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.datasets import mnist
import matplotlib.pyplot as plt

(trainX, trainy), (testX, testy) = mnist.load_data()
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()

trainX = trainX.astype('float32')
testX = testX.astype('float32')
# normalize to range 0-1
trainX = trainX / 255.0
testX = testX / 255.0

n_classes = 10
print("Shape before one-hot encoding: ", trainX.shape)
trainy = to_categorical(trainy, n_classes)
testy = to_categorical(testy, n_classes)
print("Shape after one-hot encoding: ", trainX.shape)



#creat ANN model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same',activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
 metrics=['accuracy'],
  optimizer='adam')

history = model.fit(trainX, trainy,
          batch_size=32, epochs=10,
          verbose=2,
          validation_data=(testX, testy))


# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

