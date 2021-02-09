import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten , Dense
from keras.datasets import mnist
import matplotlib.pyplot as plt

(trainX, trainy), (testX, testy) = mnist.load_data()
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()

trainX = trainX.reshape(60000, 784)
testX = testX.reshape(10000, 784)

trainX = trainX.astype('float32')
testX = testX.astype('float32')


trainX /= 255
testX /= 255

n_classes = 10
print("Shape before one-hot encoding: ", trainX.shape)
trainy = to_categorical(trainy, n_classes)
testy = to_categorical(testy, n_classes)
print("Shape after one-hot encoding: ", trainX.shape)



#creat ANN model
model = Sequential()

model.add(Dense(512,input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

# compiling model
model.compile(loss='categorical_crossentropy',
 metrics=['accuracy'],
  optimizer='adam')

history = model.fit(trainX, trainy,
          batch_size=128, epochs=20,
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

