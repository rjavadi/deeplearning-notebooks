from keras.datasets import fashion_mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(np.shape(x_train))
# print(np.shape(y_train))
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32')/255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

network = models.Sequential()

network.add(layers.Conv2D(32, (3, 3), activation='relu',
                          input_shape=(28, 28, 1)))
network.add(layers.Dropout(0.2))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dropout(0.2))
network.add(layers.Dense(10, activation='softmax'))

network.summary()
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
output = network.fit(x_train, y_train,
                     epochs=5, batch_size=64,
                     validation_data=(x_test, y_test))


test_loss, test_acc = network.evaluate(x_test, y_test)
print('test_acc:', test_acc)

print(output.history.keys())

accuracy = output.history['acc']
val_accuracy = output.history['val_acc']
loss = output.history['loss']
val_loss = output.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
