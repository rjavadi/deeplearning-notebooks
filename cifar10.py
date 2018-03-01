import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras import models, layers, optimizers
from keras.utils import to_categorical


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


batch_size = 50
l_rate = 0.001

cifar_coll = unpickle("F:\DL-code\cifar-10-batches-py\data_batch_1")
train_1 = cifar_coll[b'data']
label_1 = cifar_coll[b'labels']
train = np.array(train_1)
train_label = np.array(label_1)

for i in range(2, 6):
    cifar_coll = unpickle("F:\DL-code\cifar-10-batches-py\data_batch_" + str(i))
    train = np.concatenate((train, cifar_coll[b'data']), axis=0)
    train_label = np.concatenate((train_label, cifar_coll[b'labels']), axis=0)
cifar_test = unpickle("F:\DL-code\cifar-10-batches-py\\test_batch")
test = cifar_test[b'data']
test_label = np.array(cifar_test[b'labels'])

train = np.transpose(np.reshape(train, (50000, 3, 32, 32)), (0, 2, 3, 1))
test = np.transpose(np.reshape(test, (10000, 3, 32, 32)), (0, 2, 3, 1))
train_label = train_label.reshape((50000, 1))
test_label = test_label.reshape((10000, 1))
train = train.astype('float32') / 255
test = test.astype('float32') / 255
train_label = to_categorical(train_label, 10)
test_label = to_categorical(test_label, 10)
print("train shape: ", train.shape)
print("label shape: ", train_label.shape)

# show a sample image from data
plt.imshow(train[20])
plt.show()

## My MODEL
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(3, 3))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=optimizers.Adam(lr=l_rate), loss='categorical_crossentropy', metrics=['accuracy'])
output = model.fit(train, train_label, epochs=20, batch_size=batch_size, validation_data=(test, test_label))
test_loss, test_acc = model.evaluate(test, test_label)
print('test_acc:', test_acc)

plt.figure(0)
plt.plot(output.history['acc'], 'r')
plt.plot(output.history['val_acc'], 'g')
plt.xlabel("Num of epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

plt.figure(1)
plt.plot(output.history['loss'], 'r')
plt.plot(output.history['val_loss'], 'g')
# plt.xticks(np.arange(0, 101, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.show()
