from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt


def vectorize_seq(sequences, dimensions=10000):
    result = np.zeros((len(sequences), dimensions))
    # So for each element in sequences, a tuple is produced with (counter, element);
    # the for loop binds that to i and seq, respectively.
    for i, seq in enumerate(sequences):
        if i == 2:
            print(i, seq)
        result[i, seq] = 1.
    return result


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=100)

word_index = imdb.get_word_index()
reverse_word_index = dict([
    (value, key) for (key, value) in word_index.items()
])
decoded_rev = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)
x_train = vectorize_seq(train_data)
train_data.mean(axis=0)
x_test = vectorize_seq(test_data)
y_train = vectorize_seq(train_labels)
y_test = vectorize_seq(test_labels)

print(x_train[0][:30])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
