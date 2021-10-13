from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Loads the data
from tensorflow.python.keras.models import Sequential

# This is based on the solution of the first exercise

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
# Reshapes the data to work in a FFN
num_classes = 10
EPOCHS = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

print(train_data.shape)
train_data = np.expand_dims(train_data, axis=-1) # or train_data.reshape((60000,28,28,1))
test_data = np.expand_dims(test_data, axis=-1) # or test_data.reshape((10000,28,28,1))
print(train_data.shape)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=64, validation_data=(test_data, test_labels))

print('Model accuracy: ' + str(model.evaluate(test_data, test_labels)[1]))

plt.plot(range(EPOCHS), history.history['loss'], '-', color='r', label='Training loss')
plt.plot(range(EPOCHS), history.history['val_loss'], '--', color='r', label='Validation loss')
plt.plot(range(EPOCHS), history.history['acc'], '-', color='b', label='Training accuracy')
plt.plot(range(EPOCHS), history.history['val_acc'], '--', color='b', label='Validation accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

plt.savefig('graph.png')
