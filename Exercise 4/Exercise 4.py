from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import Sequential

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
num_classes = 10
EPOCHS = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

train_data = train_data.reshape((60000,28,28,1))
test_data = test_data.reshape((10000,28,28,1))

train_data = train_data.astype('float')

# TODO: visualize the filters

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1) , kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
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
