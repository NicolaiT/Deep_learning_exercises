from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import seaborn as sns
import matplotlib.pyplot as plt
# Loads the data
from tensorflow.python.keras.models import Sequential

# This is based on the solution of the first exercise

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
# Reshapes the data to work in a FFN
train_data = train_data.reshape((60000, 28*28))
test_data = test_data.reshape((10000, 28*28))
num_classes = 10
EPOCHS = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Exercise 1
# Softmax for multi-neuron output layers (classification with multiple options)
inputs = Input(shape=(28*28,))
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
output = Dense(10, activation='softmax')(x)
#x = Dropout(0.5)(x)
Model = Model(inputs, output)
Model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = Model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=64, validation_data=(test_data, test_labels))

# Exercise 2
print('Model accuracy: ' + str(Model.evaluate(test_data, test_labels)[1]))

# Exercise 3
plt.plot(range(EPOCHS), history.history['loss'], '-', color='r', label='Training loss')
plt.plot(range(EPOCHS), history.history['val_loss'], '--', color='r', label='Validation loss')
plt.plot(range(EPOCHS), history.history['acc'], '-', color='b', label='Training accuracy')
plt.plot(range(EPOCHS), history.history['val_acc'], '--', color='b', label='Validation accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

plt.savefig('graph_L2_Big.png')