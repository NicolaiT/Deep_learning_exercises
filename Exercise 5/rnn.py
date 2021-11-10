#You must make a recurrent neural network using the Functional API. The choice of which kind of recurrent layers you want is up to you.
#Consider how you want your input data encoded.
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

numwords = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=numwords)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(x_train.shape)
print(x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=(500,))
embedding = Embedding(numwords, 64, input_length=maxlen)(inputs)
lstm = LSTM(128)(embedding)

# lstm seems to be much slower compared to the one underneath
# If we want to use the two layers below, comment out lstm and run the two layers below.
#conv1d1 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(lstm)
#pool1 = MaxPooling1D(pool_size=2)(conv1d1) 



flatten = Flatten()(lstm)
output = Dense(2, activation='softmax')(flatten)
model = Model(inputs, output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))