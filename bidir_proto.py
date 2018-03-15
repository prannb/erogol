import numpy as np
Xtrain = np.ones((3, 3, 4))
YTrain = np.ones((3, 3, 5))
Xtest = np.ones((1,2,4))
Ytest = np.ones((1,2, 5))

from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

N=2	 #10 words per question
input_dim = 4
model = Sequential()
model.add(Bidirectional(LSTM(2, return_sequences=True), merge_mode='ave', input_shape=(None, input_dim)))
model.add(TimeDistributed(Dense(5, activation='relu')))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(Xtrain, YTrain, epochs=1, batch_size=1)
pred = model.predict(Xtest)
print pred