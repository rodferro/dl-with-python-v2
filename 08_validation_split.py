# MLP with automatic validation set
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
