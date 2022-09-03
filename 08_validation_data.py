# MLP with automatic validation set
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=seed
)

# create model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)
