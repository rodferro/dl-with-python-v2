# Binary Classification with Sonar Dataset: Standardized Larger
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = read_csv("./data/sonar.csv", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:, :60].astype(float)
Y = dataset[:, 60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(60, input_shape=(60,), activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# evaluate baseline model with standardized dataset
estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(
    ("mlp", KerasClassifier(model=create_larger, epochs=100, batch_size=5, verbose=0))
)


pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
