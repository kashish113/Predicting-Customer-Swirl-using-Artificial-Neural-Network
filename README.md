#PREDICTING CUSTOMER SWIRL USING ARTIFICIAL NEURAL NETWORK

To be competitive in this market, banks have to be able to predict possible churners and take
proactive actions to retain valuable loyal customers. Building an effective and accurate
customer churn prediction model has become an important research problem for both
academics and practitioners in recent years. Profiling enables a company to act in order to
keep customers may leave (reducing churn or attrition), because it is usually far less
expensive to keep a customer than to acquire a new one

Procedure followed
# STEP 1:
Randomly initialise the weights to small numbers close to 0 (but not 0).
# STEP 2:
Input the first observation of your dataset in the input layer, each feature in one input node.
# STEP 3:
Forward-Propagation: from left to right, the neurons are activated in a way that the impact
of each neuron& activation is limited by the weights. Propagate the activations until getting the
predicted result y.
# STEP 4:
Compare the predicted result to the actual result. Measure the generated error.
# STEP 5:
Back-Propagation: from right to left, the error is back-propagated. Update the weights
according to
how much they are responsible for the error. The learning rate decides by how much we
update the weights.
# STEP 6:
Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning).
Or:
Repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch
Learning).
# STEP 7:

When the whole training set passed through the ANN, that makes an epoch. Redo more
epochs.

STEPS TO BE FOLLOWED FOR DEVELOPING THE MODEL

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

 # Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

 # Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	Making the ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

	Making the PREDICTIONS AND EVALUATING THE MODEL
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


The confusion matrix is calculated as shown
  we got a total of (2162) correct predictions and  (709) incorrect predictions with an accuracy of 83%



