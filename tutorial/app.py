#! /usr/bin/env python3
"""Implementation of ANN"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.layers import Dense


# load dataset into application
dataset = pd.read_csv('churn_modelling.csv')
print(dataset.head())

# split dataset into X and Y
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography = pd.get_dummies(X['Geography'], drop_first = True)
gender = pd.get_dummies(X['Gender'], drop_first = True)

X = pd.concat([X, gender, geography], axis = 1)
X.drop(['Geography','Gender'], axis = 1, inplace = True)

# Training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature scaling ie normalizing data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building the Artificial neural Network
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))

# Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# train the ANN Model

## Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

## Fit the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_split=0.2)

## Predict the Test Set Results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Make a confusion matrix for large dataset
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
