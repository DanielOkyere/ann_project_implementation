#! /usr/bin/env python3
"""ANN Implementation"""

import pandas as pd
import numpy as np
from numpy import std, mean
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as clp
from  sklearn.metrics import accuracy_score as asc


def prepare_dataset(time_steps, data):
    """Prepares the dataset"""
    dataset_range = time_steps * 32592
    custom_steps = data[data["date"] < time_steps]

    X = np.array(custom_steps.drop(["final_result", "id_student"], 1))
    Y = np.array(custom_steps.drop(["final_result"]))

    # Apply OneHotEncoder
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # define Min max scaler
    scaler = MinMaxScaler()
    target_strings = label_encoder.classes_
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(1000 * time_steps)
    )

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # Reshape data for working with
    x_train = X_train.reshape(-1, time_steps, 71)
    y_train = Y_train.reshape(-1, time_steps, 4)

    x_test = X_test.reshape(-1, time_steps, 71)
    y_test = Y_test.reshape(-1, time_steps, 4)

    return x_train, x_test, y_train, y_test, target_strings


def evaluate_model(trainX, trainY, testX, testY, timesteps, target_strings):
    """Evaluation and application of Model"""
    epochs, batch_size = 100, 100

    # Building the ANN-LSTM
    classifier = Sequential()
    classifier._name = "ANN-LSTM"
    classifier.add(
        LSTM(
            200,
            input_shape=(time_steps, 71),
            return_sequences=True,
            recurrent_dropout=0.2,
            name="LSTM_Layer",
        )
    )
    classifier.add(Dropout(0.5, name="Dropout_layer"))
    classifier.add(Dense(units=100, activation="relu", name="ANN_Hidden_Layer"))

    classifier.add(Dense(units=4, activation="softmax", name="ANN_Output_Layer"))

    classifier.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )

    # print summary of model
    # print(classifier.summary())

    # Train the classifier model
    history = classifier.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.2,
    )

    # Test Model
    _, accuracy = classifier.evaluate(testX, testY, batch_size=batch_size, verbose=1)

    # print graph of result
    plt.title("Categorical Acuracy")

    plt.plot(history.history["categorical_accuracy"], label="train")
    plt.plot(history.history["val_categorical_accuracy"], label="val")

    plt.ylabel("categorical accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val", "test"], loc="upper left")
    plt.legend()
    plt.show()

    # Predictions
    predicted = classifier.predict(testX)
    predictions = [np.round(value) for value in predicted]

    d = np.array(predictions, dtype=np.int32)
    testy = np.vstack(testY)
    d = np.vstack(d)

    report = clp(testy, d, target_names=target_strings)
    print(report)

    return accuracy


def summarize_results(scores, all_accuracy):
    m, s = mean(scores), std(scores)
    all_accuracy.append(m)
    print("time steps=%d  Evaluation Accuracy: %.3f%% (+/-%.3f)" % (time_steps, m, s))


if __name__ == "__main__":
    # import data into application
    data = pd.read_csv("./datasource/data_final.csv")

    print(data.head())

    # data preprocessing
    time_steps = 1
    all_accuracy = list()
    x_train, x_test, y_train, y_test, target_strings = prepare_dataset(
        time_steps=time_steps, data=data
    )

    evaluate_model(
        trainX=x_train,
        trainY=y_test,
        testX=x_test,
        testY=y_test,
        timesteps=time_steps,
        target_strings=target_strings,
    )
