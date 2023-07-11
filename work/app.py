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
from sklearn.metrics import accuracy_score as asc


def prepare_dataset(time_steps, data):
    """Prepares the dataset"""
    dataset_range = time_steps * 32592
    custom_steps = data[data["date"] < time_steps]

    X = np.array(custom_steps.drop(labels=["final_result", "id_student"], axis=1))
    Y = np.array(custom_steps["final_result"])

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
    X_train = X_train.reshape(-1, time_steps, 71)
    Y_train = Y_train.reshape(-1, time_steps, 4)

    X_test = X_test.reshape(-1, time_steps, 71)
    Y_test = Y_test.reshape(-1, time_steps, 4)

    return X_train, X_test, Y_train, Y_test, target_strings


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

    # Predictions
    predicted = classifier.predict(testX)
    predictions = [np.round(value) for value in predicted]

    d = np.array(predictions, dtype=np.int32)
    testy = np.vstack(testY)
    d = np.vstack(d)

    report = clp(testy, d, target_names=target_strings)
    # print(report)

    return accuracy


def summarize_results(mscores, sscores, time_steps, all_accuracy):
    print(
        "\n time steps=%d  Evaluation Accuracy: %.3f%% (+/-%.3f)\n"
        % (time_steps, mscores, sscores)
    )
    # print graph of result
    plt.title("Categorical Accuracy")

    plt.plot(all_accuracy, label="test")
    plt.ylabel("categorical accuracy")
    plt.xlabel("days")
    plt.legend(["test"], loc="upper left")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # import data into application
    data = pd.read_csv("./datasource/data_final.csv")

    print(data.head())

    # data preprocessing
    time_step = 0
    scores = list()
    accuracy = list()
    for time_steps in range(1, 10):
        X_train, X_test, Y_train, Y_test, target_strings = prepare_dataset(
            time_steps=time_steps, data=data
        )

        score = evaluate_model(
            trainX=X_train,
            trainY=Y_train,
            testX=X_test,
            testY=Y_test,
            timesteps=time_steps,
            target_strings=target_strings,
        )

        scores.append(score)
        time_step += 1

    summarize_results(
        time_steps=time_step, 
        all_accuracy=scores, 
        mscores=mean(scores), 
        sscores=std(scores)
    )
