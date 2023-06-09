import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import pandas as pd
from datetime import datetime
from termcolor import colored

# Timer.
startTime = datetime.now()

# Path to created json file from mel preprocess and feature extraction script.
DATA_PATH = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/files/mel_data.json"

# Path to save model.
MODEL_SAVE = 'C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/model_1.h5'

# Path to save training history and model accuracy performance at end of training.
HISTORY_SAVE = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/history/history_1.csv"
ACC_SAVE = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/accuracy/models_acc_1.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays.
    X = np.array(data["mel"])  # The name in brackets is changed to "mfccs" if MFCC features are used to train.
    y = np.array(data["labels"])
    return X, y


def prepare_datasets(test_size, validation_size):
    # Load extracted features and labels data.
    X, y = load_data(DATA_PATH)

    # Create train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create train/validation split.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3D array.
    X_train = X_train[..., np.newaxis]  # 4-dim array: (# samples, # time steps, # coefficients, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # Create model
    model = keras.Sequential()

    # 1st convolutional layer.
    model.add(keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape))
        # 8 kernals, and 5x5 grid size of kernal
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
        # pooling size 5x5
    model.add(keras.layers.BatchNormalization())
        # Batch Normalization allows model to be more accurate and computations are faster.

    # 2nd convolutional layer.
    model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output and feed into dense layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
        # 32 = number of neurons
    model.add(keras.layers.Dropout(0.3))
    # Reduces chances of over fitting.

    # Output layer that uses softmax activation.
    model.add(keras.layers.Dense(2, activation='softmax'))
        # 2 neurons --> depends on how many categories we want to predict.

    return model


def predict(model, X, y):
    # Random prediction post-training.
    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    # Extract index with max value.
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == "__main__":
    # Create train, validation and test sets.
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)  # (test size, val size)

    # Early stopping.
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Checkpoint.
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_SAVE, monitor='val_loss',
                                                 mode='min', save_best_only=True, verbose=1)

    # Build the CNN network.
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Compile the network.
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    # Train the CNN.
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=16, epochs=1000,
                        callbacks=[callback, checkpoint])

    # Save history.
    hist = pd.DataFrame(history.history)

    # Save to csv:
    hist_csv = HISTORY_SAVE
    with open(hist_csv, mode='w') as f:
        hist.to_csv(f)

    print(
        colored("CRNN model has been trained and its training history has been saved to {}.".format(hist_csv), "green"))

    # Evaluate the CNN on the test set.
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Timer output.
    time = datetime.now() - startTime
    print(time)

    # Make prediction on a random sample.
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    # Save model accuracies on test set (for weight calculations later on).
    accuracy = {
        "model_acc": [],
        "total_train_time": [],
    }

    accuracy["model_acc"].append(test_accuracy)
    accuracy["total_train_time"].append(str(time))

    with open(ACC_SAVE, "w") as fp:
        json.dump(accuracy, fp, indent=4)


"""
The code is training a Convolutional Neural Network (CNN) on acoustic data to identify and classify different UAVs (Unmanned Aerial Vehicles) based on their sound signatures. The data has been preprocessed and features have been extracted in the form of Mel Frequency Cepstral Coefficients (MFCCs) which are fed into the CNN.

The code is split into multiple functions. load_data() loads the data from the json file. prepare_datasets() prepares the data for training by splitting it into train, validation, and test sets and reshaping it to fit the input shape of the CNN. build_model() defines the CNN architecture. predict() makes predictions on new data using a trained model.

The code sets up early stopping and model checkpoint callbacks to prevent overfitting and save the best model during training. The compiled model is then trained on the training data with a batch size of 16 and a maximum of 1000 epochs. The training history is saved in a csv file, and the final model accuracy is saved in a json file.




The programme uses audio data to train a Convolutional Neural Network (CNN) to recognise and categorise various UAVs (Unmanned Aerial Vehicles) based on their sound characteristics. Mel Frequency Cepstral Coefficients (MFCCs), which are supplied into the CNN, have been retrieved as features from the preprocessed data.

The code is broken up into several functions. The data is taken from the json file by load_data(). By dividing the data into train, validation, and test sets and reshaping it to meet the input shape of the CNN, prepare_datasets() gets the data ready for training. The CNN architecture is described by build_model(). Using a trained model, predict() makes predictions based on fresh data.


To avoid overfitting and save the best model, the code implements early halting and model checkpoint callbacks.
"""