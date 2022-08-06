import pytest
from _pytest import scope
from _pytest.fixtures import pytest_fixture_setup

from operator import index, truediv
from sklearn.model_selection import train_test_split

import pandas as pd
import sys, os
import joblib

sys.path.append(os.path.join(sys.path[0],'starter/starter/ml'))

from model import train_model
from data import process_data

@pytest.fixture(scope="session", autouse=True)
def setup():
    """ This setup function for pytest ensures that cleaned csv data file,
        model, encoder and label binarizer have been created and saved in /data and /model folders
        where tests will expect to find them.
    """

    modelFileExists = os.path.exists("starter/model/TrainedRandomForestModel.joblib")
    encoderFileExists = os.path.exists("starter/model/TrainedOneHotEncoder.joblib")
    labelbinarizerExists = os.path.exists("starter/model/TrainedLabelBinarizer.joblib")

    modelFilesFound = modelFileExists and encoderFileExists and labelbinarizerExists
    #print("\r\nModel files found? - {0}".format(modelFilesFound))

    if (modelFilesFound != True):

        #print("\r\nTraining model and saving model files..")

        # Add code to load in the data.
        data_path = "starter/data/census.csv"
        data = pd.read_csv(data_path, encoding="utf-8")

        # Remove space character from the column headers
        data.columns = data.columns.str.replace(" ","")
        data.to_csv("starter/data/census_clean_columns.csv", encoding="utf-8", index=False)

        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train, test = train_test_split(data, test_size=0.20)

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        # Process the train data with the process_data function.
        X_train, y_train, trained_encoder, trained_lb = process_data(
            train,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        # Train model
        trained_model = train_model(X_train, y_train)

        # Save model
        joblib.dump(trained_model,"starter/model/TrainedRandomForestModel.joblib", compress = 0)
        joblib.dump(trained_encoder,"starter/model/TrainedOneHotEncoder.joblib", compress = 0)
        joblib.dump(trained_lb,"starter/model/TrainedLabelBinarizer.joblib", compress = 0)

        #print("\r\nModel saved.")


@pytest.fixture(scope='session')
def cat_features():
    features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    return features

@pytest.fixture(scope='session')
def train_data(cat_features):
    data = pd.read_csv('starter/data/census_clean_columns.csv', encoding='utf-8')
    train_data, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, trained_encoder, trained_lb = process_data(
        train_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    return X_train, y_train

@pytest.fixture(scope='session')
def trained_model():
    trained_model = joblib.load("starter/model/TrainedRandomForestModel.joblib")

    return trained_model

@pytest.fixture(scope='session')
def trained_encoder():
    trained_encoder = joblib.load("starter/model/TrainedOneHotEncoder.joblib")

    return trained_encoder

@pytest.fixture(scope='session')
def trained_lb():
    trained_lb = joblib.load("starter/model/TrainedLabelBinarizer.joblib")

    return trained_lb

@pytest.fixture(scope='session')
def test_data(cat_features, trained_encoder, trained_lb):
    data = pd.read_csv('starter/data/census_clean_columns.csv', encoding='utf-8')
    _, test = train_test_split(data, test_size=0.20)

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features = cat_features,
        label = "salary",
        training = False,
        encoder = trained_encoder,
        lb = trained_lb
    )

    return X_test, y_test, data
