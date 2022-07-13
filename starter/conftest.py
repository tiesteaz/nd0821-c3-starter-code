import pytest

from operator import index, truediv
from sklearn.model_selection import train_test_split

import pandas as pd
import sys, os
import joblib

sys.path.append(os.path.join(sys.path[0],'starter/ml'))

from model import train_model, inference, compute_model_metrics, perform_slice_analysis
from data import process_data

@pytest.fixture(scope="session", autouse=True)
def setup():
    """ This setup function for pytest ensures that cleaned csv data file,
        model, encoder and label binarizer have been created and saved in /data and /model folders
        where tests will expect to find them.
    """

    # Add code to load in the data.
    data_path = "data/census.csv"
    data = pd.read_csv(data_path, encoding="utf-8")

    # Remove space character from the column headers
    data.columns = data.columns.str.replace(" ","")
    data.to_csv("data/census_clean_columns.csv", encoding="utf-8", index=False)

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
    joblib.dump(trained_model,"model/TrainedRandomForestModel.joblib", compress = 0)
    joblib.dump(trained_encoder,"model/TrainedOneHotEncoder.joblib", compress = 0)
    joblib.dump(trained_lb,"model/TrainedLabelBinarizer.joblib", compress = 0)
