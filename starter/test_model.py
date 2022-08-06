import pytest
import pandas as pd
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.join(sys.path[0],'starter/starter/ml'))
from model import train_model, compute_model_metrics, perform_slice_analysis
from data import process_data

def setup_module(module):
    """setup any state specific to the execution of the given module."""

def test_train_model(train_data):

    X_train, y_train = train_data

    trained_model = train_model(X_train, y_train)

    assert isinstance(trained_model, type(RandomForestClassifier(random_state=42)))

def test_compute_model_metrics(trained_model, test_data):

    X_test, y_test, _ = test_data
    predictions = trained_model.predict(X_test)

    precision, recall, fbeta, accuracy = compute_model_metrics(y_test, predictions)

    assert precision > 0 and recall > 0 and fbeta > 0 and accuracy > 0

def test_inference(trained_model, test_data):

    X_test, _, _ = test_data
    predictions = trained_model.predict(X_test)

    assert len(predictions) > 0
    assert len(predictions) == len(X_test)

def test_perform_slice_analysis(test_data, cat_features, trained_encoder, trained_lb, trained_model):

    _, _, orig_data = test_data

    analysis_DataFrame = pd.DataFrame()
    analysis_DataFrame = perform_slice_analysis(orig_data, cat_features, trained_model, trained_encoder, trained_lb)

    assert len(analysis_DataFrame) > 0
    assert not analysis_DataFrame.empty
