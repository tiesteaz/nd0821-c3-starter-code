import pandas as pd

from numpy import NaN
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf = RandomForestClassifier(
                                n_estimators = 100,
                                #criterion = 'gini',
                                #max_depth = 2000,
                                #min_samples_split = 2,
                                #min_samples_leaf = 1,
                                #min_weight_fraction_leaf = 0.0,
                                #min_impurity_decrease = 0.0,
                                #bootstrap = True,
                                #oob_score = False,
                                #n_jobs = 2,
                                random_state = 42,
                                #verbose = 0,
                                #warm_start = False,
                                #class_weight = "balanced",
                                #ccp_alpha = 0.0
                               )
    model = rf.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, F1, and accuracy.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    accuracy : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    accuracy = accuracy_score(y, preds)
    return precision, recall, fbeta, accuracy

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    prediction : np.array
        Predictions from the model.
    """
    prediction = model.predict(X)

    return prediction

def perform_slice_analysis(data, cat_features, trained_model, trained_encoder, trained_lb):
    """ This function calculates performance of the model on slices of data.
        data is sliced by unique values of all categorical features

    Inputs
    ------
    data :            dataframe with all data
    cat_features :    list of categorical features
    trained_model :   trained machine learning model
    trained_encoder : trained encoder
    trained_lb :      trained label binarizer
    
    Returns
    -------
    slice_analysis : DataFrame with the precision, recall, fbeta for each unique value of categorical feature.
    """

    X, y, _, _ = process_data(  data,
                                categorical_features = cat_features,
                                label = "salary",
                                training = False,
                                encoder = trained_encoder,
                                lb = trained_lb
                             )

    predictions = inference(trained_model, X)

    data_with_predictions = pd.DataFrame(data)
    data_with_predictions = data_with_predictions.assign(y=y, prediction=predictions)
    print(data_with_predictions.head())

    analysis_columns = ['slice',
                        'cat_feature',
                        'feature_value',
                        'slice_size',
                        'slice_precision',
                        'slice_recall',
                        'slice_fbeta',
                        'slice_accuracy'
                       ]

    slice_analysis = pd.DataFrame(columns=analysis_columns)

    i = 0

    for cat_feature in cat_features:
        for feature_value in data_with_predictions[cat_feature].unique():

            i += 1

            slice_data = data_with_predictions[data_with_predictions[cat_feature] == feature_value]

            slice_precision, slice_recall, slice_fbeta, slice_accuracy = compute_model_metrics(
                                                                                    slice_data['y'].to_numpy(),
                                                                                    slice_data['prediction'].to_numpy()
                                                                                    )

            slice_analysis.loc[len(slice_analysis.index)] = [i,
                                                             cat_feature,
                                                             feature_value,
                                                             len(slice_data.index),
                                                             slice_precision,
                                                             slice_recall,
                                                             slice_fbeta,
                                                             slice_accuracy
                                                            ]

    return slice_analysis
