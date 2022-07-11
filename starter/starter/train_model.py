# Script to train machine learning model.
from operator import index
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import sys, os
import joblib

sys.path.append(os.path.join(sys.path[0],'ml'))

from model import train_model, inference, compute_model_metrics, perform_slice_analysis
from data import process_data

# Add code to load in the data.
data_path = "../data/census.csv"
data = pd.read_csv(data_path, encoding="utf-8")

# Remove space character from the column headers
data.columns = data.columns.str.replace(" ","")
data.to_csv("../data/census_clean_columns.csv", encoding="utf-8", index=False)

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
print(type(trained_model))

# Save model
joblib.dump(trained_model,"../model/TrainedRandomForestModel.joblib", compress = 0)
joblib.dump(trained_encoder,"../model/TrainedOneHotEncoder.joblib", compress = 0)
joblib.dump(trained_lb,"../model/TrainedLabelBinarizer.joblib", compress = 0)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features = cat_features,
    label = "salary",
    training = False,
    encoder = trained_encoder,
    lb = trained_lb
)

predictions = inference(trained_model, X_test)

# Calculate model performance and save results
#  1. Overall model performance
overall_precision, overall_recall, overall_fbeta, overall_accuracy = compute_model_metrics(y_test, predictions)
print("precision: {0}".format(overall_precision))
print("recall: {0}".format(overall_recall))
print("fbeta: {0}".format(overall_fbeta))
print("accuracy: {0}".format(overall_accuracy))

print_output  = "overall precision: {0}\r\n".format(overall_precision)
print_output += "overall recall: {0}\r\n".format(overall_recall)
print_output += "overall fbeta: {0}\r\n".format(overall_fbeta)
print_output += "overall accuracy: {0}\r\n".format(overall_accuracy)
print_output += "\r\n"

with open('overall_metrics.txt', 'w', encoding='utf-8') as text_file:
    text_file.write(print_output)

#  2. Performance on slices of data
slice_analysis = perform_slice_analysis(data, cat_features, trained_model, trained_encoder, trained_lb)
slice_analysis.to_excel("slice_analysis.xlsx")