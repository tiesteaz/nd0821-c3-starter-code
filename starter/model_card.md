# Model Card
## Model Details
Model creator: Andrey Baranov.
Model date: July-2022.
Model data: csv file with table-formatted data including 14 social-demographic parameters, (age, race, sex, marital status, education, etc.) and Salary label with two values "<=$50k" and ">$50k".
Model type: Random Forest Classifier with two configured parameters:
  - Random_state: 42
  - N_estimators: 100

## Intended Use
Primary intended use: learn how to prepare ML model for use in a live environment, basic technics for CI/CD, and inference via API.
Secondary intended use: predict salary of a person based on a number of social-demographic parameters
Primary intended users: ML Ops engineers, and any other people interested in application of ML to demographic analysis

## Training Data
- 80% of the model dataset (~28k records), created using train_test_split from sklearn.model_selection

## Evaluation Data
- 20% of the model dataset (~7k records), and k-fold cross validation, with slicing performed by each feature's unique value

## Metrics
Overal model performance was evaluated using test data containing 20% of dataset, and also via performance on slices.
- Overall precision: 0.73
- Overall recall: 0.63
- Overall fbeta: 0.67
- Overall accuracy: 0.86

Performance overall is saved in starter/model/overall_metrics.txt
Performance on slices is saved in starter/model/slice_analysis.xlsx

To generate files, use any of the two methods:
- Run starter/starter/train_model.py
- Run starter/pytest 

## Ethical Considerations
Model data is taken from 1994 Census bureau database, link to Kaggle competition: https://www.kaggle.com/datasets/uciml/adult-census-income. Data is simplified, and doesn't offer any ground for unethical conclusions about rase or sex factors in the data derived by ML model.

## Caveats and Recommendations
Column headers have extra leading space, which has to be removed for proper processing using pandas.
