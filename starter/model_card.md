# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model creator: Andrey Baranov
- Model date: July-2022
- Model data: csv file with table-formatted data including 14 social-demographic parameters, (age, race, sex, marital status, education, etc.) and Salary label with two values "<=$50k" and ">$50k".

- Model type: Random Forest Classifier with two configured parameters:
  - Random_state: 42
  - N_estimators: 100

## Intended Use
- Primary intended use: learn how to prepare ML model for use in a live environment, basic technics for CI/CD, and inference via API
- Secondary intended use: predict salary of a person based on a number of social-demographic parameters
- Primary intended users: ML Ops engineers, and any other people interested in application of ML to demographic analysis

## Training Data
- 80% of the model dataset (~28k records), created using train_test_split from sklearn.model_selection

## Evaluation Data
- 20% of the model dataset (~7k records), and k-fold cross validation, with slicing performed by each feature's unique value

## Metrics
Overal model performance was evaluated using test data containing 20% of dataset, and also via performance on slices.

overall precision: 0.7273413897280967
overall recall: 0.630648330058939
overall fbeta: 0.6755524377411435
overall accuracy: 0.857976354982343

Performance overall is saved in starter/model/overall_metrics.txt
Performance on slices is saved in starter/model/slice_analysis.xlsx

To generate files, use either of below methods:
- Run starter/starter/train_model.py
- Run starter/pytest 

## Ethical Considerations
Model data is taken from 1994 Census bureau database, link to Kaggle competition: https://www.kaggle.com/datasets/uciml/adult-census-income.

Data is simplified, and doesn't offer any ground for unethical conclusions about rase or sex factors in the data derived by ML model.

## Caveats and Recommendations
Column headers have extra leading space, which has to be removed for proper processing using pandas.
