# Put the code for your API here.
from operator import index
import joblib
from typing import Union
from pandas.core.frame import DataFrame
from fastapi import FastAPI
from pydantic import BaseModel
from starter.starter.ml.data import process_data

# Required for Heroku
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    #os.system("rm -r .dvc .apt/usr/lib/dvc")

# Instantiate the app
app = FastAPI()

class ModelRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
            schema_extra = {
                 "example": {
                            "age": 44,
                            "workclass": "private",
                            "fnlgt": 198282,
                            "education": "Bachelors",
                            "education-num": 9,
                            "marital-status": "Married-civ-spouse",
                            "occupation": "Craft-repair",
                            "relationship": "Wife",
                            "race": "White",
                            "sex": "Female",
                            "capital-gain": 5180,
                            "capital-loss": 0,
                            "hours-per-week": 40,
                            "native-country": "Poland"
                            }
                }
            @classmethod
            def alias_generator(cls, string: str) -> str:
                return string.replace('_', '-')

class ModelResponse(BaseModel):
    salary: str

    class Config:
            schema_extra = {
                 "example": {
                    "salary": "<=$50K"
                    }
                }

class GreetingResponse(BaseModel):
    greeting: str
    model_version: str

    class Config:
            schema_extra = {
                 "example": {
                    "greeting": "This API provides method to execute ML model for a given input!",
                    "model_version": "1.2"
                    }
                }

@app.get("/", response_model=GreetingResponse)
async def say_greeting():
    return {"greeting": "This API provides method to execute ML model for a given input!",
            "model_version": "1.4"
            }

@app.post("/inference/", response_model=ModelResponse)
async def perform_inference(request_data: ModelRequest):

    request_dictionary = request_data.dict(by_alias=True)
    input_data = DataFrame(request_dictionary, index=[0])

    # load trained models
    model = joblib.load("starter/model/TrainedRandomForestModel.joblib")
    encoder = joblib.load("starter/model/TrainedOneHotEncoder.joblib")
    binarizer = joblib.load("starter/model/TrainedLabelBinarizer.joblib")

    # load list of categorical features
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

    processed_input_data, _, _, _ = process_data(
        input_data,
        categorical_features = cat_features,
        training = False,
        encoder = encoder,
        lb = binarizer
    )

    salary_prediction = model.predict(processed_input_data)

    salary_binary = salary_prediction[0]

    if salary_binary == 1:
        salary_label = ">$50K"
    elif salary_binary == 0:
        salary_label = "<=$50K"

    return {"salary": salary_label }
