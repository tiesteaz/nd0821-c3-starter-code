import json
from fastapi.testclient import TestClient

# Import our app from main.py
from main import app


# Instantiate the testing client with our app
client = TestClient(app)

def test_api_locally_get():
    r = client.get("/")
    response_json = r.json()

    #print("{0}\r\n".format(r.status_code))
    assert r.status_code == 200
    assert response_json["greeting"] == 'This API provides method to execute ML model for a given input!'

def test_api_locally_post_less_than_50k():
    r = client.post("/inference/", json={
                    "age": 20,
                    "workclass": "private",
                    "fnlgt": 1,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "construction",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "capital-gain": 5180,
                    "capital-loss": 0,
                    "hours-per-week": 10,
                    "native-country": "Poland"})

    #print("{0}\r\n".format(r.status_code))
    #print("{0}\r\n".format(r.json()))
    assert r.status_code == 200
    assert r.json() == {"salary":"<=$50K"}

def test_api_locally_post_greater_than_50k():
    r = client.post("/inference/", json={
                    "age": 43,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 292175,
                    "education": "Masters",
                    "education-num": 14,
                    "marital-status": "Divorced",
                    "occupation": "Exec-managerial",
                    "relationship": "Unmarried",
                    "race": "White",
                    "sex": "Female",
                    "capital-gain": 8000,
                    "capital-loss": 0,
                    "hours-per-week": 45,
                    "native-country": "United-States"})

    #print("{0}\r\n".format(r.status_code))
    #print("{0}\r\n".format(r.json()))
    assert r.status_code == 200
    assert r.json() == {"salary":">$50K"}
