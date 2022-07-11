import json
from fastapi.testclient import TestClient

# Import our app from main.py
from main import app


# Instantiate the testing client with our app
client = TestClient(app)

def test_api_locally_get():
    r = client.get("/")

    print("{0}\r\n".format(r.status_code))
    assert r.status_code == 200
    assert r.json() == {'greeting': 'This API provides method to execute ML model for the given input!'}

def test_api_locally_post_1():
    r = client.post("/road?query=baby", json={"body": "Big"})

    print("{0}\r\n".format(r.status_code))
    print("{0}\r\n".format(r.json()))
    assert r.status_code == 200
    assert r.json() == {"path": "road", "body": "Big", "query": "baby"}

def test_api_locally_post_2():
    r = client.post("/road?query=baby", json={"body": "Small"})

    print("{0}\r\n".format(r.status_code))
    print("{0}\r\n".format(r.json()))
    assert r.status_code == 200
    assert r.json() == {"path": "road", "body": "Small", "query": "baby"}

def test_api_locally_post_3():
    r = client.post("/road?query=baby", json={"body": "Medium"})

    print("{0}\r\n".format(r.status_code))
    print("{0}\r\n".format(r.json()))
    assert r.status_code == 200
    assert r.json() == {"path": "road", "body": "Medium", "query": "baby"}
