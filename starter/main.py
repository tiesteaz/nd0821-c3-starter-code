# Put the code for your API here.
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel


# Instantiate the app
app = FastAPI()

class MyClass(BaseModel):
    body: str

# Define a GET on the specified endpoint
@app.get("/")
async def say_greeting():
    return {"greeting": "This API provides method to execute ML model for the given input!"}


@app.post("/{path}")
async def insert_item(obj: MyClass, path: str, query: str):
    return {"path": path, "body": obj.body, "query": query}
