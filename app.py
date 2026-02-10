# pip install fastapi pydantic uvicorn[standard]

# Application -> asgi, wsgi........ Python Django (gunicorn), Flask, FastAPI, Bottle
# IPv4 -> 192.168.34.123 -> 8.8.8.8 - www.google.com
# 127.8.13.200 -> grandstay.io - DNS

from fastapi import FastAPI
from pydantic import BaseModel
import joblib


model = joblib.load('lr_model.pkl')


app = FastAPI()

class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: int
    PTRATIO: float
    B: float
    LSTAT: float


# endpoints www.example.com/update-blog/
# CRUD -> Create, Read, Update, Delete -> Database
# GPPD -> Get, Post, Put, Delete -> APIs

@app.post('/predict/')
def predict_house_price(features: HouseFeatures):
    input_data = [[features.CRIM, features.ZN, features.INDUS, features.CHAS,
                   features.NOX, features.RM, features.AGE, features.DIS,
                   features.RAD, features.TAX, features.PTRATIO, features.B,
                   features.LSTAT]]
    prediction = model.predict(input_data)
    return {'predicted_price': prediction[0]}