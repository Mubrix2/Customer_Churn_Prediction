from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pickle
from typing import Literal
import pandas as pd

app = FastAPI()

with open('XGBModel.pkl', 'rb') as f:
 model = pickle.load(f)


class UserInput(BaseModel):

 Age: int = Field(..., gt=0, lt=120)
 Annual_Income: float|int = Field(..., gt=0, lt=150)
 Spending_Score: int = Field(..., gt=0, lt=100)

@app.get('/')
def home():
 return {'message': 'You are welcome'}

class_probs = model.classes_.tolist()
@app.post('/predict')
def predict_customer(data: UserInput):

 input_df = pd.DataFrame([{

  "Age": data.Age,
  "Annual_Income": data.Annual_Income,
  "Spending_Score": data.Spending_Score
 }])

 
 probabilities = model.predict_proba(input_df)[0]
 confidence = float(max(probabilities))

 prediction = int(model.predict(input_df)[0])

 class_probabilities = {
        int(cls): float(prob)  # convert both keys and values to JSON-safe types
        for cls, prob in zip(class_probs, map(lambda x: round(x, 4), probabilities))
    }

 message = (
   'You are a Premium Spender! You earn and spend high on things as soon as you feel the need' if prediction == 3 else 
   'You are Saver! You earn high, but like to save in case of urgency' 
   if prediction == 2 else
   'You are an Impulsive Spender! You earn low but spend high' if prediction == 1
   else
   'You are a Careful Spender! You earn low and spend low'
 )


 return JSONResponse(status_code=200, content={
  'message': message,
  'confidence': round(confidence, 4),
  'class probabilities': class_probabilities
 })