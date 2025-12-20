import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/attendance_model.pkl")

# define the event schema using pydantic model
class EventPrediction(BaseModel):
    points: int
    food_present: int
    is_virtual: int
    event_type_competition: bool
    event_type_fundraising: bool
    event_type_gbm: bool
    event_type_industry_speaker: bool
    event_type_workshop: bool
    weekday: int
    month: int
    day: int
    start_hour: int

@app.get("/")
async def root():
    return {"message": "Hello World"}

# create a route to predict attendance for a future event
@app.post("/predict")
async def predict(event: EventPrediction):
    try:
        # convert data to dataframe
        input_data = pd.DataFrame([event.dict()])

        # convert to correct data types
        input_data = input_data.astype({
            "points": "int64",
            "food_present": "int64",
            "is_virtual": "int64",
            "event_type_competition": "bool",
            "event_type_fundraising": "bool",
            "event_type_gbm": "bool",
            "event_type_industry_speaker": "bool",
            "event_type_workshop": "bool",
            "weekday": "int32",
            "month": "int32",
            "day": "int32",
            "start_hour": "int32"
        })

        # make prediction
        prediction = model.predict(input_data)

        return {"predicted_attendance": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")