from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("app/LRModel.pkl")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict_html", response_class=HTMLResponse)
async def predict_html(
    request: Request,
    Pclass: int = Query(...),
    Age: int = Query(...),
    SibSp: int = Query(...),
    Parch: int = Query(...),
    Fare: float = Query(...),
    Sex_male: int = Query(...),
    Embarked_Q: int = Query(...),
    Embarked_S: int = Query(...)
):
    data = {
        "Pclass": Pclass,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Sex_male": Sex_male,
        "Embarked_Q": Embarked_Q,
        "Embarked_S": Embarked_S
    }
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    classification_map = {0: "Non-survived", 1: "Survived"}
    classification = classification_map.get(prediction[0], "Unknown")

    return HTMLResponse(f"""
        <html>
            <head><title>Result</title></head>
            <body>
                <h2>Prediction: {classification} (class {prediction[0]})</h2>
                <a href="/">Back</a>
            </body>
        </html>
    """)
