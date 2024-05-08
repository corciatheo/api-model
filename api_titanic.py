import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.post("/titanic")
def prediction_api(pclass: int, sex: int, age: int) -> bool:
    # Load model
    titanic_model = joblib.load("model_titanic.joblib")

    # predict
    x = [pclass, sex, age]
    prediction = titanic_model.predict(pd.DataFrame(x).transpose())

    return prediction[0] == 1


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
#test