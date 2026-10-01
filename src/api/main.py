from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():

    return """
<!DOCTYPE html>
<html>
<head>
<title>House Price Predictor</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f6f9;
    padding: 40px;
}

.container {
    width: 500px;
    margin: auto;
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px #ccc;
}

input {
    width: 95%;
    padding: 10px;
    margin-top: 8px;
    margin-bottom: 15px;
}

button {
    background: green;
    color: white;
    border: none;
    padding: 12px;
    width: 100%;
    cursor: pointer;
}

button:hover {
    background: darkgreen;
}

#result {
    margin-top: 20px;
    font-size: 22px;
    font-weight: bold;
    color: blue;
}

</style>

</head>

<body>

<div class="container">

<h2>House Price Prediction</h2>

<label>Square Feet</label>
<input type="number" id="sqft">

<label>Bedrooms</label>
<input type="number" id="bedrooms">

<label>Bathrooms</label>
<input type="number" id="bathrooms">

<label>Location</label>
<input type="text" id="location">

<label>Year Built</label>
<input type="number" id="year_built">

<label>Condition</label>
<input type="text" id="condition">

<button onclick="predictPrice()">
Predict Price
</button>

<div id="result"></div>

</div>

<script>

async function predictPrice() {

    let payload = {
        sqft: parseFloat(
            document.getElementById("sqft").value
        ),

        bedrooms: parseInt(
            document.getElementById("bedrooms").value
        ),

        bathrooms: parseFloat(
            document.getElementById("bathrooms").value
        ),

        location: document.getElementById("location").value,

        year_built: parseInt(
            document.getElementById("year_built").value
        ),

        condition: document.getElementById("condition").value
    };

    let response = await fetch(
        "/predict",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        }
    );

    let result = await response.json();

    document.getElementById("result").innerHTML =
        "Predicted Price: ₹ " +
        JSON.stringify(result);
}

</script>

</body>
</html>
"""


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: HousePredictionRequest
):
    return predict_price(request)


@app.post("/batch-predict")
async def batch_predict_endpoint(
    requests: list[HousePredictionRequest]
):
    return batch_predict(requests)
