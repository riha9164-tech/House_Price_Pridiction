from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)


# Load Trained Model


MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model Loaded Successfully")
else:
    model = None
    print("Model Not Found!")


# Home Page


@app.route("/")
def home():
    return render_template("home.html")



# About Page


@app.route("/about")
def about():
    return render_template("about.html")



# Prediction Page


@app.route("/predict")
def predict():
    return render_template("predict.html")



# Predict House Price


@app.route("/prediction", methods=["POST"])
def prediction():

    if model is None:
        return render_template(
            "predict.html",
            prediction_text="Model file not found! Train the model first."
        )

    try:

        square_footage = float(request.form["square_footage"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        year_built = int(request.form["year_built"])
        lot_size = float(request.form["lot_size"])
        garage_size = int(request.form["garage_size"])
        neighborhood_quality = float(request.form["neighborhood_quality"])

        data = np.array([[
            square_footage,
            bedrooms,
            bathrooms,
            year_built,
            lot_size,
            garage_size,
            neighborhood_quality
        ]])

        prediction = model.predict(data)[0]

        prediction = round(prediction, 2)

        return render_template(
            "result.html",
            prediction=prediction
        )

    except ValueError:
        return render_template(
            "predict.html",
            prediction_text="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "predict.html",
            prediction_text=f"Error: {str(e)}"
        )



# Error Pages


@app.errorhandler(404)
def page_not_found(e):
    return "<h2>404 - Page Not Found</h2>", 404


@app.errorhandler(500)
def internal_server_error(e):
    return "<h2>500 - Internal Server Error</h2>", 500



# Run Flask App


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )