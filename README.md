# House Price Prediction Web Application 🏡 Machine Learning & Flask

An end-to-end Machine Learning web application that predicts house prices based on property features. Built using Python, Scikit-Learn, and an interactive Flask web interface.

---

## 📌 Project Overview
- **Machine Learning Pipeline:** Trains a regression model (`train_model.py`) using historical property data to estimate house prices.
- **Trained Model Artifact:** Serialized model (`model.pkl`) loaded dynamically for real-time predictions.
- **Web Interface:** Interactive frontend built with HTML/CSS and Flask (`app.py`), allowing users to input house specs and view predictions instantly.

---

## 📁 Repository Structure

```text
house_price_Prediction/
│
├── static/
│   ├── images/                       # UI visual assets
│   └── style.css                     # Custom styling for web pages
│
├── templates/
│   ├── about.html                    # About page
│   ├── home.html                     # Landing page
│   ├── predict.html                  # Feature input form
│   └── result.html                   # Prediction output view
│
├── app.py                            # Flask application entry point
├── train_model.py                    # Script to train and serialize the model
├── model.pkl                         # Saved ML model artifact
├── house_price_regression_dataset.csv # Dataset used for model training
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
