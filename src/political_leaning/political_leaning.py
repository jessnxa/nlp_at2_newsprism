import joblib

model = joblib.load("models/political_leaning.joblib")

def predict_political_leaning(text):
    return model.predict([text])[0]