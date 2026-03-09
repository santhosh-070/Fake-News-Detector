from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

API_KEY = "AIzaSyBjlIGJjn2f4V64GCUv1S_cwBN3H5_yEGU"

# Load dataset
fake = pd.read_csv("Fake.csv", encoding="latin1", low_memory=False)
true = pd.read_csv("True.csv", encoding="latin1", low_memory=False)

fake["label"] = "fake"
true["label"] = "real"

data = pd.concat([fake, true])
data = data[["text","label"]].dropna()

X_text = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

model = LogisticRegression()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    text = request.json["news"]

    # Google Fact Check API
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={text}&key={API_KEY}"
    response = requests.get(url)
    api_data = response.json()

    fact_result = "NO VERIFIED FACT FOUND"
    title = ""
    link = ""

    if "claims" in api_data:
        claim = api_data["claims"][0]
        title = claim.get("text","")

        if "claimReview" in claim:
            review = claim["claimReview"][0]
            fact_result = review.get("textualRating","").upper()
            link = review.get("url","")

    vec = vectorizer.transform([text])
    ai_result = model.predict(vec)[0]

    if fact_result == "FALSE":
        ai_result = "fake"
    elif fact_result == "TRUE":
        ai_result = "real"

    return jsonify({
        "ai_prediction": ai_result.upper(),
        "fact_check": fact_result,
        "title": title,
        "link": link
    })