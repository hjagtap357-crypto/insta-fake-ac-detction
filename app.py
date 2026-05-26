import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load Dataset
data = pd.read_csv(r"C:\Users\admin\Desktop\insta fake ac detection\instagram_fake.csv")

X = data.drop("fake", axis=1)
y = data["fake"]

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    followers = int(request.form["followers"])
    following = int(request.form["following"])
    posts = int(request.form["posts"])
    profile_pic = int(request.form["profile_pic"])
    bio = int(request.form["bio"])

    if followers > 1000 and posts > 20 and profile_pic == 1 and bio == 1:
        result = "REAL Instagram Account"
    else:
        result = "FAKE Instagram Account"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
