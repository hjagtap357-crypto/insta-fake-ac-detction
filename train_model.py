import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample dataset (replace with real dataset later)
data = {
    "followers": [100, 5000, 50, 8000, 200, 3000],
    "following": [150, 200, 3000, 100, 400, 500],
    "posts": [10, 50, 2, 80, 5, 60],
    "bio_length": [20, 100, 5, 150, 10, 90],
    "label": [0, 1, 0, 1, 0, 1]   # 0 = real, 1 = fake
}

df = pd.DataFrame(data)

X = df[["followers", "following", "posts", "bio_length"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ model.pkl created successfully")