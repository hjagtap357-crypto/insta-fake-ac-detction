import streamlit as st
import numpy as np
import pickle

# Load trained model
# Make sure your model file name is correct
model = pickle.load(open("model.pkl", "rb"))

# App Title
st.title("📊 Instagram Fake Account Detection")
st.write("Enter account details to check whether it is Fake or Real")

# Input fields (you can modify based on your dataset features)
followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)
posts = st.number_input("Posts", min_value=0)
bio_length = st.number_input("Bio Length", min_value=0)

# Predict button
if st.button("Predict"):
    # Convert input to numpy array
    input_data = np.array([[followers, following, posts, bio_length]])

    # Prediction
    prediction = model.predict(input_data)

    # Output result
    if prediction[0] == 1:
        st.error("🚨 Fake Instagram Account Detected")
    else:
        st.success("✅ Real Instagram Account")
