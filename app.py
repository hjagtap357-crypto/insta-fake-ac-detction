import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("📊 Instagram Fake Account Detection")

st.write("Enter account details:")

followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)
posts = st.number_input("Posts", min_value=0)
bio_length = st.number_input("Bio Length", min_value=0)

if st.button("Predict"):
    input_data = np.array([[followers, following, posts, bio_length]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fake Account Detected")
    else:
        st.success("✅ Real Account")
