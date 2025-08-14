# app.py
import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

st.title("📚 Course Recommendation App")
st.markdown("""
### What is this app?

This is a **Course Recommendation System** built with Streamlit.  
Select any course you’re interested in, and the app will suggest similar courses for you.  
It uses machine learning to find the best matches, helping you easily discover new learning opportunities!
""")

try:
    # Load everything from model.pkl
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    features = data["features"]
    course_names = data["course_names"]

    st.success("✅ Data loaded successfully!")

    # Show available courses
    st.subheader("Available Courses")
    st.dataframe(course_names)

    # User input
    course = st.selectbox("Select a course", course_names)
    n = st.slider("Number of recommendations", 1, 10, 3)

    # Recommendation logic
    if st.button("Recommend"):
        if course not in course_names.values:
            st.error(f"Course '{course}' not found.")
        else:
            idx = course_names[course_names == course].index[0]
            course_vector = features.iloc[[idx]]
            distances, indices = model.kneighbors(course_vector, n_neighbors=n+1)
            recommended_courses = course_names.iloc[indices[0][1:]].values.tolist()

            st.write("### 🎯 Recommended Courses:")
            for c in recommended_courses:
                st.write(f"- {c}")

except Exception as e:
    st.error(f"❌ Error: {e}")
