import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("Advertising.csv")

# Train the model (or load from file)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

# App title
st.title("📈 Sales Prediction App")
st.write("Predict product sales based on advertising budget across TV, Radio, and Newspaper.")

# User input
tv = st.slider("TV Budget ($)", 0.0, 300.0, 100.0)
radio = st.slider("Radio Budget ($)", 0.0, 50.0, 25.0)
newspaper = st.slider("Newspaper Budget ($)", 0.0, 100.0, 20.0)

# Make prediction
input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
predicted_sales = model.predict(input_df)[0]

# Display result
st.subheader("📊 Predicted Sales")
st.success(f"${predicted_sales:.2f} thousand units")

# Show user inputs
st.markdown("### Your Inputs:")
st.write(input_df)
