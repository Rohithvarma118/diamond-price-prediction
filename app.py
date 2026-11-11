
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Prediction", page_icon="💎")
st.title("💎 Diamond Price Prediction")
st.write("Enter the diamond characteristics to estimate its price using a trained KNN regression model.")

@st.cache_resource
def load_model():
    with open("knn_diamond_pipeline.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=0.5, step=0.01)
        cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
        depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)

    with col2:
        table = st.number_input("Table (%)", min_value=40.0, max_value=90.0, value=57.0, step=0.1)
        x = st.number_input("Length (x, mm)", min_value=0.0, max_value=15.0, value=4.0, step=0.01)
        y = st.number_input("Width (y, mm)", min_value=0.0, max_value=15.0, value=4.0, step=0.01)
        z = st.number_input("Height (z, mm)", min_value=0.0, max_value=15.0, value=2.5, step=0.01)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated diamond price: ${prediction:,.2f}")
    except Exception as error:
        st.error(f"Prediction failed: {error}")
