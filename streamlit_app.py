import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Predictor (Dhaka)")
st.markdown("Predict property prices using area and location.")

beds = st.number_input("Number of Bedrooms", min_value=1, value=2)
baths = st.number_input("Number of Bathrooms", min_value=1, value=2)
area = st.number_input("Area (in sqft)", min_value=100, value=1000)

area_per_bed = area/beds
area_per_bath = area/baths

location_cols = ['neighborhood_Aftab Nagar',
       'neighborhood_Agargaon', 'neighborhood_Badda', 'neighborhood_Banani',
       'neighborhood_Banasree', 'neighborhood_Baridhara',
       'neighborhood_Baridhara DOHS', 'neighborhood_Bashundhara R-A',
       'neighborhood_Cantonment', 'neighborhood_Dakshin Khan',
       'neighborhood_Dhanmondi', 'neighborhood_Eskaton',
       'neighborhood_Gulshan', 'neighborhood_Hazaribag',
       'neighborhood_Ibrahimpur', 'neighborhood_Khilgaon',
       'neighborhood_Khilkhet', 'neighborhood_Lalbagh',
       'neighborhood_Maghbazar', 'neighborhood_Mirpur',
       'neighborhood_Mohammadpur', 'neighborhood_Motijheel',
       'neighborhood_Nikunja', 'neighborhood_Other', 'neighborhood_Rampura',
       'neighborhood_Shyamoli', 'neighborhood_Tejgaon', 'neighborhood_Turag',
       'neighborhood_Uttar Khan', 'neighborhood_Uttara']

location_list = [x.replace("neighborhood_", "") for x in location_cols]


location = st.selectbox("Select Location", location_list)

location_input = [1 if loc == location else 0 for loc in location_list]

features = np.array([beds, baths, area, area_per_bed, area_per_bath] + location_input).reshape(1, -1)

if st.button("Predict Price:"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Price: BDT {prediction:,.0f}")
