# pip install streamlit

# streamlit run app.py
import streamlit as st
import pickle
import numpy as np
import webbrowser
import warnings
warnings.filterwarnings("ignore")

def load_model():
    with open('saved_model.pkl','rb') as file:
        model = pickle.load(file)
    return model
    
model = load_model()

def show_predict_page():
    st.title("Diabetes Classification")
    
    if st.button('Dataset Details'):
        webbrowser.open_new_tab("export_data.html")

    st.write("""
    ##
    ### Fill the informations for the prediction.
    ##
    """)

    pregnancies = st.slider("Pregnancies",0,50)
    glucose = st.number_input("Glucose",min_value=0)
    bloodPressure = st.number_input("Blood Pressure",min_value=0)
    skinThickness = st.number_input("Skin Thickness",min_value=0)
    insuling = st.number_input("Insuling",min_value=0)
    bmi = st.number_input("Body Mess Index (BMI)",min_value=0.)
    diabetespedigreefunction = st.number_input("Diabetes Pedigree Function",min_value=0.)
    age = st.slider("Age",0,100,20)
    
    if st.button("Prediction"):
        X = np.array([[pregnancies,glucose,bloodPressure,skinThickness,insuling,bmi,diabetespedigreefunction,age]])
        prediction = model.predict(X)[0]

        classes = {0:'not diabetes',1:'diabetes'}
        forecolor = {'not diabetes':'green','diabetes':'red'}

        predicted_class = classes.get(prediction)
        st.subheader(f"The prediction is: :{forecolor.get(predicted_class)}[{predicted_class}]")



show_predict_page()