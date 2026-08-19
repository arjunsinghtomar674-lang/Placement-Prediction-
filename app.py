#-------------------------------
## STEP.1 -> Import Libraries
#-------------------------------

import pandas as pd 
import streamlit as st
import joblib
import os

#------------------------------
## STEP.2 -> Load the model
#------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model aur Scaler ke Dynamic Paths
model_path = os.path.join(BASE_DIR, "model","placement_model.joblib")
scaler_path = os.path.join(BASE_DIR, "model","scalerp.joblib")

# Agar dono files subfolder mein hain (e.g., 'model' folder mein):
# model_path = os.path.join(BASE_DIR, "model", "placement_model.joblib")
# scaler_path = os.path.join(BASE_DIR, "model", "scaler.joblib")

# Dono Files Load Karein
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.error("Model ya Scaler file nahi mili! Check karein ki files sahi location par hain.")
    st.stop()


#-----------------------------------
## STEP.3 -> Title and Subheader 
#-----------------------------------

# Title of the APP
st.title("PLACEMENT PREDICTOR")

# Purpose 
st.subheader("Predict the Student is placed or not")

# Process 
st.write("Enter the details of the student given below and predict the output")

# Image
img_path = os.path.join(BASE_DIR, "pexels-tima-miroshnichenko-5439368.jpg")

if os.path.exists(img_path):
    st.image(img_path, width=400)


#-------------------------
## STEP.4 -> USER INPUTS
#-------------------------


# student name
name = st.text_input("Enter your name")
st.write(f"The name is {name}")

# Use feature of actual datasets for inputs

# 1. CGPA 
cgpa = st.number_input("The CGPA is",
                       min_value = 2.00,
                       max_value = 10.00,
                       value = 5.00,
                       step = 0.01)

st.write(f"The CGPA is {cgpa}")


# 2. Internship 
intern = st.slider(" Internship ",0,2,5,1)
st.write(f"Internship : {intern}")


# 3. Projects
project  = st.slider("project",0,2,5,1)
st.write(f"Project : {project}")


# 4. workshop certification 
workshop = st.slider("Workshop certification ",0,2,5,1)
st.write(f"Workshop Certifoication : {workshop}")


# 5. Aptitude Score 
aptitude = st.number_input("Aptitude Score ",
                           min_value = 10,
                           max_value = 100,
                           value = 30,
                           step = 1)

st.write(f"The Aptitude Score is {aptitude}")


# 6. Soft Skill Rating 
softskill = st.number_input("SoftSkillRating",
                            min_value = 1.00,
                            max_value = 5.00,
                            value = 2.00,
                            step = 0.01)
st.write(f"The SoftSkillRating is {softskill}")


# 7. ExtraCricularActivities
eca = st.selectbox("ExtraCuricularActivities",["Yes","No"])


# 8. Placement Training 
placement = st.selectbox("Placement Training ",["Yes","No"])


# 9. SSC marks
ssc = st.number_input("SSC marks ",
                      min_value = 33,
                      max_value = 100,
                      value = 40,
                      step = 1)

st.write(f"SSC marks is {ssc}")



# 10. HSC marks
hsc = st.number_input("HSC marks",
                      min_value = 33,
                      max_value = 100,
                      value = 55,
                      step = 1)

st.write(f"HSC marks {hsc}")




#---------------------------------------------------------
## STEP.5 -> Convert the user input into the dataframes
#---------------------------------------------------------

input_data = pd.DataFrame({
    "CGPA":[cgpa],
    "Internships":[intern],
    "Projects":[project],
    "Workshops/Certifications":[workshop],
    "AptitudeTestScore":[aptitude],
    "SoftSkillsRating":[softskill],
    "ExtracurricularActivities":[eca],
    "PlacementTraining":[placement],
    "SSC_Marks":[ssc],
    "HSC_Marks":[hsc],
})

input_data = pd.get_dummies(
    input_data,
    columns=["ExtracurricularActivities", "PlacementTraining"],
    dtype=int
)

for col in [
    "ExtracurricularActivities_No",
    "ExtracurricularActivities_Yes",
    "PlacementTraining_No",
    "PlacementTraining_Yes"
]:
    if col not in input_data:
        input_data[col] = 0


input_data["StudentID"] = 0

input_data = input_data[
    [
        "StudentID",
        "CGPA",
        "Internships",
        "Projects",
        "Workshops/Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "SSC_Marks",
        "HSC_Marks",
        "ExtracurricularActivities_No",
        "ExtracurricularActivities_Yes",
        "PlacementTraining_No",
        "PlacementTraining_Yes"
    ]
]

num_cols = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "SSC_Marks",
    "HSC_Marks"
]

input_data[num_cols] = scaler.transform(input_data[num_cols])




#-----------------------
## STEP.6 -> Prediction
#-----------------------

if st.button("🚀 Predict Score "):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Name : {name}")

    if prediction[0] == 1:

        st.success("You are Placed")
        st.image("pexels-ersin-1398102958-31689192.jpg",width = 100)

    else:
        st.error("You are not Placed")
        st.image("pexels-mikhail-nilov-7534380.jpg")

    st.metric(
        "Placement Probability ",
        f"{probability * 100:.2f}%"
    )