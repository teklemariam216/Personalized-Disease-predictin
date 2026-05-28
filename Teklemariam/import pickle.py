import pickle
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

@st.cache_data
def load_model():
    with open(r"C:\Users\hp\Desktop\Teklemariam\disease_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    Dosage = pd.read_csv(r"C:\Users\hp\Desktop\Teklemariam\Dosage.csv")
    precautions = pd.read_csv(r"C:\Users\hp\Desktop\Teklemariam\Precaution.csv")
    description = pd.read_csv(r"C:\Users\hp\Desktop\Teklemariam\Description.csv")
    medications = pd.read_csv(r"C:\Users\hp\Desktop\Teklemariam\Medication.csv")
    diets = pd.read_csv(r"C:\Users\hp\Desktop\Teklemariam\Diets.csv")
    return Dosage, precautions, description, medications, diets

def helper(dis, Dosage, precautions, description, medications, diets):
    desc = description[description["Disease"] == dis]["Description"].iloc[0]
    pre = precautions[precautions["Disease"] == dis]["Precaution"].iloc[0].split(", ")
    med = medications[medications["Disease"] == dis]["medication"].iloc[0].split(", ")
    die = diets[diets["Disease"] == dis]["Diets"].iloc[0].split(", ")
    dosage_info = []
    processed = set()
    for medication in med:
        pattern = r"\b" + re.escape(medication) + r"\b"
        matches = Dosage[Dosage["Treatment"].str.contains(pattern, case=False, na=False, regex=True)]
        for _, row in matches.iterrows():
            if row["Treatment"] not in processed:
                dosage_info.append(f"{row['Treatment']}: {row['Dosage']}")
                processed.add(row["Treatment"])
        if medication not in processed:
            dosage_info.append(f"{medication}: No specific dosage information available.")
            processed.add(medication)
    if not dosage_info:
        dosage_info = ["No specific dosage information available for the recommended medications."]
    return desc, pre, med, die, dosage_info

def predict(age, symptoms, model_data):
    input_vector = np.zeros(len(model_data["symptoms_dict"]))
    if "Age" in model_data["symptoms_dict"]:
        input_vector[model_data["symptoms_dict"]["Age"]] = age
    for symptom in symptoms:
        if symptom in model_data["symptoms_dict"]:
            input_vector[model_data["symptoms_dict"][symptom]] = 1
    idx = model_data["model"].predict([input_vector])[0]
    return model_data["diseases_list"][idx]

def main():
    st.title("Disease Prediction + Medical Recommendation")
    model_data = load_model()
    Dosage, precautions, description, medications, diets = load_data()
    symptoms = sorted([s for s in model_data["symptoms_dict"].keys() if s != "Age"])

    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    selected_symptoms = st.multiselect("Select symptoms", symptoms)

    if st.button("Predict"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            disease = predict(age, selected_symptoms, model_data)
            desc, pre, med, die, dosage_info = helper(disease, Dosage, precautions, description, medications, diets)
            st.subheader("Predicted disease")
            st.write(disease)
            st.subheader("Description")
            st.write(desc)
            st.subheader("Medication")
            st.write("\n".join(med))
            st.subheader("Dosage")
            st.write("\n".join(dosage_info))
            st.subheader("Diet")
            st.write("\n".join(die))
            st.subheader("Precautions")
            st.write("\n".join(pre))

if __name__ == "__main__":
    main()
