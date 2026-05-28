
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# ---------- INSERT COLOR CODE HERE ----------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: blue;
}

/* Title */
h1 {
    color: darkgreen;
    text-align: center;
    font-size: 2.4rem;
}
h2, h3, h4, h5, h6 {
    font-size: 3rem;
}

/* Form headings (large labels for inputs) */
.form-heading {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0.5rem 0 0.25rem 0;
    color: inherit;
}

/* Form labels (make Age and symptom labels larger) */
.stApp label, .stNumberInput label, .stMultiSelect label, label[for] {
    font-size: 2.2rem;
    font-weight: 700;
}

/* Buttons - larger and more prominent */
.stButton>button {
    background-color: green;
    color: white;
    border-radius: 10px;
    font-size: 20px;
    padding: 12px 18px;
    width: 100%;
    min-height: 68px;
    box-sizing: border-box;
}

/* Button hover */
.stButton>button:hover {
    background-color: darkgreen;
    color: white;
}


</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    with open(r"C:\Users\lenevo\Desktop\Teklemariam\disease_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    Dosage = pd.read_csv(r"C:\Users\lenevo\Desktop\Teklemariam\Dosage.csv")
    precautions = pd.read_csv(r"C:\Users\lenevo\Desktop\Teklemariam\Precaution.csv")
    description = pd.read_csv(r"C:\Users\lenevo\Desktop\Teklemariam\Description.csv")
    medications = pd.read_csv(r"C:\Users\lenevo\Desktop\Teklemariam\Medication.csv")
    diets = pd.read_csv(r"C:\Users\lenevo\Desktop\Teklemariam\Diets.csv")
    return Dosage, precautions, description, medications, diets
def helper(dis, age, Dosage, precautions, description, medications, diets):
    # Map short disease names to full CSV names
    disease_mapping = {
        'Acute Febrile Illness': 'Acute Febrile Illness (AFI)',
        'Acute Gastroenteritis': 'Acute Gastroenteritis (AGE)',
        'Intestinal Parasites': 'Intestinal Parasites (I/P)',
        'Upper Respiratory Tract Infection': 'Upper Respiratory Tract Infection (URTI)'
    }
    dis_csv = disease_mapping.get(dis, dis)
    
    # Ensure to get the single string value from the Series
    desc = description[description['Disease'] == dis_csv]['Description'].iloc[0]

    # Precaution column is a single string, split it into a list of individual precautions
    pre = precautions[precautions['Disease'] == dis_csv]['Precaution'].iloc[0].split(', ')

    # Medication column is named 'medication' (lowercase)
    med = medications[medications['Disease'] == dis_csv]['medication'].iloc[0].split(', ')

    # Diets column is named 'Diets' (capital D)
    die = diets[diets['Disease'] == dis_csv]['Diets'].iloc[0].split(', ')

    # Handling 'Dosage'
    dosage_info_list = []
    processed_medications = set()

    for medication_from_meds_list in med:
        # Create a regex pattern to match the whole word, case-insensitive
        pattern = r'\b' + re.escape(medication_from_meds_list) + r'\b'

        # Find matching treatments for the given age first (strict match)
        matching_treatments_df = Dosage[
            (Dosage['Treatment'].str.contains(pattern, case=False, na=False, regex=True)) &
            (Dosage['Age'] == int(age))
        ]

        if not matching_treatments_df.empty:
            for _, row in matching_treatments_df.iterrows():
                treatment_name = row['Treatment']
                dosage_value = row['Dosage']
                if treatment_name not in processed_medications:
                    dosage_info_list.append(f"{treatment_name}: {dosage_value}")
                    processed_medications.add(treatment_name)
        else:
            # No exact-age dosage found; try nearest age as fallback
            # Find all rows matching the treatment (any age)
            all_matches = Dosage[Dosage['Treatment'].str.contains(pattern, case=False, na=False, regex=True)]
            if not all_matches.empty:
                # find nearest age available
                available_ages = all_matches['Age'].dropna().astype(int).unique()
                if len(available_ages) > 0:
                    nearest_age = int(available_ages[np.argmin(np.abs(available_ages - int(age)))])
                    nearest_rows = all_matches[all_matches['Age'] == nearest_age]
                    for _, row in nearest_rows.iterrows():
                        treatment_name = row['Treatment']
                        dosage_value = row['Dosage']
                        note = f"(no exact match for age {age}; showing dosage for age {nearest_age})"
                        if treatment_name not in processed_medications:
                            dosage_info_list.append(f"{treatment_name}: {dosage_value} {note}")
                            processed_medications.add(treatment_name)
                else:
                    if medication_from_meds_list not in processed_medications:
                        dosage_info_list.append(f"{medication_from_meds_list}: No specific dosage information available for age {age}.")
                        processed_medications.add(medication_from_meds_list)
            else:
                if medication_from_meds_list not in processed_medications:
                    dosage_info_list.append(f"{medication_from_meds_list}: No specific dosage information available for age {age}.")
                    processed_medications.add(medication_from_meds_list)

    # Fallback if no medications were suggested or no dosage info was generated
    if not dosage_info_list and not med:
        dosage_info_list = ["No specific dosage information available for the recommended medications."]

    return desc, pre, med, die, dosage_info_list


def get_predicted_value(age, patient_symptoms, model_data):
    input_vector = np.zeros(len(model_data["symptoms_dict"]))

    # Set age in the input vector
    if 'Age' in model_data["symptoms_dict"]:
        input_vector[model_data["symptoms_dict"]['Age']] = age

    # Set symptoms in the input vector
    for item in patient_symptoms:
        if item in model_data["symptoms_dict"]:
            input_vector[model_data["symptoms_dict"][item]] = 1

    # predict returns an array, take the first element for the label index
    predicted_label_index = model_data["model"].predict([input_vector])[0]
    return model_data["diseases_list"][predicted_label_index]

def main():
    # --- Initialize session state for customization ---
    if 'nurse_img' not in st.session_state:
        st.session_state.nurse_img = ""
    if 'btn_bg' not in st.session_state:
        st.session_state.btn_bg = "#2ecc71"
    if 'btn_hover' not in st.session_state:
        st.session_state.btn_hover = "#27ae60"
    if 'btn_font' not in st.session_state:
        st.session_state.btn_font = 20
    if 'bullet_style' not in st.session_state:
        st.session_state.bullet_style = "• Dot"
    if 'lbl_predicted' not in st.session_state:
        st.session_state.lbl_predicted = "🏥 Predicted\nDisease"
    if 'lbl_description' not in st.session_state:
        st.session_state.lbl_description = "📋 Description\nof Disease"
    if 'lbl_med' not in st.session_state:
        st.session_state.lbl_med = "💊 Medication"
    if 'lbl_dosage' not in st.session_state:
        st.session_state.lbl_dosage = "⚕️ Dosage"
    if 'lbl_diet' not in st.session_state:
        st.session_state.lbl_diet = "🥗 Diet"
    if 'lbl_precaution' not in st.session_state:
        st.session_state.lbl_precaution = "⚠️ Precaution"
    if 'title_color' not in st.session_state:
        st.session_state.title_color = "#1abc9c"
    if 'display_bg_color' not in st.session_state:
        st.session_state.display_bg_color = "#ecf0f1"
    if 'display_text_color' not in st.session_state:
        st.session_state.display_text_color = "#000000"
    if 'title_text' not in st.session_state:
        st.session_state.title_text = "Personalized Disease Detection with Medical Recommendation System for Bahir Dar University Students"
    if 'title_alignment' not in st.session_state:
        st.session_state.title_alignment = "center"

    # --- UI customization controls in the sidebar ---
    st.sidebar.title("🎨 Dashboard Customization")
    with st.sidebar.expander("Customize UI"):
        st.session_state.nurse_img = st.text_input("Nurse image URL or local path (optional)", st.session_state.nurse_img)
        st.session_state.btn_bg = st.color_picker("Button color", st.session_state.btn_bg)
        st.session_state.btn_hover = st.color_picker("Button hover color", st.session_state.btn_hover)
        st.session_state.btn_font = st.slider("Button font size (px)", 12, 32, st.session_state.btn_font)
        st.session_state.bullet_style = st.selectbox(
            "Bullet style for lists",
            ["⭐ Star", "→ Arrow", "• Dot", "✓ Check", "◆ Diamond", "► Play", "✦ Sparkle", "- Dash"],
            index=["⭐ Star", "→ Arrow", "• Dot", "✓ Check", "◆ Diamond", "► Play", "✦ Sparkle", "- Dash"].index(st.session_state.bullet_style)
        )
        # Button label customizations
        st.session_state.lbl_predicted = st.text_input("Predicted button label", st.session_state.lbl_predicted)
        st.session_state.lbl_description = st.text_input("Description button label", st.session_state.lbl_description)
        st.session_state.lbl_med = st.text_input("Medication button label", st.session_state.lbl_med)
        st.session_state.lbl_dosage = st.text_input("Dosage button label", st.session_state.lbl_dosage)
        st.session_state.lbl_diet = st.text_input("Diet button label", st.session_state.lbl_diet)
        st.session_state.lbl_precaution = st.text_input("Precaution button label", st.session_state.lbl_precaution)

    with st.sidebar.expander("🎯 Advanced Styling"):
        st.session_state.title_color = st.color_picker("Title color", st.session_state.title_color)
        st.session_state.display_bg_color = st.color_picker("Display area background", st.session_state.display_bg_color)
        st.session_state.display_text_color = st.color_picker("Display text color", st.session_state.display_text_color)
        st.session_state.title_text = st.text_input("Custom title", st.session_state.title_text)
        st.session_state.title_alignment = st.selectbox("Title alignment", ["left", "center", "right"], index=["left", "center", "right"].index(st.session_state.title_alignment))

    with st.sidebar.expander("ℹ️ About"):
        st.write("**Version:** 1.0")
        st.write("**Author:** Medical System")
        st.write("Customize all aspects of the dashboard for your needs!")

    # --- Extract bullet character from selected style ---
    bullet_map = {
        "⭐ Star": "⭐",
        "→ Arrow": "→",
        "• Dot": "•",
        "✓ Check": "✓",
        "◆ Diamond": "◆",
        "► Play": "►",
        "✦ Sparkle": "✦",
        "- Dash": "-"
    }
    selected_bullet = bullet_map[st.session_state.bullet_style]

    # Inject dynamic CSS for buttons using chosen colors and font size
    st.markdown(f"""<style>
    /* Button styles */
    .stButton>button {{ background-color: {st.session_state.btn_bg} !important; color: white !important; font-size: {st.session_state.btn_font}px !important; padding: 10px 16px !important; border-radius: 8px !important; transition: all 0.3s ease; width: 100% !important; min-height: 68px !important; box-sizing: border-box !important; }}
    .stButton>button:hover {{ background-color: {st.session_state.btn_hover} !important; color: white !important; transform: scale(1.03); }}
    /* Title styles */
    h1 {{ color: {st.session_state.title_color} !important; text-align: {st.session_state.title_alignment} !important; font-size: 2.4rem !important; }}
    h2, h3, h4, h5, h6 {{ font-size: 3rem !important; }}
    /* Form labels (Age, symptoms) */
    label[for] {{ font-size: 2.2rem !important; font-weight: 700 !important; }}
    /* Form headings used as big labels */
    .form-heading {{ color: {st.session_state.title_color} !important; font-size: 2.6rem !important; font-weight: 700 !important; margin: 0.5rem 0 0.25rem 0 !important; }}
    /* Display area styles */
    .display-area {{ background-color: {st.session_state.display_bg_color}; color: {st.session_state.display_text_color}; padding: 20px; border-radius: 10px; margin: 15px 0; }}
    </style>""", unsafe_allow_html=True)

    # Title area with optional nurse image
    if st.session_state.nurse_img:
        try:
            col_img, col_space = st.columns([1, 4])
            with col_img:
                st.image(st.session_state.nurse_img, width=120)
        except Exception:
            st.write("[Could not load nurse image from the provided path/URL]")

    st.title(st.session_state.title_text)
    
    model_data = load_model()
    Dosage, precautions, description, medications, diets = load_data()
    
    symptom_list = sorted([s for s in model_data["symptoms_dict"].keys() if s != 'Age'])
    
    st.markdown("<div class='form-heading'>Age</div>", unsafe_allow_html=True)
    age = st.number_input("", min_value=20, max_value=27, value=25, key="age_input")
    st.markdown("<div class='form-heading'>Select your symptoms</div>", unsafe_allow_html=True)
    selected_symptoms = st.multiselect("", symptom_list, key="symptoms_input")
    
    # Submit button before all
    if st.button("Submit"):
        # Validate inputs
        if age is None:
            st.warning("Please enter your age.")
        elif not (20 <= age <= 27):
            st.warning("Age must be between 20 and 27 (inclusive).")
        elif not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Get prediction and store in session state
            predicted_disease = get_predicted_value(age, selected_symptoms, model_data)
            desc, pre, med, die, dosage_info = helper(predicted_disease, age, Dosage, precautions, description, medications, diets)
            
            st.session_state.predicted_disease = predicted_disease
            st.session_state.desc = desc
            st.session_state.pre = pre
            st.session_state.med = med
            st.session_state.die = die
            st.session_state.dosage_info = dosage_info
            st.session_state.prediction_done = True
            st.success("Submitted!")
    
    if "prediction_done" in st.session_state and st.session_state.prediction_done:
        st.markdown("---")
        # Buttons in a single row (columns) — they write HTML into session_state
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button(st.session_state.lbl_predicted, key="predicted_btn"):
                words = st.session_state.predicted_disease.split()
                st.session_state['details_html'] = '<div style="font-size:26px; font-weight:700;">' + " ".join(f"{word}" for word in words) + '</div>'
        
        with col2:
            if st.button(st.session_state.lbl_description, key="description_btn"):
                # store description (full-width) in session_state
                st.session_state['details_html'] = f"<div style=\"white-space:pre-wrap; font-size:20px;\">{st.session_state.desc}</div>"
        
        with col3:
            if st.button(st.session_state.lbl_med, key="medication_btn"):
                if st.session_state.med:
                    bullet = selected_bullet
                    meds_md = "\n".join(f"{bullet} {m}" for m in st.session_state.med)
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">' + meds_md + '</div>'
                else:
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">- None</div>'
        
        with col4:
            if st.button(st.session_state.lbl_dosage, key="dosage_btn"):
                if st.session_state.dosage_info:
                    bullet = selected_bullet
                    dosage_md = "\n".join(f"{bullet} {d}" for d in st.session_state.dosage_info)
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">' + dosage_md + '</div>'
                else:
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">- No specific dosage information available.</div>'
        
        with col5:
            if st.button(st.session_state.lbl_diet, key="diet_btn"):
                if st.session_state.die:
                    bullet = selected_bullet
                    diet_md = "\n".join(f"{bullet} {d}" for d in st.session_state.die)
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">' + diet_md + '</div>'
                else:
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">- None</div>'
        
        with col6:
            if st.button(st.session_state.lbl_precaution, key="precaution_btn"):
                if st.session_state.pre:
                    bullet = selected_bullet
                    prec_md = "\n".join(f"{bullet} {p}" for p in st.session_state.pre)
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">' + prec_md + '</div>'
                else:
                    st.session_state['details_html'] = f'<div class="display-area" style="font-size:18px; background-color:{st.session_state.display_bg_color}; color:{st.session_state.display_text_color};">- None</div>'

        # After the buttons row, render the details area below (larger text)
        details_area = st.empty()
        if 'details_html' in st.session_state:
            details_area.markdown(st.session_state['details_html'], unsafe_allow_html=True)


if __name__ == "__main__":
    main()
