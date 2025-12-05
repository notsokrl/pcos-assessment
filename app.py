import streamlit as st
import pandas as pd
import pickle

# ===============================
# LOAD PIPELINES
# ===============================
with open("pipeline_kaggle.pkl", "rb") as f:
    model_kaggle = pickle.load(f)

with open("pipeline_lifestyle.pkl", "rb") as f:
    model_lifestyle = pickle.load(f)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="PCOS Risk Assessment Tool", layout="wide")

st.markdown("""
    <div style="text-align: center; margin-top: -20px;">
        <h1 style="font-size: 48px;">💗 PCOS Risk Assessment Tool</h1>
        <p style="font-size: 18px;">
            Provide your information below to estimate your PCOS risk.
        </p>
    </div>
""", unsafe_allow_html=True)

# Helper function to check empty inputs
def missing_values(dictionary):
    return any(v in ["", None] for v in dictionary.values())

# ===============================
# SECTION: PHYSICAL / SYMPTOMS
# ===============================
with st.container():
    st.markdown("### Physical & Symptom Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age (yrs)", "")
        weight = st.text_input("Weight (Kg)", "")
        height = st.text_input("Height (Cm)", "")

        # BMI (compute only if valid)
        try:
            BMI = float(weight) / ((float(height) / 100) ** 2)
        except:
            BMI = ""

        st.metric("Calculated BMI", f"{BMI if BMI != '' else '--'}")

    with col2:
        cycle = st.selectbox("Cycle (R = 1, I = 0)", ["", 1, 0])
        cycle_length = st.text_input("Cycle Length (days)", "")

        pregnant = st.selectbox("Pregnant (Y/N)", ["", 1, 0])

        hip = st.text_input("Hip (inch)", "")
        waist = st.text_input("Waist (inch)", "")

        try:
            ratio = float(waist) / float(hip)
        except:
            ratio = ""

        st.metric("Waist–Hip Ratio", f"{ratio if ratio != '' else '--'}")

    with col3:
        weight_gain = st.selectbox("Weight Gain (Y/N)", ["", 1, 0])
        hair_growth = st.selectbox("Hair Growth (Y/N)", ["", 1, 0])
        skin_dark = st.selectbox("Skin Darkening (Y/N)", ["", 1, 0])
        hair_loss = st.selectbox("Hair Loss (Y/N)", ["", 1, 0])
        pimples = st.selectbox("Pimples (Y/N)", ["", 1, 0])
        fast_food = st.selectbox("Fast Food (Y/N)", ["", 1, 0])
        exercise = st.selectbox("Regular Exercise (Y/N)", ["", 1, 0])

# Kaggle input dictionary
kaggle_dict = {
    "Age (yrs)": age,
    "Weight (Kg)": weight,
    "Height(Cm)": height,
    "BMI": BMI,
    "Cycle(R/I)": cycle,
    "Cycle length(days)": cycle_length,
    "Pregnant(Y/N)": pregnant,
    "Hip(inch)": hip,
    "Waist(inch)": waist,
    "Waist:Hip Ratio": ratio,
    "Weight gain(Y/N)": weight_gain,
    "Hair Growth(Y/N)": hair_growth,
    "Skin darkening (Y/N)": skin_dark,
    "Hair loss(Y/N)": hair_loss,
    "Pimples(Y/N)": pimples,
    "Fast food (Y/N)": fast_food,
    "Reg.Exercise(Y/N)": exercise,
}

# ===============================
# SECTION: LIFESTYLE
# ===============================
with st.container():
    st.markdown("### Lifestyle Information")

    colA, colB, colC = st.columns(3)

    with colA:
        rice = st.text_input("Rice Servings per Day", "")
        sweets = st.text_input("Sweet Food / Drink Intake per Day", "")

    with colB:
        sleep = st.text_input("Sleep per Day (hours)", "")
        activity = st.text_input("Physical Activity per Week (hours)", "")

    with colC:
        access = st.selectbox("Access to Health Center (1 = Yes, 0 = No)", ["", 1, 0])

lifestyle_dict = {
    "Average Rice Servings per Day": rice,
    "Sweet Food or Beverage Intake per Day": sweets,
    "Sleep Duration per Day": sleep,
    "Hours of Physical Activity per Week": activity,
    "Access to Health Center": access,
}

# ===============================
# PREDICTION
# ===============================
st.markdown("---")

# Center the button
left, center, right = st.columns([1, 1, 1])
with center:
    predict_btn = st.button("🔍 Predict PCOS Risk", use_container_width=True)

# ---------- VALIDATION ----------
if predict_btn:

    if missing_values(kaggle_dict) or missing_values(lifestyle_dict):
        st.warning("⚠ Please fill in all fields before predicting.")
        st.stop()

    try:
        # Convert values to numeric
        kaggle_input = pd.DataFrame([{k: float(v) for k, v in kaggle_dict.items()}])
        lifestyle_input = pd.DataFrame([{k: float(v) for k, v in lifestyle_dict.items()}])

        kaggle_prob = model_kaggle.predict_proba(kaggle_input)[0][1]
        lifestyle_prob = model_lifestyle.predict_proba(lifestyle_input)[0][1]

        final_prob = (0.55 * kaggle_prob) + (0.45 * lifestyle_prob)

        # Centered heading
        st.markdown(
            "<h2 style='text-align:center;'>⭐ Prediction Results</h2>",
            unsafe_allow_html=True
        )

        # Final combined probability (centered)
        st.markdown(
            f"<h2 style='text-align:center; color:#7b1fa2;'>PCOS Risk: <b>{final_prob:.3f}</b></h2>",
            unsafe_allow_html=True
        )

        # Risk classification (centered)
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        if final_prob >= 0.5:
            st.error("⚠ High Risk for PCOS.")
        else:
            st.success("✓ Low Risk for PCOS.")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Prediction failed. Check input values.")
        st.code(str(e))
