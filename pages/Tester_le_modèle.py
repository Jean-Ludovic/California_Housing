
import streamlit as st
import pandas as pd
from pathlib import Path
import math

from ml.models import MODEL_DIR, load_saved_payload, load_data

st.set_page_config(page_title="Tester le modèle", layout="centered")

st.title("Tester le modèle")
st.write("Charge un modèle sauvegardé puis entre des valeurs pour obtenir une prédiction et un pourcentage de fiabilité.")

candidates = sorted(MODEL_DIR.glob("*.joblib"))
if not candidates:
    st.warning("Aucun modèle sauvegardé trouvé. Va d’abord sur la page Modèles & Scores pour en entraîner et sauvegarder un.")
    st.stop()

model_path = st.selectbox("Modèle à utiliser", candidates)
payload = load_saved_payload(model_path)
pipe = payload["pipeline"]
rmse = float(payload["metrics"]["rmse"])

df = load_data()
X = df.drop(columns=["median_house_value"])

st.subheader("Entrer les caractéristiques")
with st.form("form_pred"):
    cols = st.columns(2)
    inputs = {}
    for i, col in enumerate(X.columns):
        if col == "ocean_proximity":
            inputs[col] = cols[i % 2].selectbox(
                col, options=sorted(df[col].dropna().unique().tolist())
            )
        else:
            default_val = float(df[col].median())
            inputs[col] = cols[i % 2].number_input(col, value=default_val)
    rel_band = st.slider("Bande de tolérance ±% autour de la prédiction", 1, 30, 10, 1)
    submitted = st.form_submit_button("Prédire")

if submitted:
    row = pd.DataFrame([inputs])
    pred = float(pipe.predict(row)[0])

    band = (rel_band / 100.0) * abs(pred)
    conf = math.erf(band / (math.sqrt(2.0) * rmse))
    conf_pct = max(0.0, min(1.0, conf)) * 100.0

    st.success(f"Prix médian estimé : ${pred:,.0f}")
    st.write(f"D'après le calcul, cette information serait exacte à {conf_pct:.1f}% avec une bande ±{rel_band}% autour de la prédiction.")
    st.caption(f"RMSE global du modèle: {rmse:,.0f}. La fiabilité utilise une approximation normale basée sur le RMSE.")
