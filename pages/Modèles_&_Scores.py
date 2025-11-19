# pages/4_Modeles_et_Scores_Blog.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# XGBoost optionnel
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

from ml.models import (
    load_data, split, evaluate_model, save_model_with_meta, ModelResult, MODEL_DIR
)

st.set_page_config(page_title="Modèles_et_Scores", layout="wide")

st.title("Comparaison de modèles – Style blog")
st.write(
    "Choisissez la taille du jeu de test et le seed dans le panneau latéral. "
    "Dans chaque section, cliquez sur le bouton pour entraîner et afficher les métriques de ce modèle, "
    "puis sauvegardez-le si nécessaire."
)

with st.sidebar:
    st.header("Paramètres")
    test_size = st.slider("Taille du jeu de test", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    seed = st.number_input("Seed", min_value=1, max_value=100000, value=42, step=1)

df = load_data()

DESCR = {
    "Random Forest": (
        "Forêt d'arbres de décision sur échantillons bootstrap et sous-espaces de variables. "
        "Robuste aux non-linéarités et interactions, peu sensible au scaling; fournit des importances de variables."
    ),
    "HistGradientBoosting": (
        "Gradient boosting version histogramme. Efficace, rapide et accepte nativement des valeurs manquantes."
    ),
    "XGBoost": (
        "Boosting par arbres optimisé. Très performant avec un bon réglage; souvent un excellent compromis biais/variance."
    ),
}

# Définir la liste finale des modèles à afficher
MODELS = [
    ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=42)),
]
if XGB_OK:
    MODELS.append((
        "XGBoost",
        XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        ),
    ))
else:
    st.warning("XGBoost n'est pas installé. Exécute:  pip install xgboost")

# Indiquer le dossier artifacts utilisé
st.caption(f"Dossier des modèles: {MODEL_DIR.resolve()}")

# Boucle d'affichage style blog
# Boucle d'affichage style blog
for name, estimator in MODELS:
    st.markdown("---")
    st.subheader(name)
    st.write(DESCR.get(name, ""))

    # Clés pour le session_state
    res_key = f"res_{name.replace(' ', '_')}"
    data_key = f"data_{name.replace(' ', '_')}"

    col1, col2 = st.columns(2)
    eval_clicked = col1.button(f"Évaluer {name}", key=f"btn_{name.replace(' ', '_')}")
    save_clicked = col2.button(f"Sauvegarder {name}", key=f"save_{name.replace(' ', '_')}")

    result_container = st.container()

    # 1) Si on clique sur Évaluer -> on calcule et on stocke
    if eval_clicked:
        X_train, X_test, y_train, y_test = split(df, test_size, seed)
        res = evaluate_model(name, estimator, X_train, X_test, y_train, y_test)

        # On stocke dans le session_state pour pouvoir sauvegarder après
        st.session_state[res_key] = res
        st.session_state[data_key] = (X_test, y_test)

    # 2) Si on a déjà un résultat pour ce modèle, on l’affiche
    if res_key in st.session_state:
        res = st.session_state[res_key]
        X_test, y_test = st.session_state[data_key]

        with result_container:
            st.write("Scores sur le jeu de test")
            st.write(pd.DataFrame([{
                "Modèle": res.name,
                "MAE": round(res.mae, 2),
                "RMSE": round(res.rmse, 2),
                "R2": round(res.r2, 4),
            }], index=[0]))

            # Réel vs prédit
            fig1, ax1 = plt.subplots()
            ax1.scatter(y_test, res.pipeline.predict(X_test), alpha=0.5)
            ax1.set_xlabel("Valeurs réelles")
            ax1.set_ylabel("Valeurs prédites")
            ax1.set_title(f"{name} - Réel vs Prédit")
            st.pyplot(fig1)

            # Distribution des résidus
            preds = res.pipeline.predict(X_test)
            errors = y_test - preds
            fig2, ax2 = plt.subplots()
            ax2.hist(errors, bins=50)
            ax2.set_title(f"{name} - Distribution des résidus")
            ax2.set_xlabel("Erreur (réel - prédit)")
            st.pyplot(fig2)

    # 3) Bouton Sauvegarder : on sauvegarde le dernier résultat connu
    if save_clicked:
        if res_key in st.session_state:
            res = st.session_state[res_key]
            saved_path = save_model_with_meta(res, test_size=test_size, seed=seed)
            st.success(f"Modèle sauvegardé: {saved_path}")
        else:
            st.warning("Aucun résultat à sauvegarder pour ce modèle. Clique d'abord sur 'Évaluer'.")

# Liste des modèles présents pour vérification
st.markdown("#### Modèles présents dans artifacts/")
saved_files = sorted(MODEL_DIR.glob("*.joblib"))
if saved_files:
    st.write([p.name for p in saved_files])
else:
    st.info("Aucun fichier .joblib trouvé pour l'instant.")
