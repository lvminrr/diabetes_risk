import joblib, shap
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from src.data import FEATURES, LABELS_FR, risk_level

st.set_page_config(page_title="AI Predictive Intelligence", page_icon="🩺", layout="wide")
st.title("🩺 AI Predictive Intelligence — risque de diabète")
st.warning("Outil expérimental d'apprentissage, entraîné sur des données synthétiques. "
           "Ce n'est PAS un dispositif médical ni un outil de diagnostic.")


@st.cache_resource
def load():
    return joblib.load("outputs/models.joblib")


art = load(); models, background = art["models"], art["background"]

with st.sidebar:
    st.header("Patient")
    age = st.slider(LABELS_FR["age"], 18, 85, 45)
    imc = st.slider(LABELS_FR["imc"], 16.0, 50.0, 26.0, 0.1)
    gly = st.slider(LABELS_FR["glycemie"], 60, 300, 95)
    pres = st.slider(LABELS_FR["pression"], 80, 200, 120)
    act = st.slider(LABELS_FR["activite"], 0, 900, 150, 10)
    fam = int(st.checkbox("Antécédents familiaux de diabète"))
    name = st.selectbox("Modèle", list(models), index=3)

x = pd.DataFrame([[age, imc, gly, pres, act, fam]], columns=FEATURES)
model = models[name]
p = float(model.predict_proba(x)[0, 1])
level, icon = risk_level(p)

tab1, tab2, tab3 = st.tabs(["Prédiction", "Comparaison des modèles", "Données"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilité estimée", f"{p:.1%}")
    c2.metric("Niveau de risque", f"{icon} {level}")
    c3.metric("Modèle", name)
    st.progress(min(p, 1.0))

    st.subheader("Facteurs influençant cette prédiction (SHAP)")
    explainer = shap.Explainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1], background)
    sv = explainer(x)
    fig, _ = plt.subplots(figsize=(7, 3.5))
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig, clear_figure=True)
    st.caption("Rouge : le facteur augmente le risque. Bleu : il le diminue. "
               "Ce sont des corrélations apprises par le modèle, pas des causes.")

with tab2:
    st.dataframe(pd.read_csv("outputs/metrics.csv", index_col=0).style.highlight_max(axis=0, color="#cde7cd"))
    a, b = st.columns(2)
    a.image("outputs/roc.png", caption="Courbes ROC")
    b.image("outputs/correlations.png", caption="Matrice de corrélations")
    st.image("outputs/confusion.png", caption="Matrices de confusion (seuil 0,5)")

with tab3:
    st.write("Échantillon du jeu d'entraînement (synthétique).")
    st.dataframe(background.head(50))
