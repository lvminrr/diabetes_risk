import joblib, shap
import pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from src.data import FEATURES, risk_level

st.set_page_config(page_title="Risque de diabète — outil d'étude", layout="centered")

st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 860px;}
h1 {font-weight: 600; letter-spacing: -0.5px; font-size: 1.9rem;}
h3 {font-weight: 600; font-size: 1.1rem; margin-top: 1.6rem;}
.note {color: #6B7177; font-size: 0.85rem; margin-bottom: 1.5rem;}
.result {border-left: 4px solid var(--c); background: #F1EFE9; padding: 1rem 1.2rem; border-radius: 4px;}
.result .big {font-size: 2.2rem; font-weight: 600; line-height: 1.1;}
.result .lvl {color: var(--c); font-weight: 600;}
.bar {position: relative; height: 8px; border-radius: 4px; margin: 1.2rem 0 0.3rem;
      background: linear-gradient(90deg,#3B7A57 0 20%,#B08A1E 20% 50%,#C2662D 50% 75%,#A63D40 75% 100%);}
.bar span {position: absolute; top: -5px; width: 4px; height: 18px; background: #1D2328; border-radius: 2px;}
.ticks {display: flex; justify-content: space-between; font-size: 0.75rem; color: #6B7177;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("outputs/models.joblib")


art = load(); models, background = art["models"], art["background"]
SHORT = ["Âge", "IMC", "Glycémie", "Pression", "Activité", "Antécédents"]

st.title("Estimation du risque de diabète")
st.markdown('<p class="note">Projet d\'étude réalisé sur données synthétiques. Il ne remplace '
            'en aucun cas un avis médical ni un diagnostic.</p>', unsafe_allow_html=True)

st.subheader("Profil")
c1, c2, c3 = st.columns(3)
age = c1.number_input("Âge (ans)", 18, 85, 45)
imc = c2.number_input("IMC (kg/m²)", 16.0, 50.0, 26.0, 0.1)
gly = c3.number_input("Glycémie à jeun (mg/dL)", 60, 300, 95)
c4, c5, c6 = st.columns(3)
pres = c4.number_input("Pression systolique (mmHg)", 80, 200, 120)
act = c5.number_input("Activité physique (min/semaine)", 0, 900, 150, 10)
fam = c6.selectbox("Antécédents familiaux", ["Non", "Oui"])

name = st.selectbox("Modèle utilisé", list(models), index=3)
go = st.button("Calculer", type="primary")

tab1, tab2 = st.tabs(["Résultat", "Comparaison des modèles"])

with tab1:
    if not go:
        st.caption("Renseignez le profil puis cliquez sur « Calculer ».")
    else:
        x = pd.DataFrame([[age, imc, gly, pres, act, int(fam == "Oui")]], columns=FEATURES)
        model = models[name]
        p = float(model.predict_proba(x)[0, 1])
        level, color = risk_level(p)

        st.markdown(f"""
        <div class="result" style="--c:{color}">
          <div class="big">{p:.0%}</div>
          <div>Risque estimé : <span class="lvl">{level}</span></div>
        </div>
        <div class="bar"><span style="left:calc({min(p,1)*100:.1f}% - 2px)"></span></div>
        <div class="ticks"><span>0 %</span><span>20</span><span>50</span><span>75</span><span>100 %</span></div>
        """, unsafe_allow_html=True)

        st.subheader("Ce qui pèse dans ce résultat")
        explainer = shap.Explainer(
            lambda d: model.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1], background)
        sv = explainer(x)
        sv.feature_names = SHORT
        fig = plt.figure(figsize=(7, 3.2))
        shap.plots.waterfall(sv[0], show=False)
        plt.gcf().patch.set_alpha(0)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Les barres vers la droite augmentent le risque estimé, celles vers la gauche le "
                   "diminuent. Ce sont des associations apprises par le modèle, pas des relations de cause.")

with tab2:
    m = pd.read_csv("outputs/metrics.csv", index_col=0)
    st.dataframe(m.style.format("{:.2f}"), width="stretch")
    st.caption("Évaluation sur un jeu de test de 800 profils, seuil de décision à 0,5.")
    a, b = st.columns(2)
    a.image("outputs/roc.png")
    b.image("outputs/correlations.png")
    st.image("outputs/confusion.png")