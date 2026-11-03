"""Génération / chargement des données.

Par défaut : jeu de données SYNTHÉTIQUE aux 6 variables du projet.
Pour utiliser de vraies données anonymisées, fournir un CSV avec les colonnes
FEATURES + 'diabetes' (0/1) via load_data(path="mon_fichier.csv").
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "imc", "glycemie", "pression", "activite", "antecedents"]
LABELS_FR = {
    "age": "Âge (ans)",
    "imc": "IMC (kg/m²)",
    "glycemie": "Glycémie à jeun (mg/dL)",
    "pression": "Pression systolique (mmHg)",
    "activite": "Activité physique (min/semaine)",
    "antecedents": "Antécédents familiaux (0/1)",
}


def generate_synthetic(n=4000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(45, 14, n), 18, 85)
    imc = np.clip(rng.normal(26 + 0.04 * (age - 45), 4.5, n), 16, 50)
    glyc = np.clip(rng.normal(92 + 0.25 * (age - 45) + 0.8 * (imc - 26), 14, n), 60, 300)
    pres = np.clip(rng.normal(115 + 0.3 * (age - 45) + 0.6 * (imc - 26), 12, n), 80, 200)
    act = np.clip(rng.gamma(2.0, 80, n), 0, 900)
    fam = rng.binomial(1, 0.25, n)

    z = (-2.0 + 0.025 * (age - 45) + 0.06 * (imc - 26) + 0.055 * (glyc - 92)
         + 0.012 * (pres - 120) - 0.003 * (act - 160) + 0.8 * fam
         + rng.normal(0, 0.6, n))  # bruit : le monde réel n'est pas déterministe
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame({"age": age.round(0), "imc": imc.round(1), "glycemie": glyc.round(0),
                         "pression": pres.round(0), "activite": act.round(0),
                         "antecedents": fam, "diabetes": y})


def load_data(path=None) -> pd.DataFrame:
    df = pd.read_csv(path) if path else generate_synthetic()
    df = df.drop_duplicates().dropna()
    return df


def risk_level(p: float):
    """Probabilité -> (niveau, couleur). Seuils pédagogiques, non cliniques."""
    if p < 0.20: return "Faible", "#3B7A57"
    if p < 0.50: return "Modéré", "#B08A1E"
    if p < 0.75: return "Élevé", "#C2662D"
    return "Très élevé", "#A63D40"