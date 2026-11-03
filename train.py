"""Entraîne et compare 4 modèles, sauvegarde modèles, métriques et graphiques."""
import sys, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from src.data import load_data, FEATURES

OUT = "outputs"
df = load_data(sys.argv[1] if len(sys.argv) > 1 else None)
X, y = df[FEATURES], df["diabetes"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"{len(df)} lignes | prévalence = {y.mean():.1%}")

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                             eval_metric="logloss", scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(), random_state=42),
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows, probas = [], {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]; pred = (p >= 0.5).astype(int)
    probas[name] = p
    rows.append({"Modèle": name,
                 "Accuracy": accuracy_score(y_te, pred),
                 "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred),
                 "F1-score": f1_score(y_te, pred),
                 "ROC-AUC": roc_auc_score(y_te, p),
                 "ROC-AUC (CV 5 plis)": cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()})
res = pd.DataFrame(rows).set_index("Modèle").round(3)
print(res.to_string())
res.to_csv(f"{OUT}/metrics.csv")

# --- Graphiques ---
fig, ax = plt.subplots(figsize=(6, 5))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_te, p); ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.2f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set(xlabel="FPR", ylabel="TPR", title="Courbes ROC"); ax.legend()
fig.savefig(f"{OUT}/roc.png", dpi=130, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, (name, p) in zip(axes, probas.items()):
    cm = confusion_matrix(y_te, (p >= .5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Non", "Oui"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)
fig.savefig(f"{OUT}/confusion.png", dpi=130, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Corrélations"); fig.savefig(f"{OUT}/correlations.png", dpi=130, bbox_inches="tight"); plt.close(fig)

joblib.dump({"models": models, "background": X_tr.sample(100, random_state=0)}, f"{OUT}/models.joblib")
print("Sauvegardé dans", OUT)
