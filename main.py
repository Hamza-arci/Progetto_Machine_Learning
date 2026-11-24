import warnings

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ignoriamo i warning per mantenere l'output pulito durante l'esecuzione
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

# === 1. CARICAMENTO DATASET ===
print("Caricamento dataset in corso...")
df = pd.read_csv("NPHA-doctor-visits.csv")

# FIX 1: Rimuoviamo eventuali spazi vuoti accidentali all'inizio/fine dei nomi delle colonne
df.columns = df.columns.str.strip()

# FIX 2: Stampiamo i nomi esatti delle colonne letti da Pandas
print("\nNomi delle colonne letti dal CSV:")
print(df.columns.tolist())

X = df.drop(columns=["Number of Doctors Visited"])
y = df["Number of Doctors Visited"]

# === 2. EDA - DISTRIBUZIONE CLASSI ===
# Visualizzazione semplice per mostrare eventuale sbilanciamento delle classi
fig, ax = plt.subplots(figsize=(6, 4))
conteggio = y.value_counts().sort_index()
ax.bar(
    conteggio.index.astype(str),
    conteggio.values,
    color=["#4C72B0", "#55A868", "#C44E52"],
)
ax.set_title("Distribuzione Target: Numero di Medici Visitati")
ax.set_xlabel("Classi (1: 0-1, 2: 2-3, 3: 4+)")
plt.tight_layout()
plt.show()

# === 3. SPLIT E BASELINE ===
# Split 80/20. Lo stratify garantisce la stessa proporzione di classi in train e test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)


# === 4. PREPROCESSING (ANTI-LEAKAGE) ===
scaler = StandardScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)


# === 5. MODEL SELECTION (GRID SEARCH CV) ===
print("\nAvvio GridSearchCV...")
modelli_e_griglie = {
    "LogReg": (
        LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "SVM": (
        SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [3, 5, 10]},
    ),
}

best_estimators = []
miglior_modello_assoluto = None
miglior_score_assoluto = 0
nome_miglior_modello = ""

for nome, (modello, parametri) in modelli_e_griglie.items():
    gs = GridSearchCV(modello, parametri, cv=5, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train_proc, y_train)
    best_estimators.append((nome, gs.best_estimator_))
    print(
        f"{nome} -> Miglior F1-Macro CV: {gs.best_score_:.4f} (Params: {gs.best_params_})"
    )

    # Salviamo dinamicamente il modello se ha superato lo score massimo precedente
    if gs.best_score_ > miglior_score_assoluto:
        miglior_score_assoluto = gs.best_score_
        miglior_modello_assoluto = gs.best_estimator_
        nome_miglior_modello = nome

print(
    f"\n Il miglior modello in assoluto estratto dalla GridSearch è: {nome_miglior_modello}"
)

# === 6. STACKING ENSEMBLE (CONFRONTO DINAMICO) ===
print("\nAddestramento e valutazione Stacking Ensemble (CV)...")
stacking_clf = StackingClassifier(
    estimators=best_estimators,
    final_estimator=LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    cv=5,
    n_jobs=-1,
)

score_stacking = cross_val_score(
    stacking_clf, X_train_proc, y_train, cv=5, scoring="f1_macro", n_jobs=-1
).mean()

print(f"Stacking Ensemble -> F1-Macro CV stimato: {score_stacking:.4f}")

# Confronto dinamico e "Survival of the fittest"
if score_stacking > miglior_score_assoluto:
    print(
        f"Lo Stacking Ensemble ({score_stacking:.4f}) ha superato {nome_miglior_modello} ({miglior_score_assoluto:.4f})!"
    )
    miglior_score_assoluto = score_stacking
    miglior_modello_assoluto = stacking_clf
    nome_miglior_modello = "Stacking Ensemble"
else:
    print(
        f"Lo Stacking Ensemble ({score_stacking:.4f}) NON ha superato {nome_miglior_modello} ({miglior_score_assoluto:.4f})."
    )
    print(f"Mantengo {nome_miglior_modello} come miglior modello.")

# Indipendentemente da chi ha vinto, facciamo un fit finale sul training set
print(f"\nAddestramento finale di {nome_miglior_modello} su tutto il training set...")
miglior_modello_assoluto.fit(X_train_proc, y_train)

# === 7. VALUTAZIONE SUL TEST SET (DINAMICA SUL MIGLIORE) ===
y_pred = miglior_modello_assoluto.predict(X_test_proc)
print(f"\n--- Classification Report ({nome_miglior_modello} su Test Set) ---")
print(classification_report(y_test, y_pred))

# === 7.1 MATRICE DI CONFUSIONE ===
print("\n--- Matrice di Confusione (Testuale) ---")
cm = confusion_matrix(y_test, y_pred, labels=miglior_modello_assoluto.classes_)
print(cm)

# Visualizzazione grafica della matrice di confusione
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=miglior_modello_assoluto.classes_
)
disp.plot(cmap="Blues", values_format="d", ax=ax)
ax.set_title(f"Matrice di Confusione - {nome_miglior_modello}")
plt.tight_layout()
plt.show()

# === 8. VISUALIZZAZIONE PCA 3D (ANALISI DEGLI ERRORI) ===
pca = PCA(n_components=3, random_state=RANDOM_STATE)
X_test_pca = pca.fit_transform(X_test_proc)

corrette = y_pred == y_test.values

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X_test_pca[corrette, 0],
    X_test_pca[corrette, 1],
    X_test_pca[corrette, 2],
    c="green",
    s=40,
    alpha=0.5,
    label="Predizioni Corrette",
)
ax.scatter(
    X_test_pca[~corrette, 0],
    X_test_pca[~corrette, 1],
    X_test_pca[~corrette, 2],
    c="red",
    marker="x",
    s=80,
    alpha=1.0,
    label="Errori",
)

ax.set_title(f"Analisi Errori 3D PCA ({nome_miglior_modello})")
ax.legend()
plt.tight_layout()
plt.show()
