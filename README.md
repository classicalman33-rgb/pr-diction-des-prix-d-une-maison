# 🏠 Prédiction des prix d'une maison

Projet de régression supervisée : prédire le prix d'un logement à partir de ses caractéristiques (surface, nombre de chambres, équipements, etc.). Le projet couvre toute la chaîne : exploration des données (EDA), transformation de la cible, encodage, détection des outliers, diagnostic de multicolinéarité, entraînement de plusieurs modèles, réglage des hyperparamètres et évaluation.

---

## 📁 Structure du projet

```
prédiction des prix d'une maison/
├── .vscode/
│   └── settings.json          # Réglages VS Code (Jupyter)
├── data/
│   ├── Housing-selected-columns.csv
│   └── loading_data.py        # Chargement du CSV (data_loader)
├── notebook/
│   └── eda.ipynb              # Analyse exploratoire
├── src/
│   ├── config.py              # Script principal : orchestration du pipeline
│   ├── data_processing.py     # Prétraitement + split train/test
│   ├── outliers.py            # Détection des outliers (méthode IQR)
│   ├── find_best_params.py    # Recherche de alpha (RidgeCV / LassoCV)
│   ├── train_model.py         # Entraînement
│   └── evaluate_model.py      # Calcul des métriques
├── models/
│   ├── price.py               # Modèles sur la cible brute
│   ├── price_log.py           # Modèles sur la cible log
│   └── price_boxcox.py        # Modèles sur la cible Box-Cox
├── test/                      # Tests (pytest)
├── guide/
└── pyproject.toml             # (optionnel) projet installable en mode éditable
```

---

## 📊 Données

Fichier : `data/Housing-selected-columns.csv` (~545 lignes).

| Colonne | Type | Description |
|---|---|---|
| `price` | continue | **Cible** : prix du logement |
| `area` | continue | Surface |
| `bedrooms` | discrète | Nombre de chambres |
| `bathrooms` | discrète | Nombre de salles de bain |
| `stories` | discrète | Nombre d'étages |
| `mainroad` | binaire (yes/no) | Situé sur une route principale |
| `guestroom` | binaire (yes/no) | Chambre d'amis |
| `basement` | binaire (yes/no) | Sous-sol |
| `hotwaterheating` | binaire (yes/no) | Chauffage à eau chaude |
| `airconditioning` | binaire (yes/no) | Climatisation |

---

## 🔬 Démarche et résultats

### 1. Analyse de l'asymétrie (skewness) de `price`

| Transformation | Skewness |
|---|---|
| Aucune | 1.212 |
| Logarithme (`log1p`) | 0.141 |
| Box-Cox (λ = −0.135) | 0.003 |

- Box-Cox est statistiquement plus symétrique, mais le gain face au log est marginal.
- Le log est plus simple à expliquer et à inverser (`np.expm1`) ; Box-Cox demande de conserver λ et d'utiliser `inv_boxcox`.
- **Règle retenue** : on ne transforme que les variables **continues, positives, à forte cardinalité** (`price`, `area`). Pas les binaires ni les discrètes à faible cardinalité (`bedrooms`, `bathrooms`, `stories`).
- Le λ de Box-Cox doit être appris **sur le train uniquement**, puis réappliqué tel quel au test.

### 2. Corrélations avec le prix

| Variable | Corrélation avec `price` |
|---|---|
| `area` | 0.54 |
| `bathrooms` | 0.52 |
| `stories` | 0.42 |
| `bedrooms` | 0.37 |

Les corrélations `price` ↔ `price_log` ↔ `price_boxcox` (0.96–1.00) sont triviales (même variable transformée) et ne doivent pas être interprétées.

### 3. Encodage des variables catégorielles

```python
cat_cols = data.select_dtypes(include='object').columns.tolist()
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)
```

- `drop_first=True` évite le **piège des variables muettes** (corrélation parfaite de −1 entre `mainroad_no` et `mainroad_yes`, etc.).
- La conversion bool → int est nécessaire pour `.quantile()`, `statsmodels` (VIF) et la plupart des modèles.

### 4. Séparation features / cible

```python
X = data.drop(columns=['price', 'price_log', 'price_boxcox'])  # jamais une version du prix dans X
y = data['price']          # ou price_log / price_boxcox selon l'expérience
```

Garder `price_log` ou `price_boxcox` dans `X` provoquerait une **fuite de données** (data leakage).

### 5. Détection des outliers (méthode IQR)

| Colonne | Nb de valeurs aberrantes |
|---|---|
| `price` | 15 |
| `area` | 12 |
| `bedrooms` | 12 |
| `bathrooms` | 1 |
| `stories` | 41 |

- L'IQR n'est pertinent que pour des variables continues : les chiffres de `stories` et `bedrooms` sont **trompeurs** (peu de valeurs distinctes).
- Pour `price` et `area`, il s'agit de vrais logements haut de gamme, pas d'erreurs de saisie.
- **Décision** : on **conserve** ces lignes (2,7 % des données, les plus difficiles à prédire) ; la transformation log/Box-Cox atténue déjà leur effet.

### 6. Multicolinéarité (VIF)

| Variable | VIF |
|---|---|
| `bedrooms` | 15.16 |
| `bathrooms` | 9.33 |
| `stories` | 7.82 |
| `area` | 7.58 |
| `mainroad_yes` | 6.66 |
| `basement_yes` | 1.94 |
| `airconditioning_yes` | 1.72 |
| `guestroom_yes` | 1.46 |
| `hotwaterheating_yes` | 1.08 |

Multicolinéarité « naturelle » entre variables de taille du logement. Elle pénalise surtout l'interprétation des coefficients, peu la performance prédictive.

### 7. Split et réglage des hyperparamètres

- `train_test_split(..., test_size=0.2, random_state=42)` — **sans `stratify`** (réservé à la classification).
- Recherche de `alpha` pour Ridge et Lasso avec `RidgeCV` / `LassoCV` (`np.logspace(-3, 3, 50)`, `cv=5`), **uniquement sur le train**.

### 8. Métriques d'évaluation (régression)

`MAE`, `MSE`, `RMSE`, `R²` — les métriques de classification (`accuracy`, `f1`, `roc_auc`, `confusion_matrix`…) ne s'appliquent pas à une cible continue.

### 9. Résultats des modèles

**Cible `price_boxcox`** (métriques sur l'échelle transformée) :

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 0.0283 | 0.0353 | **0.6005** |
| Ridge | 0.0284 | 0.0355 | 0.5966 |
| Lasso | 0.0297 | 0.0368 | 0.5656 |

Toutes cibles confondues (`price`, `price_log`, `price_boxcox`), les modèles linéaires plafonnent autour de **R² ≈ 0,6** : la transformation de la cible n'est pas le facteur limitant.

**Random Forest** :

| Configuration | R² train | R² test |
|---|---|---|
| Hyperparamètres par défaut | 0.9377 | 0.5585 |
| Régularisé | — | **0.6798** |

Le Random Forest par défaut **sur-apprend** fortement (écart train/test de 0,38). En contraignant les arbres, il dépasse la régression linéaire, ce qui indique une composante non linéaire.

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features='sqrt',
    random_state=42,
)
```

---

## 🧠 Leçons apprises (pièges rencontrés)

| Problème | Cause | Solution |
|---|---|---|
| `Running cells requires the ipykernel package` | Kernel `python3.13t.exe` (version *free-threaded* expérimentale) | Choisir l'interpréteur `python.exe` standard (Global Env) |
| `ModuleNotFoundError: No module named 'data'` | Le dossier de travail est `notebook/` ou `src/`, pas la racine | `sys.path.append(racine)`, ou installer le projet avec `pip install -e .` |
| `jupyter.notebookFileRoot` instable | Réglage appliqué de façon incohérente selon la session | Ne pas s'y fier seul ; préférer `pyproject.toml` + `pip install -e .` |
| `'function' object has no attribute 'head'` | Module importé avant modification du fichier source | Redémarrer le kernel ou utiliser `%load_ext autoreload` / `%autoreload 2` |
| Explosion de colonnes après `get_dummies` | Sélection par position `data.columns[5:10]` | Sélectionner par nom ou par type (`select_dtypes`) |
| `numpy boolean subtract` dans `.quantile()` | Colonnes booléennes issues de `get_dummies` | Convertir en `int` |
| `ufunc 'isfinite' not supported` (VIF) | Types mixtes bool/object dans `X.values` | `X.astype(float)` |
| `The least populated class in y has only 1 member` | `stratify=y` sur une cible continue | Retirer `stratify` |
| `inconsistent numbers of samples: [436, 109]` | `y_test` passé au lieu de `y_train` à la recherche d'alpha | Utiliser `y_train` pendant tout le réglage |
| Transformations du notebook absentes dans `config.py` | Notebook et script ne partagent pas la mémoire | Coder le prétraitement dans une fonction réutilisable (`data_processing.py`) |
| `Type annotation not supported` | `;` au lieu de `,` dans un dictionnaire | Corriger la ponctuation |
| `No module named 'models'` (tests) | `models*` absent de `pyproject.toml` / pas de `__init__.py` | Ajouter `models*` à `include`, créer `models/__init__.py`, réinstaller |

---

## ⚙️ Installation et utilisation

```bash
# 1. (Recommandé) installer le projet en mode éditable depuis la racine
pip install -e .

# 2. Dépendances principales
pip install pandas numpy scipy scikit-learn statsmodels seaborn matplotlib ipykernel pytest
```

`pyproject.toml` minimal :

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "prediction-prix-maison"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["data*", "src*", "models*"]
```

Chaque dossier (`data/`, `src/`, `models/`) doit contenir un fichier `__init__.py` (même vide). Après modification du `pyproject.toml`, relancer `pip install -e .`.

**Lancer le pipeline** : `python src/config.py`
**Lancer les tests** : `pytest`
**Explorer** : ouvrir `notebook/eda.ipynb` avec un kernel `python.exe` standard.

---

## 🚀 Pistes d'amélioration

1. `GridSearchCV` sur le Random Forest (`max_depth`, `min_samples_leaf`, `min_samples_split`, `n_estimators`).
2. Tester `GradientBoostingRegressor` (souvent meilleur sur petits jeux de données).
3. Recalculer les métriques **sur l'échelle réelle** des prix (`np.expm1` ou `inv_boxcox`) pour obtenir un RMSE/MAE interprétable en unités monétaires.
4. Feature engineering : interactions (`area × bathrooms`), surface par chambre, etc.
5. Analyse des résidus (résidus vs valeurs prédites) pour détecter d'éventuelles non-linéarités.
6. Comparer les modèles par validation croisée plutôt que sur un seul split.
7. Documenter la limite du jeu de données : sans localisation, année de construction ni état du bien, un R² autour de 0,6–0,7 est un plafond plausible.
