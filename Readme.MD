# 🩺 Diabetes Prediction with Machine Learning

## Description du projet

Ce projet propose une solution complète de prédiction du risque de diabète à partir d’indicateurs de santé. Il combine une phase de segmentation des patients (clustering K-means) puis une classification supervisée pour prédire le profil de risque d’un nouvel individu. L’application finale permet une utilisation simple via une interface web (Streamlit).

**Objectif :** Classer automatiquement les patients en deux groupes de risque (faible/élevé) à partir de 4 mesures médicales : Glucose, BMI, Age, DiabetesPedigreeFunction.

## Structure des fichiers

```
Diabetes-Prediction-with-Machine-Learning-Clustering-Classification/
│
├── data/
│   ├── dataset.csv                # Données brutes
│   ├── df_cluster.csv             # Données nettoyées/features sélectionnées
│   ├── df_cluster_final.csv       # Données avec clusters et catégories de risque
│   └── df_cluster_scaled.csv      # Données standardisées
│
├── models/
│   └── model.pkl                  # Modèle ML entraîné et optimisé
│
├── notebooks/
│   ├── 01-clean_data.ipynb        # Nettoyage et préparation des données
│   ├── 02-clustring_kmeans.ipynb  # Clustering K-means et analyse des groupes
│   └── 03-classification_supervisee_based_on_clusters.ipynb # Classification supervisée
│
├── app.py                        # Application web Streamlit pour la prédiction
└── RAPPORT_TECHNIQUE.md
└── README.md                     # Documentation du projet
```

## Instructions pour l’exécuter

### 1. Prérequis
- Python 3.7 ou plus
- Jupyter Notebook
- Streamlit

### 2. Installation des dépendances

Ouvrir un terminal à la racine du projet et exécuter :

```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn imbalanced-learn
```

### 3. Lancer l’analyse (notebooks)

Ouvrir les notebooks dans l’ordre :

1. Nettoyage des données :
   ```bash
   jupyter notebook notebooks/01-clean_data.ipynb
   ```
2. Clustering K-means :
   ```bash
   jupyter notebook notebooks/02-clustring_kmeans.ipynb
   ```
3. Classification supervisée :
   ```bash
   jupyter notebook notebooks/03-classification_supervisee_based_on_clusters.ipynb
   ```

### 4. Lancer l’application web

```bash
streamlit run app.py
```
L’application sera accessible sur http://localhost:8501

### 5. Prédire le risque de diabète

Utiliser l’interface Streamlit pour saisir :
- Glucose
- BMI
- Age
- DiabetesPedigreeFunction

Le modèle prédit instantanément le profil de risque du patient.


## Résumé du rapport technique

- **Prétraitement** : Nettoyage, sélection de 4 variables clés, standardisation, équilibrage des classes.
- **Clustering** : K-means (k=2) pour segmenter les patients en deux groupes de risque.
- **Classification** : Entraînement et optimisation de 4 modèles (Random Forest, SVM, Gradient Boosting, Régression Logistique). Sélection du meilleur modèle (SVM, F1-score ≈ 0.99).
- **Validation** : Validation croisée 5-fold, matrice de confusion quasi-parfaite.
- **Déploiement** : Modèle sauvegardé et utilisable via l’application web.

*Projet réalisé dans le cadre d’un sprint de formation Data Science.*
