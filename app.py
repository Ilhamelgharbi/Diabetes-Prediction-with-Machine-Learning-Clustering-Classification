
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# --- Chargement du modèle et scaler sauvegardés ---
model = joblib.load("models/model.pkl")

scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    scaler_loaded = True
    st.sidebar.success("✅ Scaler chargé depuis scaler.pkl")
else:
    scaler = None
    scaler_loaded = False
    st.sidebar.error("❌ Scaler non trouvé ! Veuillez placer 'scaler.pkl' dans le dossier models.")
    st.sidebar.info("Dans votre notebook d'entraînement, ajoutez : joblib.dump(scaler, '../models/scaler.pkl') juste après l'entraînement du scaler.")

# --- Titre ---
st.title("🩺 Prédiction du Risque de Diabète")

st.markdown("""
Ce simulateur prédit si une personne appartient à un **groupe à risque élevé** ou **faible** de diabète, 
en se basant sur les caractéristiques de santé renseignées ci-dessous.

**Modèle utilisé :** Classification supervisée optimisée avec GridSearchCV
""")

# --- Informations sur le modèle ---
with st.expander("ℹ️ Informations sur le modèle"):
    st.write("""
    - **Type de modèle :** Modèle de classification optimisé
    - **Variables d'entrée :** 4 caractéristiques de santé
    - **Features utilisées :** Glucose, BMI, Age, DiabetesPedigreeFunction
    - **Sortie :** Probabilité de risque de diabète (0 = Faible, 1 = Élevé)
    - **Préprocessing :** Standardisation avec StandardScaler
    """)

# --- Formulaire utilisateur ---
st.header("📋 Entrez les données de la personne :")

st.info("ℹ️ Ce modèle utilise 4 caractéristiques principales pour la prédiction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Données médicales")
    glucose = st.slider("Taux de Glucose (mg/dL)", 0, 200, 100, help="Concentration de glucose dans le sang")
    bmi = st.slider("Indice de Masse Corporelle (BMI)", 10.0, 60.0, 25.0, help="BMI = poids(kg) / taille(m)²")

with col2:
    st.subheader("👤 Données personnelles")
    age = st.slider("Âge (années)", 10, 100, 35, help="Âge de la personne")
    dpf = st.slider("Fonction génétique (DPF)", 0.0, 2.5, 0.5, help="Diabetes Pedigree Function - prédisposition génétique")

# --- Prédiction ---
st.header("🔍 Prédiction du Risque")

if st.button("🔍 Prédire le risque", type="primary"):
    # Vérification de la présence du scaler
    if not scaler_loaded or scaler is None:
        st.error("❌ Impossible de prédire : le scaler pré-entraîné est requis.")
        st.info("Veuillez placer le fichier 'scaler.pkl' dans le dossier models pour garantir la cohérence des prédictions.")
    else:
        # Création du DataFrame avec les données d'entrée (ordre des features doit correspondre à l'entraînement)
        input_data = pd.DataFrame([[glucose, bmi, age, dpf]], columns=['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction'])

        # Affichage des données saisies
        with st.expander("📊 Données saisies"):
            st.dataframe(input_data)

        try:
            # Standardisation des données
            input_data_scaled = scaler.transform(input_data)
            st.success("✅ Données standardisées avec le scaler pré-entraîné")

            # Vérification de la forme d'entrée pour le modèle
            if input_data_scaled.shape[1] != model.n_features_in_:
                st.error(f"❌ Le modèle attend {model.n_features_in_} features, mais {input_data_scaled.shape[1]} ont été fournis.")
                st.info("Vérifiez l'ordre et le nombre de variables d'entrée.")
            else:
                # Prédiction
                prediction = model.predict(input_data_scaled)
                prob = model.predict_proba(input_data_scaled)[0]

                # Affichage des résultats
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if prediction[0] == 1:
                        st.error(f"⚠️ **RISQUE ÉLEVÉ de diabète**")
                        st.error(f"Probabilité : **{prob[1]:.2%}**")
                        st.markdown("""
                        **Recommandations :**
                        - Consultez un professionnel de santé
                        - Surveillez votre alimentation
                        - Pratiquez une activité physique régulière
                        """)
                    else:
                        st.success(f"✅ **RISQUE FAIBLE de diabète**")
                        st.success(f"Probabilité de non-diabète : **{prob[0]:.2%}**")
                        st.markdown("""
                        **Conseils de prévention :**
                        - Maintenez un mode de vie sain
                        - Contrôles médicaux réguliers
                        - Alimentation équilibrée
                        """)

                # Détails des probabilités
                st.markdown("---")
                st.subheader("📈 Détail des probabilités")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité Risque Faible", f"{prob[0]:.2%}")
                with col2:
                    st.metric("Probabilité Risque Élevé", f"{prob[1]:.2%}")
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            st.info("Vérifiez que le modèle et le scaler sont compatibles avec les données d'entrée.\nSi le problème persiste, assurez-vous d'avoir sauvegardé le scaler avec joblib.dump(scaler, 'models/scaler.pkl') dans votre notebook d'entraînement.")
