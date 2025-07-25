# 🩺 Diabetes Prediction with Machine Learning

A complete machine learning pipeline that uses clustering analysis followed by supervised classification to predict diabetes risk from health indicators.


## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Workflow](#-project-workflow)
- [Web Application](#-web-application)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contact](#-contact)


## 🎯 Project Overview

This project implements a comprehensive diabetes prediction system using a two-phase machine learning approach:

1. **Phase 1 - Clustering**: K-means clustering to identify patient groups with similar health profiles
2. **Phase 2 - Classification**: Supervised learning models to predict diabetes risk based on health indicators

**Objective**: Build an accurate diabetes prediction model that can classify patients into risk categories using key health metrics.

**Dataset**: Pima Indians Diabetes Dataset with 8 health-related features including glucose levels, BMI, age, and genetic factors.


## ✨ Features

- **K-means Clustering**: Patient segmentation based on health profiles
- **Multi-Algorithm Classification**: Random Forest, SVM, Gradient Boosting, Logistic Regression
- **Hyperparameter Optimization**: GridSearchCV for model tuning
- **Data Preprocessing**: StandardScaler normalization and class balancing
- **Cross-Validation**: 5-fold validation for robust evaluation
- **Interactive Web App**: Streamlit interface for real-time predictions
- **Complete Pipeline**: From data cleaning to model deployment



## 📊 Dataset

**Source**: Pima Indians Diabetes Dataset
- **Samples**: 768 patient records
- **Features**: 8 health indicators
- **Target**: Binary diabetes classification (0: No diabetes, 1: Diabetes)

**Key Features Used in Final Model**:
- `Glucose`: Plasma glucose concentration 
- `BMI`: Body mass index (weight in kg/(height in m)²)
- `Age`: Age in years
- `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic score)

**Data Files**:
- `dataset.csv`: Original Pima Indians diabetes dataset
- `df_cluster.csv`: Dataset with cluster assignments
- `df_cluster_final.csv`: Final processed dataset with cluster labels
- `df_cluster_scaled.csv`: Standardized features for clustering



## 🛠 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Git

### Required Libraries
```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn imbalanced-learn
```

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/Ilhamelgharbi/Diabetes-Prediction-with-Machine-Learning-Clustering-Classification.git
   cd Diabetes-Prediction-with-Machine-Learning-Clustering-Classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # if available
   # or install manually:
   pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn imbalanced-learn
   ```



## 🚀 Usage

### 1. Run the Analysis Notebooks

**Step 1: Data Cleaning**
```bash
jupyter notebook notebooks/01-clean_data.ipynb
```

**Step 2: Clustering Analysis**
```bash
jupyter notebook notebooks/02-clustring_kmeans.ipynb
```

**Step 3: Classification Model**
```bash
jupyter notebook notebooks/03-classification_supervisee_based_on_clusters.ipynb
```

### 2. Launch the Web Application

```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

### 3. Make Predictions

Use the Streamlit interface to input:
- Glucose level (0-200 mg/dL)
- BMI (10-60)
- Age (10-100 years)
- Diabetes Pedigree Function (0.0-2.5)



## 🔄 Project Workflow

### Phase 1: Data Preparation & Clustering
1. **Data Cleaning** (`01-clean_data.ipynb`)
   - Load and explore the Pima Indians diabetes dataset
   - Handle missing values and outliers
   - Data quality assessment

2. **Clustering Analysis** (`02-clustring_kmeans.ipynb`)
   - Apply K-means clustering to identify patient groups
   - Determine optimal number of clusters
   - Analyze cluster characteristics

### Phase 2: Classification
3. **Supervised Classification** (`03-classification_supervisee_based_on_clusters.ipynb`)
   - Feature selection: Glucose, BMI, Age, DiabetesPedigreeFunction
   - Train multiple algorithms: Random Forest, SVM, Gradient Boosting, Logistic Regression
   - Hyperparameter tuning with GridSearchCV
   - Model evaluation and selection
   - Save best model to `models/model.pkl`



## 🌐 Web Application

The project includes a professional Streamlit web application (`app.py`) for real-time diabetes risk prediction.

### Features:
- **User-friendly Interface**: Clean, intuitive design with sliders for input
- **Real-time Predictions**: Instant risk assessment based on health metrics
- **Risk Classification**: Binary output (Low Risk / High Risk)
- **Probability Scores**: Confidence levels for predictions
- **Health Recommendations**: Actionable advice based on results

### Interface Components:
- **Input Section**: 4 key health metrics with helpful descriptions
- **Prediction Engine**: Uses the trained model for classification
- **Results Display**: Clear risk assessment with probability scores
- **Recommendations**: Health guidance based on risk level

### Technical Implementation:
- Loads the trained model from `models/model.pkl`
- Handles data standardization automatically
- Provides fallback for missing scaler file
- Error handling for robust user experience



## 📁 Project Structure

```
Diabetes-Prediction-with-Machine-Learning-Clustering-Classification/
│
├── data/                           # Dataset files (tracked in git)
│   ├── dataset.csv                # Original Pima Indians diabetes dataset
│   ├── df_cluster.csv            # Data with cluster assignments  
│   ├── df_cluster_final.csv      # Final processed dataset
│   └── df_cluster_scaled.csv     # Standardized features
│
├── models/                         # Trained models (tracked in git)
│   └── model.pkl                 # Best trained classification model
│
├── notebooks/                      # Analysis notebooks (tracked in git)
│   ├── 01-clean_data.ipynb      # Data cleaning and exploration
│   ├── 02-clustring_kmeans.ipynb # K-means clustering analysis
│   └── 03-classification_supervisee_based_on_clusters.ipynb # Classification
│
├── app.py                         # Streamlit web application (tracked in git)
└── README.md                     # Project documentation
```

**Note**: Files tracked in git repository are the core components of the project.



## 📈 Results

### Model Performance
The classification pipeline successfully delivers:
- **Multi-algorithm comparison**: Random Forest, SVM, Gradient Boosting, and Logistic Regression
- **Optimized hyperparameters**: GridSearchCV for best model selection
- **Cross-validation**: 5-fold validation for robust performance assessment
- **Production-ready model**: Best model saved as `model.pkl` for deployment

### Key Achievements
- ✅ **Complete ML Pipeline**: From raw data to deployable model
- ✅ **Clustering Integration**: Patient segmentation enhances classification approach
- ✅ **Feature Engineering**: Identified 4 most predictive health indicators
- ✅ **Web Deployment**: Streamlit app for real-world usage
- ✅ **Robust Evaluation**: Cross-validation and multiple metrics

### Model Capabilities
The final model can predict diabetes risk using only 4 key health metrics:
- Glucose level (most important predictor)
- BMI (body composition indicator)  
- Age (demographic factor)
- Diabetes Pedigree Function (genetic predisposition)



## 📞 Contact

**Author**: Ilham El Gharbi

- **GitHub**: [@Ilhamelgharbi](https://github.com/Ilhamelgharbi)
- **Repository**: [Diabetes-Prediction-with-Machine-Learning-Clustering-Classification](https://github.com/Ilhamelgharbi/Diabetes-Prediction-with-Machine-Learning-Clustering-Classification)



## � Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Streamlit**: Web application framework
- **Jupyter Notebook**: Interactive development environment
- **Joblib**: Model serialization
- **Matplotlib & Seaborn**: Data visualization



*Project developed as part of machine learning sprint focusing on diabetes prediction using clustering and classification techniques.*
