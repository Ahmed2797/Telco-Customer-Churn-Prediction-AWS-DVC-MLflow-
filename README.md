# Telco Customer Churn Prediction (AWS + DVC + MLflow)

Developed an end-to-end ML pipeline to predict telecom customer churn. Implemented DVC for data versioning, MLflow for experiment tracking, and deployed the best model on AWS. Optimized features, trained multiple classifiers, and delivered a reproducible, production-ready solution for data-driven retention strategies.

## 📌 Project Overview

This project is an end-to-end Machine Learning pipeline for predicting customer churn in a telecom company. It identifies customers who are likely to leave the service based on historical behavior, service usage, and account information. The project is designed with production readiness and reproducibility in mind.

## 📂 Dataset

    https://drive.google.com/file/d/1hA0hdGr_mzWsrnRgQkzN5uwYmht7l3bC/view?usp=sharing
    

## 🎯 Objective

* Predict customer churn using machine learning algorithms.
* Identify important features influencing churn.
* Provide actionable insights for the bank to reduce churn.

---

## 🛠️ Methodology

1. **Data Exploration & Visualization**

   * Check for missing values, distributions, and correlations.
   * Visualize churn patterns using plots.

2. **Data Preprocessing**

   * Encode categorical variables.
   * Scale numerical features.
   * Split data into training and test sets.

3. **Modeling**

   * Algorithms used:

     * Logistic Regression
     * Decision Tree
     * Random Forest
     * XGBoost / Gradient Boosting
   * Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

4. **Evaluation Metrics**

   * Accuracy
   * Precision, Recall, F1-Score
   * ROC-AUC score

5. **Feature Importance**

   * Identify key features contributing to churn prediction.

---

## ⚙️ Installation

Clone the repository and install dependencies:

    # Clone the repository
    git clone https://github.com/Ahmed2797/Telco-Customer-Churn-Prediction-AWS-DVC-MLflow-.git
    cd bank-churn-modelling

    # Create and activate a conda environment
    conda create -n ml python=3.10
    conda activate ml

    # setup
    python setup.py install

    # Install dependencies
    pip install -r requirements.txt
    
    # Push data Mongo
    python push_data_mongo.py

---

## 🏃 How to Run

1. Launch Jupyter Notebook or Python scripts.
2. Follow the notebooks for:

   * Data exploration
   * Preprocessing
   * Model training and evaluation
3. Visualize results and feature importance.

---

## 📈 Key Features

* Clean and modular **end-to-end ML pipeline**
* Multiple algorithms for prediction
* Hyperparameter tuning for optimal performance
* Feature importance insights for business decisions
* Ready for **deployment / integration**

---

## 📬 Contact

**Author:** github.com/Ahmed2797

**Interest:** Machine Learning, Data Science, Predictive Modelling

---

⭐ If you find this project helpful, give it a star on GitHub!
