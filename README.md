# Disease Forecaster App

## Overview
The Disease Forecaster App is a web application developed using Streamlit for predicting two major health conditions: Brain Tumor Detection and Heart Disease Prediction. The app leverages machine learning models trained on medical data to provide real-time predictions based on user input or uploaded images.

## Features
- **Heart Disease Prediction:** Predicts the likelihood of heart disease based on various medical attributes such as age, blood pressure, cholesterol levels, and more.
- **Brain Tumor Detection:** Classifies MRI images to detect different types of brain tumors including Meningioma, Glioma, and Pituitary tumors.

## Methodology

### Data Collection and Preparation
- **Heart Disease Prediction:** Used a Heart Disease dataset from Kaggle containing various attributes related to heart health. Data preprocessing involved elementary data analysis, handling missing values, encoding categorical variables, and scaling numerical features.
- **Brain Tumor Detection:** Utilized a dataset of Brain MRI images with labels for different brain tumor types. Images were preprocessed using transformations such as resizing and normalization, and they were further augmented to increase the size of the dataset.

### Model Training
- **Heart Disease Prediction Models:** Trained several models including Logistic Regression, Guassian Naive Bayes, KNN, Random Forest, XGBoost, and Support Vector Machines (SVM). Achieved a Mean ROC AUC Score of 0.91 from the Naive Bayes and SVM models, indicating good predictive performance.
- **Brain Tumor Detection Model:** Developed a Convolutional Neural Network (CNN) using PyTorch achieving a test accuracy of 99% on the validation dataset. The CNN architecture consisted of multiple convolutional layers with batch normalization, followed by fully connected layers for classification.

### Deployment
- **Web Application:** Developed a user-friendly web interface using Streamlit to allow users to interact with the models. Integrated model predictions into the app to provide instantaneous results.

## Models Used
- **Logistic Regression:** Baseline model for heart disease prediction.
- **Naive Bayes:** Probabilistic classifier based on Bayes' theorem with strong independence assumptions between features.
- **Random Forest:** Ensemble learning method known for robustness and accuracy, by utilizing multiple decision trees.
- **XGBoost:** Gradient boosting algorithm optimized for performance.
- **Support Vector Machines (SVM):** Effective for classification tasks with high-dimensional data.
- **Convolutional Neural Network (CNN):** Deep learning model tailored for image classification tasks.

## Usage
To run the Disease Forecaster App locally:
1. Clone the repository.
2. Install the required dependencies, in requirements.txt.
3. Run the streamlit app by :  streamlit run combined.py, in the project folder.
4. Access the app in your browser at http://localhost:8501.



