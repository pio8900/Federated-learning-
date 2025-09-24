# Federated Learning for Heart Disease Prediction

A comprehensive federated learning implementation that trains machine learning models across distributed clients for heart disease prediction while preserving data privacy and comparing multiple algorithms.

## 🎯 Project Overview

This project implements **federated learning** using the Heart Disease dataset, simulating a real-world scenario where multiple hospitals collaborate to build better predictive models without sharing sensitive patient data. The system trains three different machine learning algorithms (Logistic Regression, SVM, Random Forest) across distributed clients and aggregates them into robust global models.

## ✨ Key Features

- **Multi-Algorithm Support**: Logistic Regression, Support Vector Machine (SVM), and Random Forest
- **Federated Averaging**: Custom implementation for model aggregation across clients
- **Privacy Preservation**: Data remains distributed across simulated clients
- **Automated Pipeline**: End-to-end execution with single command
- **Model Evaluation**: Comprehensive accuracy assessment for all algorithms
- **Medical Application**: Heart disease prediction using cardiovascular health indicators

## 📊 Dataset

**Heart Disease Dataset** containing 303 patient records with 13 clinical features:
- **Age**: Patient age in years
- **Sex**: Gender (1 = male, 0 = female) 
- **CP**: Chest pain type (0-3)
- **Trestbps**: Resting blood pressure
- **Chol**: Serum cholesterol level
- **FBS**: Fasting blood sugar > 120 mg/dl
- **Restecg**: Resting electrocardiographic results
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise induced angina
- **Oldpeak**: ST depression induced by exercise
- **Slope**: Slope of peak exercise ST segment
- **CA**: Number of major vessels colored by fluoroscopy
- **Thal**: Thalassemia type
- **Output**: Target variable (1 = heart disease, 0 = no heart disease)

## 🚀 Quick Start

### Prerequisites
