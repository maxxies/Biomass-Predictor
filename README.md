# Biomass-Predictor

The repository contains for training models to predict biomass from plots from a multispectral image. 

# Code Structure
```
.
├── assets
|   ├── data_loader.py
|   ├── feature_engineering.py
|   ├── model.py
|   └── utils.py
├── outputs
|   ├── models
|   └── plots
├── .gitignore
├── License
├── main.py
└── README.md
```
1. `assets/`: Contains all the helper functions for data loading, feature engineering, model training and evaluation.
2. `main.py`: Main script to run the model training and evaluation.
3. `README.md`: Contains the information about the repository.

# Model Training & Experimentations

1. The dataset is split into 90% training and 10% testing.

2. Features are standardized using StandardScaler.

3. Features engineering is done to create new features creating another dataset containing initial features and the new features. New features include:
    -  Enhanced Vegetation Index (EVI)
    -  Soil Adjusted Vegetation Index (SAVI)
    -  Normalized Difference Water Index (NDWI)
    -  Simple Ratio (SR)
    -  Green Chlorophyll Index (GCI)
    -  Normalized Difference Red Edge (NDRE)
    -  Normalized Difference Vegetation Index (NDVI)

4. Dimensionality reduction is done using **PCA** to select the most important features for both datasets.

3. Models are trained on the features and the reduced features for each of the datasets.

4. The models are evaluated using the **Mean Squared Error**, **R2 Score**, **Mean Absolute Error** and **Root Mean Squared Error**.

5. Models used for training are:
    - Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Elastic Net
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    - XGBoost

# Results

The results of the models are as follows:

## 1. Initial Dataset

| Model                   | MSE    | R2 Score | MAE    | RMSE   |
|-------------------------|--------|----------|--------|--------|
| Linear Regression       | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Lasso Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Ridge Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Elastic Net             | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Decision Tree           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Random Forest           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Support Vector Machine  | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| XGBoost                 | 0.0001 | 0.9999   | 0.0001 | 0.0100 |

## 2. Reduced Features for Initial Dataset

| Model                   | MSE    | R2 Score | MAE    | RMSE   |
|-------------------------|--------|----------|--------|--------|
| Linear Regression       | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Lasso Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Ridge Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Elastic Net             | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Decision Tree           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Random Forest           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Support Vector Machine  | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| XGBoost                 | 0.0001 | 0.9999   | 0.0001 | 0.0100 |

## 3. New Features Dataset

| Model                   | MSE    | R2 Score | MAE    | RMSE   |
|-------------------------|--------|----------|--------|--------|
| Linear Regression       | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Lasso Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Ridge Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Elastic Net             | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Decision Tree           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Random Forest           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Support Vector Machine  | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| XGBoost                 | 0.0001 | 0.9999   | 0.0001 | 0.0100 |

## 4. Reduced Features for New Features Dataset

| Model                   | MSE    | R2 Score | MAE    | RMSE   |
|-------------------------|--------|----------|--------|--------|
| Linear Regression       | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Lasso Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Ridge Regression        | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Elastic Net             | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Decision Tree           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Random Forest           | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| Support Vector Machine  | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
| XGBoost                 | 0.0001 | 0.9999   | 0.0001 | 0.0100 |
# Installation & Usage
1. Clone the repository
```bash
git clone <path>
```
2. Install the required packages
```bash
pip install -r requirements.txt
```
3. Run the main script

