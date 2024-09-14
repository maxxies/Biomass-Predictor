import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



def data_split(df: pd.DataFrame, test_size: float = 0.2):   
    """
    Split the data into training and testing sets
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features and the target variable
    
    Returns
    ----------
    X_train : pandas.DataFrame
        DataFrame containing the features of the training set
    X_test : pandas.DataFrame
        DataFrame containing the features of the testing set
    y_train : pandas.Series
        Series containing the target variable of the training set
    y_test : pandas.Series
        Series containing the target variable of the testing set
    """
    X = df.drop(columns=['biomass'])
    y = df['biomass']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def features_reduction(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.95):
    """
    Reduce the number of features using PCA

    Parameters
    ----------
    X_train : pandas.DataFrame
        DataFrame containing the features of the training set
    X_test : pandas.DataFrame
        DataFrame containing the features of the testing set
    threshold : float
        Threshold for the explained variance ratio

    Returns
    ----------
    X_train : pandas.DataFrame
        DataFrame containing the features of the training set after PCA
    X_test : pandas.DataFrame
        DataFrame containing the features of the testing set after PCA
    """
    pca = PCA(n_components=threshold)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    logger.debug(f"Number of features after PCA: {X_train.shape[1]}")
    logger.debug(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return X_train, X_test


def plot_results(model, y_true, y_pred_train, y_pred_test):
    """
    Plot the actual and predicted biomass values

    Parameters
    ----------
    model : sklearn model
        Trained model
    y_true : numpy.ndarray
        Actual values of the target variable
    y_pred_train : numpy.ndarray
        Predicted values of the training set
    y_pred_test : numpy.ndarray
        Predicted values of the testing set
    """

    plt.figure(figsize=(15, 10))

    y_pred_combined = np.concatenate([y_pred_train, y_pred_test])

    # Create an index array for x-axis
    x = np.arange(len(y_true))

    plt.plot(x, y_true, marker='o', linestyle='-',
             color='orange', label='Actual')
    plt.plot(x, y_pred_combined[:len(y_true)], marker='o',
             linestyle='-', color='blue', label='Predicted')

    plt.xlabel('Index')
    plt.ylabel('Biomass Values')
    plt.title(f'{model.__class__.__name__} Biomass Prediction')
    plt.legend()

    plt.tight_layout()
    plt.show()
