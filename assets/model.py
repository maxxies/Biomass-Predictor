import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from assets.utils import plot_results

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class StatisticalModels:
    def __init__(self,df, X_train, X_test, y_train, y_test, model_type):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type

        self.models = {
            'lr': (LinearRegression(), {}),
            'lasso': (Lasso(), {'alpha': np.logspace(-4, 1, 50)}),
            'ridge': (Ridge(), {'alpha': np.logspace(-4, 1, 50)}),
            'elastic': (ElasticNet(), {'alpha': np.logspace(-4, 1, 50), 'l1_ratio': np.linspace(0.1, 0.9, 9)}),
            'dt': (DecisionTreeRegressor(), {'max_depth': [3, 5, 7, 9, 11], 'min_samples_split': [2, 5, 10]}),
            'rf': (RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15, None]}),
            'svr': (SVR(), {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}),
            'xgb': (XGBRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]})
        }

    def fit_models(self):
        for name, (model, param_grid) in self.models.items():
            rs = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            rs.fit(self.X_train, self.y_train)
            y_pred = rs.predict(self.X_test)
            
            # Evaluate the model
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            logger.debug(f"Model: {name}")
            logger.debug(f"R2: {r2}")
            logger.debug(f"MSE: {mse}")
            logger.debug(f"MAE: {mae}")
            logger.debug(f"RMSE: {rmse}")

            plot_results(rs, self.df['biomass'], rs.predict(self.X_train), self.y_test)

            # Save model
            os.makedirs('/output', exist_ok=True)
            joblib.dump(rs.best_estimator_, f'/output/{self.model_type}_{name}_model.pkl')

class NeuralNetwork:    
    def __init__(self, df, X_train, X_test, y_train, y_test, model_type):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type

    def nn_train(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss='mae',
                      metrics=['mse', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.R2Score()])
        

        tf.keras.utils.plot_model(model, to_file='model.png',
                                  show_shapes=True, show_layer_names=True)

        history = model.fit(self.X_train, self.y_train, epochs=100,
                            batch_size=16, validation_split=0.1)

        # Evaluate the model
        loss, mae, rmse, r2 = model.evaluate(self.X_test, self.y_test)
        logger.debug(f"Loss(MSE): {loss}")
        logger.debug(f"MAE: {mae}")
        logger.debug(f"RMSE: {rmse}")
        logger.debug(f"R2: {r2}")

        # Make predictions
        y_pred_nn = model.predict(self.X_test)

        # Plot the results for the neural network
        plot_results(model, self.df['biomass'], model.predict(self.X_train), y_pred_nn)

        # Save the model
        os.makedirs('/output', exist_ok=True)
        model.save(f'/output/{self.model_type}_nn_model.keras')