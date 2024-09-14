import numpy as np
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
from assets.utils import plot_results

# Set up the logger
logging.basicConfig(level=logging.DEBUG)

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
            self.best_models[name] = rs.best_estimator_
            y_pred = rs.predict(self.X_test)
            
            # Evaluate the model
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            logging.debug(f"Model: {name}")
            logging.debug(f"R2: {r2}")
            logging.debug(f"MSE: {mse}")
            logging.debug(f"MAE: {mae}")
            logging.debug(f"RMSE: {rmse}")

            plot_results(rs, self.df['biomass'], rs.predict(self.X_train), self.y_test)

            # Save model
            joblib.dump(rs.best_estimator_, f'{self.model_type}_{name}_model.pkl')