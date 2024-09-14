from sklearn.decomposition import PCA
import logging
from assets.data_loader import DataLoader
from assets.feature_engineering import FeatureGeneration
from assets.utils import data_split
from assets.model import StatisticalModels

# Set up the logger
logging.basicConfig(level=logging.DEBUG)


# Load the data
shapefile_path = "data/shapefile.shp"
raster_path = "data/raster.tif"
data_loader = DataLoader(shapefile_path, raster_path)
df, raster_data, raster_meta = data_loader.load_and_preprocess_data()
logging.debug(f"Raster data shape: {raster_data.shape}")
logging.debug(f"DataFrame shape: {df.shape}")
logging.debug(f"Raster metadata: {raster_meta}")


# Feature engineering
feature_generator = FeatureGeneration(df)
new_df = feature_generator.get_df()
logging.debug(f"DataFrame shape after feature engineering: {new_df.shape}")
logging.debug(f"Columns after feature engineering: {new_df.columns}")

# Split the data
X_train, X_test, y_train, y_test = data_split(df)
new_X_train, new_X_test, new_y_train, new_y_test = data_split(new_df)


# Reduce the number of features using PCA for initial features
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

logging.debug(f"Number of features before PCA: {X_train.shape[1]}")
logging.debug(f"Number of features after PCA: {X_train_pca.shape[1]}")

# Reduce the number of features using PCA for new features
pca = PCA(n_components=0.95)
new_X_train_pca = pca.fit_transform(new_X_train)
new_X_test_pca = pca.transform(new_X_test)

logging.debug(f"Number of features before PCA: {new_X_train.shape[1]}")
logging.debug(f"Number of features after PCA: {new_X_train_pca.shape[1]}")


#  Model training
logging.debug("Training models on initial features")
models = StatisticalModels(df, X_train, X_test, y_train, y_test, 'initial') 
models.fit_models()

logging.debug("Training models on reduced initial features")
reduced_models = StatisticalModels(df, X_train_pca, X_test_pca, y_train, y_test, 'reduced_initial')
reduced_models.fit_models()

logging.debug("Training models on new features")
new_models = StatisticalModels(new_df, new_X_train, new_X_test, new_y_train, new_y_test, 'new')
new_models.fit_models()

logging.debug("Training models on reduced new features")
new_reduced_models = StatisticalModels(new_df, new_X_train_pca, new_X_test_pca, new_y_train, new_y_test, 'reduced_new')
new_reduced_models.fit_models()










