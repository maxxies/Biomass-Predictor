from sklearn.decomposition import PCA
import logging
from assets.data_loader import DataLoader
from assets.feature_engineering import FeatureGeneration
from assets.utils import data_split
from assets.model import StatisticalModels, NeuralNetwork

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Load the data
shapefile_path = "data/shapefile.dbf"
raster_path = "data/raster.tif"

data_loader = DataLoader(shapefile_path, raster_path)
df, raster_data, raster_meta = data_loader.load_and_preprocess_data()
logger.debug(f"Raster data shape: {raster_data.shape}")
logger.debug(f"DataFrame shape: {df.shape}")
logger.debug(f"Raster metadata: {raster_meta}")


# Feature engineering
feature_generator = FeatureGeneration(df)
updated_df = feature_generator.get_df()
logger.debug(f"DataFrame shape after feature engineering: {updated_df.shape}")
logger.debug(f"Columns after feature engineering: {updated_df.columns}")

# Split the data
X_train, X_test, y_train, y_test = data_split(df)
updated_X_train, updated_X_test, updated_y_train, updated_y_test = data_split(updated_df)


# Reduce the number of features using PCA for initial features
logger.debug("Reducing the number of initial features using PCA")

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

logger.debug(f"Number of features before PCA: {X_train.shape[1]}")
logger.debug(f"Number of features after PCA: {X_train_pca.shape[1]}")

# Reduce the number of features using PCA for new features
logger.debug("Reducing the number of updated features using PCA")

pca = PCA(n_components=0.95)
updated_X_train_pca = pca.fit_transform(updated_X_train)
updated_X_test_pca = pca.transform(updated_X_test)

logger.debug(f"Number of features before PCA: {updated_X_train.shape[1]}")
logger.debug(f"Number of features after PCA: {updated_X_train_pca.shape[1]}")


#  Model training
logger.debug("Training models on initial features")
models = StatisticalModels(df, X_train, X_test, y_train, y_test, 'initial') 
models.fit_models()
nn_model = NeuralNetwork(df, X_train, X_test, y_train, y_test, 'initial')
nn_model.nn_train()

logger.debug("Training models on reduced initial features")
reduced_models = StatisticalModels(df, X_train_pca, X_test_pca, y_train, y_test, 'reduced_initial')
reduced_models.fit_models()
reduced_nn_model = NeuralNetwork(df, X_train_pca, X_test_pca, y_train, y_test, 'reduced_initial')
reduced_nn_model.nn_train()


logger.debug("Training models on updated features")
updated_models = StatisticalModels(updated_df, updated_X_train, updated_X_test, updated_y_train, updated_y_test, 'updated')
updated_models.fit_models()
updated_nn_model = NeuralNetwork(updated_df, updated_X_train, updated_X_test, updated_y_train, updated_y_test, 'updated')
updated_nn_model.nn_train()


logger.debug("Training models on reduced updated features")
updated_reduced_models = StatisticalModels(updated_df, updated_X_train_pca, updated_X_test_pca, updated_y_train, updated_y_test, 'reduced_updated')
updated_reduced_models.fit_models()
updated_reduced_nn_model = NeuralNetwork(updated_df, updated_X_train_pca, updated_X_test_pca, updated_y_train, updated_y_test, 'reduced_updated')
updated_reduced_nn_model.nn_train()











