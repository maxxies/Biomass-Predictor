import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
import logging

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLoader:
    """
    Class to load and preprocess data for training a machine learning model
    """
    def __init__(self, shapefile_path, raster_path):
        """
        Initialize the DataLoader object

        Parameters
        ----------
        shapefile_path : str
            Path to the shapefile containing the polygons
        raster_path : str
            Path to the raster file containing the data
        """
        self.shapefile_path = shapefile_path
        self.raster_path = raster_path

    def load_and_preprocess_data(self):
        """
        Load the shapefile and raster data, clip the polygons to the raster extent, and extract features for each polygon

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the extracted features and the target variable
        raster_data : numpy.ndarray
        raster_meta : dict
            Metadata of the raster file
        """
        # Load the shapefile
        gdf = gpd.read_file(self.shapefile_path)
        logging.debug(f"Number of polygons: {len(gdf)}")
        logging.debug(f"Shapefile CRS: {gdf.crs}")

        # Load the raster
        with rasterio.open(self.raster_path) as src:
            raster_data = src.read()
            raster_meta = src.meta
            raster_bounds = src.bounds
            raster_transform = src.transform

        logging.debug(f"Raster shape: {raster_data.shape}")
        logging.debug(f"Raster CRS: {raster_meta['crs']}")
        logging.debug(f"Number of bands in raster: {raster_data.shape[0]}")

        # Using the first 12 bands
        raster_data = raster_data[:12]

        # GeoDataFrame with the raster extent
        raster_extent = gpd.GeoDataFrame(
            {'geometry': [box(*raster_bounds)]}, crs=raster_meta['crs'])

        # Reproject the shapefile to match the raster CRS
        gdf = gdf.to_crs(raster_meta['crs'])

        # Clip the polygons to the raster extent
        gdf_clipped = gpd.overlay(gdf, raster_extent, how='intersection')
        logging.debug(f"Number of clipped polygons: {len(gdf_clipped)}")

        # Extract features for each polygon
        features = []
        for idx, row in gdf_clipped.iterrows():
            geom = row['geometry']
            if geom.is_empty:
                logging.warning(f"Geometry at index {idx} is empty")
                continue

            mask = geometry_mask([geom], out_shape=raster_data.shape[1:],
                                 transform=raster_transform, invert=True)
            masked_data = raster_data[:, mask]

            if masked_data.size > 0:
                feature_vector = masked_data.mean(axis=1)
                features.append(feature_vector)
            else:
                logging.warning(f"No data found for polygon at index {idx}")

        # Create a DataFrame with the extracted features
        feature_names = [f'B{i}' for i in range(1, raster_data.shape[0] + 1)]
        df = pd.DataFrame(features, columns=feature_names)
        df['biomass'] = gdf_clipped['biomass'][:len(features)]

        logging.debug(f"DataFrame shape: {df.shape}")
        logging.debug(f"Number of polygons with biomass values: {len(df)}")

        return df, raster_data, raster_meta
    
