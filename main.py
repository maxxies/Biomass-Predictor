from assets.data_loader import DataLoader
import logging

# Set up the logger
logging.basicConfig(level=logging.DEBUG)


# Load the data
shapefile_path = "data/shapefile.shp"
raster_path = "data/raster.tif"
data_loader = DataLoader(shapefile_path, raster_path)
df, raster_data, raster_meta = data_loader.load_and_preprocess_data()
