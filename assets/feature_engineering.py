
class FeatureGeneration:
    """
    This class is used to generate new features from the original bands of the satellite images.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the original bands of the satellite images 

    Returns
    ----------
    df : pandas.DataFrame
        DataFrame containing the original bands of the satellite images and the new features
    """
    def __init__(self, df):
        self.df = df.copy()
        self.add_features()

    def add_features(self):
        """
        Add new features to the DataFrame
        """

        # Enhanced Vegetation Index (EVI)
        self.df['EVI'] = 2.5 * ((self.df['B8'] - self.df['B4']) / (self.df['B8'] +
                        6 * self.df['B4'] - 7.5 * self.df['B2'] + 1))

        # Soil Adjusted Vegetation Index (SAVI)
        self.df['SAVI'] = ((self.df['B8'] - self.df['B4']) / (self.df['B8'] + self.df['B4'] + 0.5)) * 1.5

        # Normalized Difference Water Index (NDWI)
        self.df['NDWI'] = (self.df['B3'] - self.df['B8']) / (self.df['B3'] + self.df['B8'])

        # Simple Ratio (SR)
        self.df['SR'] = self.df['B8'] / self.df['B4']

        # Green Chlorophyll Index (GCI)
        self.df['GCI'] = (self.df['B8'] / self.df['B3']) - 1

        # Normalized Difference Red Edge (NDRE)
        self.df['NDRE'] = (self.df['B8'] - self.df['B5']) / (self.df['B8'] + self.df['B5'])

    def get_df(self):
        """ Return the DataFrame with the new features """
        return self.df


    