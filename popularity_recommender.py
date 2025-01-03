import pandas as pd

# Popularity-based Recommender System
class PopularityRecommender:
    def __init__(self, dataframe):
        """
        Initialize the recommender with a DataFrame.
        Args:
            dataframe (pd.DataFrame): DataFrame containing product data.
        """
        self.df = dataframe

    def get_top_products(self, 
                         gender=None, 
                         category=None, 
                         sub_category=None, 
                         color=None, 
                         n=10):
        """
        Retrieve the top N products based on popularity score.
        Args:
            gender (str): Filter products by gender.
            category (str): Filter products by masterCategory.
            sub_category (str): Filter products by subCategory.
            color (str): Filter products by baseColour.
            n (int): Number of top products to return.
        Returns:
            pd.DataFrame: Top N products sorted by popularity_score.
        """
        # Create a copy of the DataFrame
        df_filtered = self.df.copy()

        # Ensure columns exist and handle missing values
        required_columns = ['gender', 'masterCategory', 'subCategory', 'baseColour', 'rating', 'num_ratings']
        for column in required_columns:
            if column not in df_filtered.columns:
                raise ValueError(f"Missing required column: {column}")
        
        df_filtered = df_filtered.fillna({
            'gender': '', 
            'masterCategory': '', 
            'subCategory': '', 
            'baseColour': '',
            'rating': 0, 
            'num_ratings': 0
        })

        # Apply filters if specified
        if gender:
            df_filtered = df_filtered[df_filtered['gender'].str.lower() == gender.lower()]
        if category:
            df_filtered = df_filtered[df_filtered['masterCategory'].str.lower() == category.lower()]
        if sub_category:
            df_filtered = df_filtered[df_filtered['subCategory'].str.lower() == sub_category.lower()]
        if color:
            df_filtered = df_filtered[df_filtered['baseColour'].str.lower() == color.lower()]

        # Calculate the popularity score
        df_filtered['popularity_score'] = df_filtered['rating'] * df_filtered['num_ratings']

        # Sort by popularity score and return the top N products
        top_products = df_filtered.nlargest(n, 'popularity_score', 'all')

        return top_products[['id', 'productDisplayName', 'gender', 
                             'masterCategory', 'subCategory', 
                             'baseColour', 'price', 'rating', 
                             'num_ratings', 'popularity_score']]
