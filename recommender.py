import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from preprocessor import RecipePreprocessor

class RecipeRecommender:
    def __init__(self, n_clusters: int = 5):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.preprocessor = RecipePreprocessor()
        
    def fit(self, df: pd.DataFrame):
        """Train the recommendation model"""
        self.df = df
        # Combine ingredients and instructions
        combined_features = (df['TranslatedIngredients'].fillna('') + ' ' + 
                           df['TranslatedInstructions'].fillna(''))
        
        # Preprocess and vectorize
        processed_features = [self.preprocessor.clean_text(text) for text in combined_features]
        self.feature_matrix = self.preprocessor.vectorize_text(processed_features)
        
        # Cluster recipes
        self.clusters = self.kmeans.fit_predict(self.feature_matrix)
        
    def get_recommendations(self, ingredients: str, n_recommendations: int = 3) -> List[Dict]:
        """Get personalized recipe recommendations"""
        # Preprocess input
        processed_input = self.preprocessor.clean_text(ingredients)
        input_vector = self.preprocessor.vectorizer.transform([processed_input])
        
        # Calculate similarities
        similarities = cosine_similarity(input_vector, self.feature_matrix).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recipe = {
                'title': self.df.iloc[idx]['TranslatedRecipeName'],
                'ingredients': self.df.iloc[idx]['TranslatedIngredients'],
                'instructions': self.df.iloc[idx]['TranslatedInstructions'],
                'cuisine': self.df.iloc[idx]['Cuisine'],
                'cooking_time': self.df.iloc[idx]['TotalTimeInMins'],
                'similarity_score': float(similarities[idx])  # Convert to float for JSON serialization
            }
            recommendations.append(recipe)
            
        return recommendations



