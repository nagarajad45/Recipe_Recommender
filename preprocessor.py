import re
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

class RecipePreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            token_pattern=r'[a-zA-Z]+'
        )
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredients from cleaned ingredients text"""
        if not isinstance(text, str):
            return []
        ingredients = [ing.strip() for ing in text.split(',')]
        return [ing for ing in ingredients if ing]
    
    def vectorize_text(self, texts: List[str]) -> np.ndarray:
        """Convert text to TF-IDF vectors"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.fit_transform(cleaned_texts)


