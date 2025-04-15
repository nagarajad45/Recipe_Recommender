import streamlit as st
import pandas as pd
from recommender import RecipeRecommender

@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Indian_Food_Dataset.csv')

def main():
    st.set_page_config(page_title="Recipe Recommender", layout="wide")
    
    st.title("🍲 Recipe Recommender AI")
    st.write("Enter ingredients you have, and we'll recommend delicious recipes!")
    
    try:
        df = load_data()
        recommender = RecipeRecommender()
        recommender.fit(df)
        
        ingredients = st.text_area(
            "Enter your ingredients (comma-separated):",
            placeholder="e.g., rice, tomatoes, onions, ginger, garlic"
        )
        
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=5,
            value=3
        )
        
        if st.button("Get Recommendations"):
            if ingredients:
                with st.spinner("Finding recipes..."):
                    recommendations = recommender.get_recommendations(
                        ingredients,
                        n_recommendations=n_recommendations
                    )
                
                for recipe in recommendations:
                    with st.expander(f"🍳 {recipe['title']} (Match: {recipe['similarity_score']:.2f})"):
                        st.write(f"**Cuisine:** {recipe['cuisine']}")
                        st.write(f"**Cooking Time:** {recipe['cooking_time']} mins")
                        st.write("**Ingredients:**")
                        st.write(recipe['ingredients'])
                        st.write("**Instructions:**")
                        st.write(recipe['instructions'])
            else:
                st.warning("Please enter some ingredients first!")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the dataset file is available.")

if __name__ == "__main__":
    main()

