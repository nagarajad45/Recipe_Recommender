import streamlit as st
import pandas as pd
from recommender import RecipeRecommender

def load_data():
    return pd.read_csv('Cleaned_Indian_Food_Dataset.csv')

def main():
    st.set_page_config(page_title="Indian Recipe Recommender", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .recipe-card {
            background-color: Black;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .cuisine-tag {
            background-color: #e0e0e0;
            padding: 0.3rem 0.6rem;
            border-radius: 15px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
        }
        .time-tag {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🍲 Indian Recipe Recommender AI")
    st.write("Enter ingredients you have, and we'll recommend delicious Indian recipes!")
    
    # Initialize recommender
    try:
        df = load_data()
        recommender = RecipeRecommender()
        recommender.fit(df)
        
        # User input
        ingredients = st.text_area(
            "Enter your ingredients (comma-separated):",
            height=100,
            placeholder="e.g., rice, tomatoes, onions, ginger, garlic"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            n_recommendations = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=5,
                value=3
            )
        
        if st.button("Get Recommendations", type="primary"):
            if ingredients:
                with st.spinner("Finding the perfect recipes for you..."):
                    recommendations = recommender.get_recommendations(
                        ingredients,
                        n_recommendations=n_recommendations
                    )
                
                st.subheader("📝 Recommended Recipes")
                
                for i, recipe in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recipe-card">
                            <h3>{i}. {recipe['title']}</h3>
                            <span class="cuisine-tag">{recipe['cuisine']}</span>
                            <span class="time-tag">⏱️ {recipe['cooking_time']} mins</span>
                            <p><strong>Match Score:</strong> {recipe['similarity_score']:.2f}</p>
                            <p><strong>Ingredients:</strong> {recipe['ingredients']}</p>
                            <p><strong>Instructions:</strong> {recipe['instructions']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some ingredients first!")
                
    except Exception as e:
        st.error(f"Error loading data or processing recommendations: {str(e)}")
        st.info("Please make sure the dataset file is in the correct location and format.")

if __name__ == "__main__":
    main()


