import streamlit as st
import pickle
import difflib
import pandas as pd
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="CineMatch - Movie Recommender", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: #0f1419;
    }
    
    /* Card styling */
    .movie-card {
        background: #1a1f2e;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid #2a3241;
        height: 100%;
    }
    
    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border-color: #3a4251;
    }
    
    /* Title styling */
    .movie-title {
        font-size: 20px;
        font-weight: 600;
        color: #e8eaed;
        margin-bottom: 10px;
        min-height: 50px;
    }
    
    /* Subtitle styling */
    .movie-subtitle {
        font-size: 13px;
        color: #9aa0a6;
        margin-bottom: 5px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 30px 0;
        color: #e8eaed;
    }
    
    /* Search box */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #3a4251;
        padding: 10px;
        font-size: 16px;
        background: #1a1f2e;
        color: #e8eaed;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: #2a3241;
        color: #e8eaed;
        border-radius: 8px;
        padding: 15px;
        font-size: 18px;
        font-weight: 600;
        border: 1px solid #3a4251;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #3a4251;
        border-color: #4a5261;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1f2e;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: #e8eaed;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #4a5568;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2a3241;
        border-radius: 8px;
        font-weight: 600;
        color: #e8eaed;
    }
    
    /* Image styling */
    .movie-poster {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        width: 100%;
        object-fit: cover;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# TMDB API Configuration
# API key is loaded from Streamlit secrets for deployment
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = ""  # Fallback for local development
    st.warning("⚠️ TMDB API key not found in secrets. Movie posters will not load.")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Function to get movie poster from TMDB
@st.cache_data(ttl=3600)
def get_movie_poster(title, year=None):
    """
    Fetch movie poster from TMDB API
    Args:
        title: Movie title
        year: Release year (optional, helps with accuracy)
    Returns:
        URL of the movie poster or placeholder
    """
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/300x450?text=No+API+Key"
    
    try:
        # Search for the movie
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "year": year
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        
        return "https://via.placeholder.com/300x450?text=No+Poster+Available"
    
    except Exception as e:
        return "https://via.placeholder.com/300x450?text=Error+Loading"

# Function to get additional movie details
@st.cache_data(ttl=3600)
def get_movie_details(title):
    """
    Fetch additional movie details from TMDB
    Returns rating, overview, release date
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            movie = data['results'][0]
            return {
                'rating': movie.get('vote_average', 'N/A'),
                'overview': movie.get('overview', 'No overview available'),
                'release_date': movie.get('release_date', 'Unknown'),
                'popularity': movie.get('popularity', 0)
            }
        
        return None
    
    except Exception as e:
        return None

# Load the saved model
@st.cache_resource
def load_model():
    with open('movie_recommender.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data

# Load model components
try:
    model_data = load_model()
    similarity = model_data['similarity']
    movies_data = model_data['movies_data']
except:
    st.error("⚠️ Model file not found! Please ensure 'movie_recommender.pkl' is in the same directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎭 About CineMatch")
    st.markdown("""
        <div style='background: #2a3241; padding: 15px; border-radius: 8px; color: #e8eaed; border: 1px solid #3a4251;'>
        Discover your next favorite movie using AI-powered recommendations based on:
        <ul>
            <li>🎬 Genres</li>
            <li>⭐ Cast</li>
            <li>🎥 Director</li>
            <li>🔑 Keywords</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Database Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Movies", f"{len(movies_data):,}")
    with col2:
        st.metric("Features", "5")
    
    st.markdown("---")
    
    # Popular movies
    st.markdown("### 🔥 Trending Searches")
    popular_movies = ["Iron Man", "Avatar", "The Dark Knight", "Inception", "Interstellar"]
    for movie in popular_movies:
        if st.button(f"🎬 {movie}", key=movie, use_container_width=True):
            st.session_state.movie_input = movie

# Header
st.markdown("""
    <div class='main-header'>
        <h1 style='font-size: 48px; margin-bottom: 10px; font-weight: 600;'>🎬 CineMatch</h1>
        <p style='font-size: 18px; opacity: 0.7; color: #9aa0a6;'>Your Personal Movie Recommendation Engine</p>
    </div>
""", unsafe_allow_html=True)

# Main content area with white background
with st.container():
    st.markdown("<div style='background: #1a1f2e; padding: 30px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 1px solid #2a3241;'>", unsafe_allow_html=True)
    
    # Search section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        movie_name = st.text_input(
            "🔍 Search for a movie",
            placeholder="Type a movie name... (e.g., Iron Man, Batman, Avatar)",
            key="movie_input" if "movie_input" not in st.session_state else None,
            label_visibility="collapsed"
        )
    
    with col2:
        num_recommendations = st.selectbox(
            "Results",
            options=[5, 10, 15, 20, 30],
            index=1,
            label_visibility="collapsed"
        )
    
    # Get Recommendations button
    search_button = st.button("🎯 Get Recommendations", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Recommendation logic
if search_button or (hasattr(st.session_state, 'movie_input') and st.session_state.movie_input):
    movie_to_search = movie_name if movie_name else st.session_state.get('movie_input', '')
    
    if movie_to_search:
        with st.spinner('🎬 Finding perfect matches...'):
            list_of_all_titles = movies_data['title'].tolist()
            find_close_match = difflib.get_close_matches(movie_to_search, list_of_all_titles, n=1, cutoff=0.6)
            
            if find_close_match:
                close_match = find_close_match[0]
                
                # Success message
                st.success(f"✅ Found: **{close_match}**")
                
                # Get recommendations
                index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
                similarity_score = list(enumerate(similarity[index_of_the_movie]))
                sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
                
                # Display recommendations header
                st.markdown("---")
                st.markdown("<h2 style='text-align: center; color: #e8eaed;'>🎯 Recommended For You</h2>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Create columns for grid layout
                cols_per_row = 3
                
                for i, movie in enumerate(sorted_similar_movies[1:num_recommendations+1]):
                    if i % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[i % cols_per_row]:
                        index = movie[0]
                        title = movies_data[movies_data.index == index]['title'].values[0]
                        similarity_pct = movie[1] * 100
                        
                        # Get movie details
                        movie_info = movies_data[movies_data.index == index].iloc[0]
                        
                        # Get poster and additional details
                        poster_url = get_movie_poster(title)
                        extra_details = get_movie_details(title)
                        
                        # Display movie poster
                        st.image(poster_url, use_container_width=True)
                        
                        # Movie card
                        st.markdown(f"""
                            <div class='movie-card'>
                                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;'>
                                    <div style='color: #9aa0a6; font-size: 16px; font-weight: 600;'>#{i+1}</div>
                                    <div style='background: #2a3241; 
                                                color: #e8eaed; padding: 6px 12px; border-radius: 6px; 
                                                font-weight: 600; font-size: 13px; border: 1px solid #3a4251;'>
                                        {similarity_pct:.1f}%
                                    </div>
                                </div>
                                <div class='movie-title'>{title}</div>
                        """, unsafe_allow_html=True)
                        
                        # Add rating if available
                        if extra_details and extra_details['rating'] != 'N/A':
                            st.markdown(f"""
                                <div style='margin-bottom: 10px;'>
                                    <span style='background: #3a4251; color: #fbbf24; padding: 4px 10px; 
                                                 border-radius: 6px; font-size: 13px; font-weight: 600; border: 1px solid #4a5261;'>
                                        ⭐ {extra_details['rating']}/10
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                                <div class='movie-subtitle'>
                                    <strong>🎭 Genres:</strong> {movie_info['genres'][:50]}{'...' if len(str(movie_info['genres'])) > 50 else ''}
                                </div>
                                <div class='movie-subtitle'>
                                    <strong>🎥 Director:</strong> {movie_info['director']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar for similarity
                        st.progress(similarity_pct / 100)
                        
                        # Expander for more details
                        if extra_details:
                            with st.expander("📖 More Info"):
                                st.write(f"**Release Date:** {extra_details['release_date']}")
                                st.write(f"**Overview:** {extra_details['overview'][:200]}...")
            else:
                st.error("❌ Movie not found! Please try another name or check spelling.")
                
                # Show suggestions
                suggestions = difflib.get_close_matches(movie_to_search, list_of_all_titles, n=5, cutoff=0.3)
                if suggestions:
                    st.info(f"💡 Did you mean: {', '.join(suggestions[:3])}?")
    else:
        st.warning("⚠️ Please enter a movie name to get started!")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #9aa0a6; padding: 20px; 
                background: #1a1f2e; border-radius: 8px; margin-top: 40px; border: 1px solid #2a3241;'>
        <p style='margin: 0; font-size: 14px;'>Made with ❤️ using Streamlit & Machine Learning</p>
        <p style='margin: 5px 0 0 0; opacity: 0.7; font-size: 13px;'>🎬 Powered by TMDB Dataset | 🤖 AI-Driven Recommendations</p>
    </div>
""", unsafe_allow_html=True)
