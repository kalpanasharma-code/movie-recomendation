# Movie Recommender System Using Machine Learning

![Movie Recommender System](demo/6.jpeg)

A sophisticated content-based movie recommendation system built with Python, Streamlit, and machine learning. This application analyzes movie features like genres, keywords, cast, and director to recommend similar movies based on cosine similarity.

## 🎯 Features

- **Content-Based Recommendations**: Uses movie metadata to find similar films
- **Interactive Web Interface**: Beautiful Streamlit UI with movie posters
- **Rich Movie Information**: Displays release year, ratings, and poster images
- **Robust Error Handling**: Graceful fallbacks for API failures
- **Performance Optimized**: Caching and efficient similarity calculations
- **Real-time API Integration**: Fetches movie posters from TMDB API

## 🎬 Demo

### Live Demo
[Click here to run it live on server](https://movie-recommeder-system.herokuapp.com/)

### Screenshots

**Main Interface**
<img src="demo/1.png" alt="Main Interface" width="70%">

**Movie Selection**
<img src="demo/2.png" alt="Movie Selection" width="70%">

**Recommendation Results**
<img src="demo/3.png" alt="Recommendation Results" width="70%">

## 🧠 How It Works

### Recommendation Types Explained

#### 1. Content-Based Filtering
- Uses characteristic information and item attributes
- Analyzes movie features: genres, keywords, cast, director, overview
- Creates feature vectors for similarity comparison
- **Pros**: No cold-start problem for new movies, transparent recommendations
- **Cons**: Limited discovery, may recommend similar items repeatedly

#### 2. Collaborative Filtering
- Based on user-item interactions and ratings
- Finds users with similar preferences
- **Pros**: Can discover diverse content, learns from user behavior
- **Cons**: Cold-start problem, computationally expensive

#### 3. Hybrid Approach
- Combines both content and collaborative methods
- **Used in**: Modern systems like Netflix, Amazon Prime

### Our Implementation

This project uses **Content-Based Filtering** with the following approach:

1. **Data Collection**: TMDB 5000 movies dataset
2. **Feature Engineering**: 
   - Movie overviews → Text processing
   - Genres, keywords → Tag extraction
   - Cast (top 3 actors) → Feature tags
   - Director → Feature tag
3. **Text Processing**:
   - Stemming with NLTK PorterStemmer
   - Stop word removal
   - Vectorization with CountVectorizer (5000 features)
4. **Similarity Calculation**: Cosine similarity between movie vectors
5. **Recommendation**: Top 5 most similar movies

## 📊 Dataset

**Source**: [TMDB Movie Metadata](https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv)

**Files Used**:
- `tmdb_5000_movies.csv` - Movie metadata (4,805 movies)
- `tmdb_5000_credits.csv` - Cast and crew information

**Features Extracted**:
- Movie ID, Title, Overview
- Genres, Keywords
- Cast (Top 3 actors), Director
- Release Date, Vote Average

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kalpanasharma-code/movie-recomendation.git
   cd movie-recomendation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and setup data**
   ```bash
   # Option 1: Automatic download (recommended)
   python process_data.py
   
   # Option 2: Manual download
   # Download from: https://www.kaggle.com/tmdb/tmdb-movie-metadata
   # Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in data/ folder
   # Then run: python process_data.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`

## 🛠️ Project Structure

```
Movie-Recommender-System-Using-Machine-Learning/
├── app.py                          # Main Streamlit application
├── process_data.py                 # Data processing script
├── Movie Recommender System Data Analysis.ipynb  # Jupyter notebook for analysis
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
├── README.md                      # This file
├── data/                          # Dataset folder
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── artifacts/                     # Generated model files
│   ├── movie_dict.pkl            # Processed movie data
│   └── similarity.pkl            # Cosine similarity matrix
├── src/                          # Source utilities
└── demo/                         # Demo screenshots
```

## 🔧 Technical Details

### Core Technologies

- **Backend**: Python, NumPy, Pandas, Scikit-learn
- **Frontend**: Streamlit
- **ML Algorithms**: CountVectorizer, Cosine Similarity
- **Text Processing**: NLTK, PorterStemmer
- **API Integration**: TMDB API for movie posters
- **Data Storage**: Pickle files for model persistence

### Algorithm Pipeline

1. **Data Preprocessing**
   ```python
   # Merge datasets and clean data
   movies = movies.merge(credits, on='title')
   movies.dropna(inplace=True)
   ```

2. **Feature Extraction**
   ```python
   # Extract genres, keywords, cast, director
   movies['genres'] = movies['genres'].apply(convert)
   movies['cast'] = movies['cast'].apply(convert_cast)
   movies['crew'] = movies['crew'].apply(fetch_director)
   ```

3. **Text Processing**
   ```python
   # Stemming and vectorization
   ps = PorterStemmer()
   cv = CountVectorizer(max_features=5000, stop_words='english')
   vector = cv.fit_transform(new_df['tags']).toarray()
   ```

4. **Similarity Calculation**
   ```python
   # Cosine similarity matrix
   similarity = cosine_similarity(vector)
   ```

### Performance Optimizations

- **Caching**: Streamlit cache for API calls (1 hour TTL)
- **Retry Logic**: Automatic retries for failed API requests
- **Error Handling**: Graceful fallbacks for network issues
- **Memory Efficiency**: Efficient vector operations with NumPy

## 📈 Model Performance

- **Dataset Size**: 4,805 movies
- **Feature Dimensions**: 5,000 features per movie
- **Similarity Matrix**: 4,805 × 4,805 cosine similarity matrix
- **Recommendation Speed**: < 1 second per query
- **API Response Time**: < 3 seconds with retry logic

## 🎨 Features in Detail

### Movie Recommendation Process

1. **User selects a movie** from dropdown (4,805 options)
2. **System finds movie index** in the dataset
3. **Calculates similarity scores** with all other movies
4. **Returns top 5 most similar movies**
5. **Displays recommendations** with:
   - Movie poster (from TMDB API)
   - Movie title
   - Release year
   - Average rating

### Error Handling

- **Network Issues**: Automatic retry with exponential backoff
- **API Failures**: Graceful placeholder images
- **Missing Data**: Informative error messages
- **Timeout Protection**: 10-second request timeout

## 🔌 API Integration

### TMDB API Features

- **Movie Poster Fetching**: High-quality poster images
- **Fallback Handling**: Placeholder images for missing posters
- **Rate Limiting**: Built-in retry logic for API limits
- **Error Recovery**: Multiple fallback strategies

### API Configuration

```python
# TMDB API endpoint
url = "https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

# Image configuration
poster_base_url = "https://image.tmdb.org/t/p/w500/"
```

## 🧪 Testing

### Running Tests

```bash
# Test data processing
python process_data.py

# Test app functionality
streamlit run app.py

# Test API connectivity
python -c "import requests; requests.get('https://api.themoviedb.org/3/550?api_key=8265bd1679663a7ea12ac168da84d2e8')"
```

### Test Cases

- ✅ Data loading and processing
- ✅ Model file generation
- ✅ Streamlit app launch
- ✅ Movie recommendation logic
- ✅ API error handling
- ✅ Caching functionality

## 🚀 Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment (Heroku)

1. **Install Heroku CLI**
2. **Login to Heroku**
   ```bash
   heroku login
   ```
3. **Create app**
   ```bash
   heroku create your-app-name
   ```
4. **Deploy**
   ```bash
   git push heroku main
   ```

### Docker Deployment

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments to complex code
- Update documentation for new features
- Test your changes thoroughly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TMDB** for providing the movie dataset and API
- **Streamlit** for the amazing web framework
- **Scikit-learn** for machine learning tools
- **NLTK** for natural language processing
- **Kaggle** for hosting the dataset

## 📞 Contact

**Author**: Kalpana Sharma  
**Email**: kanku3140@gmail.com  
**Updated by**: Kalpana Sharma

---

## 🔮 Future Enhancements

- **Hybrid Recommendation System**: Combine collaborative filtering
- **User Profiles**: Personalized recommendations based on preferences
- **Advanced Features**: Movie trailers, reviews, ratings
- **Performance Optimization**: GPU acceleration for similarity calculations
- **Mobile App**: React Native or Flutter application
- **Real-time Updates**: Live data synchronization with TMDB

## 🐛 Troubleshooting

### Common Issues

1. **"Model files not found"**
   - Solution: Run `python process_data.py` to generate model files

2. **"Connection Error" for posters**
   - Solution: Check internet connection, API will retry automatically

3. **"Data files not found"**
   - Solution: Ensure CSV files are in the `data/` folder or run automatic download

4. **"Streamlit not found"**
   - Solution: Install with `pip install streamlit`

### Debug Mode

Enable debug logging:

```bash
streamlit run app.py --logger.level debug
```

---

**⭐ If you like this project, please give it a star on GitHub!**#   m o v i e - r e c o m e n d a t i o n 
 
 