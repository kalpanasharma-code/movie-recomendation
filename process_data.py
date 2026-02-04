'''
Author: Kalpana Sharma
Email: kanku3140@gmail.com
Date: 2026-Jan-20
'''

import numpy as np 
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
import ast

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Loading data...")
# Load the datasets (you need to place these in the data folder)
try:
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Data files not found in data/ folder.")
    print("Please download the TMDB dataset from:")
    print("https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv")
    print("And place tmdb_5000_movies.csv and tmdb_5000_credits.csv in the data/ folder")
    exit(1)

print("Processing data...")
# Merge datasets
movies = movies.merge(credits, on='title')

# Keep important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew', 'release_date', 'vote_average']]

# Handle missing data and create year column
movies.dropna(inplace=True)
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year

# Convert genres
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Convert cast (keep top 3)
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Extract director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# Remove spaces from names
def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create final dataframe
new_df = movies[['movie_id', 'title', 'tags', 'year', 'vote_average']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

# Apply stemming
ps = PorterStemmer()

def stems(text):
    T = []
    for i in text.split():
        T.append(ps.stem(i))
    return " ".join(T)

new_df['tags'] = new_df['tags'].apply(stems)

print("Creating vectors...")
# Create vectors
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

print("Calculating similarity...")
# Calculate similarity
similarity = cosine_similarity(vector)

print("Saving model files...")
# Save the files
pickle.dump(new_df.to_dict(), open('artifacts/movie_dict.pkl','wb'))
pickle.dump(similarity,open('artifacts/similarity.pkl','wb'))

print("Model files generated successfully!")
print(f"Processed {len(new_df)} movies")
print("Files saved in artifacts/ folder:")
print("- movie_dict.pkl")
print("- similarity.pkl")