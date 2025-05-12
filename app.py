from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle

app = Flask(__name__)

nltk.download('stopwords')
all_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

try:
    with open('cv.pkl', 'rb') as f:
        cv = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("cv.pkl not found. Please ensure the CountVectorizer is saved from the training notebook.")

# Load the trained Random Forest model
try:
    with open('randomforest.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("rf_model.pkl not found. Please ensure the Random Forest model is saved from the training notebook.")

def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = ' '.join(review)
    return review

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')
    
    # Preprocess the review
    processed_review = preprocess_text(review)
    
    review_vector = cv.transform([processed_review]).toarray()
    
    # Predict sentiment using Random Forest
    prediction = model.predict(review_vector)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    confidence = model.predict_proba(review_vector)[0][1]  
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)