# Social-Media-Sentiment

A sentiment analysis application for social media posts (tweets), classifying them into Positive or Negative using machine learning and deep learning models,
with a friendly web interface.


<img width="1200" height="622" alt="web_home" src="https://github.com/user-attachments/assets/91466264-acdf-4ce2-b1c5-8434f6303d3a" />

🚀 Table of Contents

Features

Architecture & Components

Setup & Installation

Usage

Model Details & Performance

Data Processing & Pipeline

Project Structure

Contributing

License

# Features

Binary Sentiment Classification: Classifies tweets as Positive or Negative.

Multiple Model Options:

Random Forest

Naive Bayes

LSTM-based Neural Network

Web App Interface: A Flask-based front end that accepts user text input and returns sentiment.

Text Preprocessing: Implements cleaning, tokenization, stop-word removal, stemming/lemmatization (using NLTK or similar).

Model Serialization: Stores trained model artifacts (e.g. .pkl) for fast inference without re-training from scratch.

# Architecture & Components
```bash
User Input (tweet or text)
     ↓
Flask Web App (app.py)
     ↓
Text Preprocessing Module
     ↓
Vectorization / Tokenization (e.g. CountVectorizer, word embeddings for LSTM)
     ↓
Sentiment Model (RandomForest / NaiveBayes / LSTM)
     ↓
Inference → Output (“Positive” / “Negative”)
```

Models are pre-trained and loaded (via cv.pkl and randomforest.pkl, etc.).

The notebook Tweet Sentiment.ipynb contains experiments, training code, and performance evaluation.

The web app ties everything together in a user-friendly interface.

# Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/Manarabdelgawad/Social-Media-Sentiment.git
cd Social-Media-Sentiment
```

2. Create & activate a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Model files and datasets

Ensure cv.pkl (vectorizer) and randomforest.pkl (or other model files) are present (they are committed in repo).

If you retrain models (via notebook), re-generate and place them accordingly.

5. Run the Flask app
```bash
   python app.py
```

Then open your browser at http://localhost:5000 to interact.

# Usage

On the web UI, type or paste a tweet (or any text) and submit.

The app returns whether the sentiment is Positive or Negative.

(Optionally) You can extend the UI to accept neutral classification, multi-language support, or sentiment score probability.

# Model Details & Performance

Model	Accuracy / Notes

Random Forest	~ 84.34 %

Naive Bayes	~ 83.92 %

LSTM Neural Network	Training accuracy: ~ 94.07 %

# Data Processing & Pipeline

Here’s a deeper look at preprocessing and transformations:

Text Cleaning

Lowercasing

Removing URLs, mentions, hashtags, punctuation, special characters

Handling emoji / emoticon removal or normalization

Tokenization & Stop-word Removal

Using NLTK (or spaCy) to split text into tokens

Removing frequent non-informative words (stop-words)

Stemming / Lemmatization

Reducing words to root forms helps generalization

Feature Extraction / Vectorization

For traditional ML: CountVectorizer, TF-IDF

For LSTM: token sequences, padded to fixed length, embedding layers, etc.

Model Training & Serialization

Training pipelines for each model

Saving (pickle, joblib) models and vectorizers for reuse

Inference Pipeline

Load saved model + vectorizer

Preprocess input text

Vectorize / Embed

Predict class & return result

# Project Structure
```bash
Social-Media-Sentiment/
├── app.py
├── cv.pkl
├── randomforest.pkl
├── requirements.txt
├── Tweet Sentiment.ipynb
├── templates/
│   └── *.html (Flask UI templates)
├── static/
│   └── images/ (static assets)
└── README.md
```
# Contributing

Thank you for your interest! Here’s how you can contribute:

Fork the repository

Create a feature branch (git checkout -b feature/YourIdea)

Make your changes & commit

Push to your fork and open a Pull Request

Describe your changes, provide example screenshots or results, and request reviews

Ensure code is clean, well-commented, tests added if possible

# LICENSE
see the LICENSE file for details.

