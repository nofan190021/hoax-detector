from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

app = Flask(__name__)

# download nltk
nltk.download('punkt')
nltk.download('stopwords')

# load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    if request.method == 'POST':
        text = request.form['text']

        clean = preprocess(text)
        vector = vectorizer.transform([clean])

        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        confidence = round(max(prob) * 100, 2)
        result = "REAL" if pred == 1 else "FAKE"

    return render_template('index.html', result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
