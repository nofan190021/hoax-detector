from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

app = Flask(__name__)

# =========================
# NLTK SETUP (ANTI ERROR)
# =========================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =========================
# LOAD MODEL
# =========================
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

# =========================
# ROUTE
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    try:
        if request.method == 'POST':
            text = request.form.get('text')

            # VALIDASI INPUT
            if not text or text.strip() == "":
                return render_template(
                    'index.html',
                    result="Silakan masukkan teks terlebih dahulu",
                    confidence=0
                )

            # PREPROCESS
            clean = preprocess(text)

            # VECTORIZER
            vector = vectorizer.transform([clean])

            # PREDIKSI
            pred = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            confidence = round(max(prob) * 100, 2)
            result = "REAL" if pred == 1 else "FAKE"

    except Exception:
        return render_template(
            'index.html',
            result="Terjadi kesalahan pada sistem",
            confidence=0
        )

    return render_template('index.html', result=result, confidence=confidence)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
