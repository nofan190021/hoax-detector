import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# download nltk
nltk.download('punkt')
nltk.download('stopwords')

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv('fake_or_real_news.csv')

# =========================
# GABUNG TITLE + TEXT
# =========================
data['content'] = data['title'] + " " + data['text']

# =========================
# UBAH LABEL KE ANGKA
# =========================
data['label'] = data['label'].map({
    'FAKE': 0,
    'REAL': 1
})

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

data['clean'] = data['content'].apply(preprocess)

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean'])
y = data['label']

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_model = LogisticRegression(max_iter=1000)

model = BaggingClassifier(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Akurasi:", acc)

# =========================
# SIMPAN MODEL
# =========================
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

print("✅ model.pkl & vectorizer.pkl berhasil dibuat!")
