from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(texts, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(texts)
    return X, tfidf
