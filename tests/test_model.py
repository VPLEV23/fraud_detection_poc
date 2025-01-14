import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import train_logistic_regression
from sklearn.feature_extraction.text import TfidfVectorizer

def test_train_logistic_regression():
    # Mock dataset
    X_train = ["This is a test", "Another test sentence"]
    y_train = [0, 1]
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = train_logistic_regression(X_train_tfidf, y_train)
    assert model is not None
    assert hasattr(model, "predict")
