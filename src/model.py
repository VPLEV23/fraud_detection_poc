from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def train_svm(X_train, y_train, kernel='linear', random_state=42):
    model = SVC(kernel=kernel, random_state=random_state, probability=True)
    model.fit(X_train, y_train)
    return model