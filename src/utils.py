import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_confusion_matrix(y_test, y_pred, output_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_path)
    plt.close()

def save_classification_report(y_test, y_pred, output_path):
    report = classification_report(y_test, y_pred)
    with open(output_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(report)

def save_metrics(y_test, y_pred, output_path):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_0": precision_score(y_test, y_pred, pos_label=0),
        "recall_0": recall_score(y_test, y_pred, pos_label=0),
        "precision_1": precision_score(y_test, y_pred, pos_label=1),
        "recall_1": recall_score(y_test, y_pred, pos_label=1),
        "f1_score_0": f1_score(y_test, y_pred, pos_label=0),
        "f1_score_1": f1_score(y_test, y_pred, pos_label=1),
    }
    with open(output_path, 'w') as file:
        json.dump(metrics, file, indent=4)