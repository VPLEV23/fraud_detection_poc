import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text
from src.feature_engineering import create_tfidf_features
from src.model import train_logistic_regression, train_svm, evaluate_model
from src.utils import save_confusion_matrix, save_classification_report, save_metrics
from src.preprocess import preprocess_text
from src.bert_model import preprocess_data_for_bert, train_bert_model
from transformers import AutoTokenizer
import json

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    df = pd.read_csv("data/raw/Scam_Not_scam.csv")
except FileNotFoundError:
    print("Data file not found. Please check the file path.")
    sys.exit(1)

df['cleaned_about_me'] = df['about_me'].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_about_me'], df['label'], test_size=0.2, random_state=42)


X_train_tfidf, tfidf = create_tfidf_features(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression
logreg_model = train_logistic_regression(X_train_tfidf, y_train)
evaluate_model(logreg_model, X_test_tfidf, y_test)

# Save Logistic Regression outputs
baseline_folder = "outputs/baseline"
os.makedirs(baseline_folder, exist_ok=True)
save_confusion_matrix(y_test, logreg_model.predict(X_test_tfidf), f"{baseline_folder}/confusion_matrix.png")
save_classification_report(y_test, logreg_model.predict(X_test_tfidf), f"{baseline_folder}/classification_report.txt")
save_metrics(y_test, logreg_model.predict(X_test_tfidf), f"{baseline_folder}/metrics.json")
print(f"Baseline outputs saved in {baseline_folder}")

# Train SVM
svm_model = train_svm(X_train_tfidf, y_train)
evaluate_model(svm_model, X_test_tfidf, y_test)

# Save SVM outputs
svm_folder = "outputs/svm"
os.makedirs(svm_folder, exist_ok=True)
save_confusion_matrix(y_test, svm_model.predict(X_test_tfidf), f"{svm_folder}/confusion_matrix.png")
save_classification_report(y_test, svm_model.predict(X_test_tfidf), f"{svm_folder}/classification_report.txt")
save_metrics(y_test, svm_model.predict(X_test_tfidf), f"{svm_folder}/metrics.json")
print(f"SVM outputs saved in {svm_folder}")

print("Results saved to 'outputs/' folder.")



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = preprocess_data_for_bert(tokenizer, X_train.tolist(), y_train.tolist())
eval_dataset = preprocess_data_for_bert(tokenizer, X_test.tolist(), y_test.tolist())


# Save BERT outputs
bert_output_dir = "outputs/bert"
os.makedirs(bert_output_dir, exist_ok=True)
bert_trainer = train_bert_model(train_dataset, eval_dataset, model_name="bert-base-uncased", output_dir=bert_output_dir)

# Save evaluation results
bert_metrics = bert_trainer.evaluate()
with open(f"{bert_output_dir}/metrics.json", "w") as f:
    json.dump(bert_metrics, f, indent=4)

print(f"BERT outputs saved in {bert_output_dir}")