import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from transformers import AutoTokenizer
from src.bert_model import preprocess_data_for_bert

def test_preprocess_data_for_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = ["This is a test", "Another sentence"]
    labels = [0, 1]

    dataset = preprocess_data_for_bert(tokenizer, texts, labels)


    expected_features = ["input_ids", "attention_mask", "labels"]
    for feature in expected_features:
        assert feature in dataset.features
