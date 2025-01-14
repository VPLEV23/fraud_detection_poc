import re
import emoji

def normalize_emojis(text):
    text = emoji.demojize(text)
    text = text.replace(":T_button:", "T").replace(":G_button:", "G")
    text = text.replace(":M_button:", "M").replace(":R_button:", "R")
    return text

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.strip()
    return text

def preprocess_text(text):
    text = normalize_emojis(text)
    text = clean_text(text)
    return text
