from flask import Flask, request, jsonify, send_from_directory
import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__, static_folder='static')

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-small-uncased")

def get_text_from_html(url):
    print("Fetching URL:", url)
    response = requests.get(url)
    print("Response status code:", response.status_code)
    soup = BeautifulSoup(response.content, 'html.parser')
    texts = []
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ol', 'li', 'span']:
        for element in soup.find_all(tag):
            texts.append(element.get_text())
    return ' '.join(texts)

def generate_embeddings(text):
    print("Generating embedding")
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    print("Tokenized inputs")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Model outputs")
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

@app.route('/generate_embeddings', methods=['GET'])
def generate_embeddings_route():
    print("Generating embedding route")
    url = request.args.get('url')
    print("URL parameter:", url)
    if not url:
        return jsonify({'error': 'URL parameter is required'}), 400
    
    try:
        text = get_text_from_html(url)
        print("Extracted text:", text[:200])  # Печатаем первые 200 символов текста для проверки
        embeddings = generate_embeddings(text)
        print("Generated embeddings")
        return jsonify({'embeddings': embeddings})
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/index.html')
def serve_index():
    print("Generating serve_index()")
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
